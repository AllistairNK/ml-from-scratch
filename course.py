"""Local course app: lectures, an in-browser editor, check runner and progress tracker.

    python course.py            # then open http://127.0.0.1:8000
    python course.py --port 8001 --no-browser

Progress is saved to progress.json in the repo. Use the app's "Commit progress" button
(or `git commit`) to save it, together with your practice files, into git history.
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "course")
PROGRESS = os.path.join(ROOT, "progress.json")
SYLLABUS = os.path.join(STATIC, "syllabus.json")
WORK_FILES = ["my_practice.py", "day2_blank.py"]
RUN_TARGET = re.compile(r"^(solution|explained|my_practice|day2_blank|attempts/[\w\-]+)$")
SPACED_DAYS = 3
LOCK = threading.Lock()


def python_exe():
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python", "venv/Scripts/python.exe", "venv/bin/python"):
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            return path
    return sys.executable


def today():
    return dt.date.today().isoformat()


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- syllabus & steps

def load_syllabus():
    with open(SYLLABUS, encoding="utf-8") as f:
        units = json.load(f)["units"]
    for unit in units:
        unit["steps"] = steps_for(unit)
    return units


def steps_for(unit):
    if unit["kind"] == "foundations":
        steps = [{"id": "read:" + r["file"], "label": "Read " + r["title"], "auto": False} for r in unit["readings"]]
        steps.append({"id": "drills", "label": "NumPy drills pass", "auto": True})
        return steps
    if unit["kind"] == "explain":
        steps = []
        for r in unit["readings"]:
            steps.append({"id": "read:" + r["file"], "label": "Notes written: " + r["title"], "auto": False})
            steps.append({"id": "explain:" + r["file"], "label": "Can explain " + r["title"] + " in 2 min", "auto": False})
        return steps
    target = unit.get("target_minutes", 20)
    return [
        {"id": "lecture", "label": "Read the lecture", "auto": False},
        {"id": "tutorial", "label": "Answered the tutorial questions", "auto": False},
        {"id": "explained", "label": "Studied explained.py and ran the example", "auto": False},
        {"id": "practice", "label": "Practice file passes", "auto": True},
        {"id": "blank", "label": "Blank-file rewrite passes", "auto": True},
        {"id": "timed", "label": "Timed attempt passes", "auto": True},
        {"id": "spaced", "label": "Spaced repeat ({}+ days later) passes".format(SPACED_DAYS), "auto": True},
        {"id": "ready", "label": "Interview-ready: under {} min, no peeking".format(target), "auto": True},
    ]


def unit_by_id(unit_id):
    for unit in load_syllabus():
        if unit["id"] == unit_id:
            return unit
    raise ValueError("unknown unit: " + str(unit_id))


def unit_files(unit):
    folder = os.path.join(ROOT, unit["id"])
    attempts_dir = os.path.join(folder, "attempts")
    attempts = sorted(f[:-3] for f in os.listdir(attempts_dir) if f.endswith(".py")) if os.path.isdir(attempts_dir) else []
    return {
        "my_practice": os.path.exists(os.path.join(folder, "my_practice.py")),
        "day2_blank": os.path.exists(os.path.join(folder, "day2_blank.py")),
        "attempts": ["attempts/" + a for a in attempts],
        "has_code": os.path.exists(os.path.join(folder, "example.py")),
    }


# ---------------------------------------------------------------- progress store

def empty_progress():
    return {"version": 1, "units": {}, "active_attempt": None, "time": {}, "runs": []}


def load_progress():
    if not os.path.exists(PROGRESS):
        return empty_progress()
    with open(PROGRESS, encoding="utf-8") as f:
        data = json.load(f)
    base = empty_progress()
    base.update(data)
    return base


def save_progress(data):
    tmp = PROGRESS + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, PROGRESS)


def unit_progress(data, unit_id):
    return data["units"].setdefault(unit_id, {"checks": {}, "attempts": []})


def mark(data, unit_id, step, done=True):
    checks = unit_progress(data, unit_id)["checks"]
    if done:
        checks.setdefault(step, today())
    else:
        checks.pop(step, None)


# ---------------------------------------------------------------- paths

def safe_path(rel, write=False):
    """Resolve a repo-relative path, refusing anything outside the course units."""
    rel = rel.replace("\\", "/").lstrip("/")
    if ".." in rel.split("/"):
        raise ValueError("'..' is not allowed in paths")
    full = os.path.realpath(os.path.join(ROOT, rel))
    if not full.startswith(os.path.realpath(ROOT) + os.sep):
        raise ValueError("path outside the repo")
    parts = rel.split("/")
    unit_ids = {u["id"] for u in load_syllabus()}
    top_level_ok = len(parts) == 1 and parts[0] in ("README.md", "LOG.md")
    if not (top_level_ok or parts[0] in unit_ids):
        raise ValueError("not a course file: " + rel)
    if write:
        name = parts[-1]
        allowed = (
            rel == "LOG.md"
            or (len(parts) == 2 and name in WORK_FILES + ["LOG.md"])
            or (len(parts) == 2 and parts[0] == "explain_only" and name.endswith(".md"))
            or (len(parts) == 3 and parts[1] == "attempts" and re.match(r"^[\w\-]+\.py$", name))
        )
        if not allowed:
            raise ValueError("this file is read-only in the app: " + rel)
    elif not rel.endswith((".md", ".py")):
        raise ValueError("only .md and .py files can be read")
    return full


def setup_header(unit_id):
    """The import/setup lines of practice.py (everything before the first def/class)."""
    with open(os.path.join(ROOT, unit_id, "practice.py"), encoding="utf-8") as f:
        lines = f.read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() and not line.startswith("#"))
    end = next(i for i, line in enumerate(lines) if line.startswith(("def ", "class ")))
    return "\n".join(lines[start:end]).rstrip() + "\n"


def create_work_file(unit_id, kind):
    unit = unit_by_id(unit_id)
    folder = os.path.join(ROOT, unit_id)
    if kind == "practice":
        path = os.path.join(folder, "my_practice.py")
        if not os.path.exists(path):
            shutil.copyfile(os.path.join(folder, "practice.py"), path)
        return "my_practice"
    if kind == "blank":
        path = os.path.join(folder, "day2_blank.py")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write("# {}: blank-file rewrite. Write the whole implementation from memory.\n"
                        "# Same class/function names as solution.py. Peek only when stuck.\n\n".format(unit["title"])
                        + setup_header(unit_id))
        return "day2_blank"
    if kind == "attempt":
        os.makedirs(os.path.join(folder, "attempts"), exist_ok=True)
        name, n = today(), 1
        while os.path.exists(os.path.join(folder, "attempts", name + ".py")):
            n += 1
            name = "{}_{}".format(today(), n)
        with open(os.path.join(folder, "attempts", name + ".py"), "w", encoding="utf-8", newline="\n") as f:
            f.write("# {}: timed attempt, started {}. No peeking!\n\n".format(unit["title"], now().replace("T", " "))
                    + setup_header(unit_id))
        return "attempts/" + name
    raise ValueError("unknown work file kind: " + kind)


# ---------------------------------------------------------------- running checks

def run_checks(unit_id, target, peeked=False):
    unit = unit_by_id(unit_id)
    if not RUN_TARGET.match(target):
        raise ValueError("bad target: " + target)
    if not os.path.exists(os.path.join(ROOT, unit_id, target + ".py")):
        raise ValueError("file does not exist yet: " + target + ".py")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    started = dt.datetime.now()
    try:
        proc = subprocess.run([python_exe(), os.path.join(unit_id, "example.py"), target], cwd=ROOT, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        output = proc.stdout.decode("utf-8", errors="replace")
        passed = proc.returncode == 0 and "All checks passed." in output
    except subprocess.TimeoutExpired:
        output, passed = "Timed out after 300 seconds (infinite loop?).", False
    seconds = (dt.datetime.now() - started).total_seconds()

    events = []
    with LOCK:
        data = load_progress()
        data["runs"] = (data["runs"] + [{"at": now(), "unit": unit_id, "target": target, "passed": passed}])[-300:]
        if passed:
            events = record_pass(data, unit, target, peeked)
        save_progress(data)
    return {"output": output, "passed": passed, "seconds": round(seconds, 1), "events": events}


def record_pass(data, unit, target, peeked):
    unit_id, events = unit["id"], []

    def done(step, message):
        if step not in unit_progress(data, unit_id)["checks"]:
            mark(data, unit_id, step)
            events.append(message)

    if unit["kind"] == "foundations":
        if target == "my_practice":
            done("drills", "NumPy drills complete!")
        return events
    if target == "my_practice":
        done("practice", "Practice step complete.")
    elif target == "day2_blank":
        done("blank", "Blank-file rewrite complete.")
    elif target.startswith("attempts/"):
        active = data.get("active_attempt") or {}
        if active.get("unit") != unit_id or active.get("file") != target:
            events.append("Passed, but this attempt wasn't timed (start one with 'New timed attempt').")
            return events
        started = dt.datetime.fromisoformat(active["started"])
        minutes = round((dt.datetime.now() - started).total_seconds() / 60, 1)
        peeked = peeked or active.get("peeked", False)
        attempts = unit_progress(data, unit_id)["attempts"]
        earlier = [a for a in attempts if a.get("passed")]
        attempts.append({"date": today(), "file": target, "minutes": minutes, "peeked": peeked, "passed": True})
        data["active_attempt"] = None
        events.append("Timed attempt passed in {} min{}.".format(minutes, " (peeked)" if peeked else ""))
        done("timed", "Timed attempt step complete.")
        first = min((a["date"] for a in earlier), default=None)
        if first and (dt.date.today() - dt.date.fromisoformat(first)).days >= SPACED_DAYS:
            done("spaced", "Spaced repeat complete.")
        if not peeked and minutes <= unit.get("target_minutes", 20):
            done("ready", "Interview-ready! Under the {}-minute target.".format(unit.get("target_minutes", 20)))
    return events


# ---------------------------------------------------------------- git

GIT_PATHS = ["progress.json", "LOG.md", ":(glob)*/LOG.md", ":(glob)*/my_practice.py", ":(glob)*/day2_blank.py",
             ":(glob)*/attempts/*.py", ":(glob)explain_only/*.md"]


def git(*args):
    proc = subprocess.run(["git"] + list(args), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout.decode("utf-8", errors="replace").strip()


def git_status():
    code, out = git("status", "--porcelain", "--untracked-files=all", "--", *GIT_PATHS)
    if code != 0:
        return {"ok": False, "error": out, "changes": []}
    return {"ok": True, "changes": [line[3:] for line in out.splitlines() if line.strip()]}


def git_commit(message):
    status = git_status()
    if not status["ok"]:
        return {"ok": False, "output": status["error"]}
    if not status["changes"]:
        return {"ok": True, "output": "Nothing new to save."}
    code, out = git("add", "--", *GIT_PATHS)
    if code != 0:
        return {"ok": False, "output": out}
    message = message.strip() or "Save course progress ({})".format(today())
    body = "\n".join("- " + c for c in status["changes"])
    code, out = git("commit", "-m", message, "-m", "Files:\n" + body)
    return {"ok": code == 0, "output": out}


# ---------------------------------------------------------------- HTTP

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC, **kwargs)

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/state":
                units = load_syllabus()
                for unit in units:
                    unit["files"] = unit_files(unit) if unit["kind"] == "algorithm" or unit["kind"] == "foundations" else {}
                return self.send_json({"units": units, "progress": load_progress(), "today": today(),
                                       "git": git_status(), "python": os.path.relpath(python_exe(), ROOT)})
            if url.path == "/api/file":
                full = safe_path(query["path"])
                if not os.path.exists(full):
                    return self.send_json({"error": "not found"}, 404)
                with open(full, encoding="utf-8") as f:
                    return self.send_json({"path": query["path"], "content": f.read()})
        except (ValueError, KeyError) as e:
            return self.send_json({"error": str(e)}, 400)
        if url.path.startswith("/api/"):
            return self.send_json({"error": "unknown endpoint"}, 404)
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = self.read_json()
            if url.path == "/api/file":
                full = safe_path(body["path"], write=True)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w", encoding="utf-8", newline="\n") as f:
                    f.write(body["content"])
                return self.send_json({"ok": True})
            if url.path == "/api/workfile":
                target = create_work_file(body["unit"], body["kind"])
                if body["kind"] == "attempt":
                    with LOCK:
                        data = load_progress()
                        data["active_attempt"] = {"unit": body["unit"], "file": target, "started": now(), "peeked": False}
                        save_progress(data)
                return self.send_json({"target": target})
            if url.path == "/api/reset":
                path = os.path.join(ROOT, body["unit"], "my_practice.py")
                unit_by_id(body["unit"])
                shutil.copyfile(os.path.join(ROOT, body["unit"], "practice.py"), path)
                return self.send_json({"ok": True})
            if url.path == "/api/run":
                return self.send_json(run_checks(body["unit"], body["target"], bool(body.get("peeked"))))
            if url.path == "/api/check":
                unit_by_id(body["unit"])
                with LOCK:
                    data = load_progress()
                    mark(data, body["unit"], body["step"], bool(body["done"]))
                    save_progress(data)
                return self.send_json({"ok": True})
            if url.path == "/api/peek":
                with LOCK:
                    data = load_progress()
                    if data.get("active_attempt"):
                        data["active_attempt"]["peeked"] = True
                        save_progress(data)
                return self.send_json({"ok": True})
            if url.path == "/api/attempt/cancel":
                with LOCK:
                    data = load_progress()
                    data["active_attempt"] = None
                    save_progress(data)
                return self.send_json({"ok": True})
            if url.path == "/api/heartbeat":
                with LOCK:
                    data = load_progress()
                    day = data["time"].setdefault(today(), {})
                    key = body.get("unit") or "dashboard"
                    day[key] = day.get(key, 0) + 1
                    save_progress(data)
                return self.send_json({"ok": True})
            if url.path == "/api/commit":
                return self.send_json(git_commit(body.get("message", "")))
        except (ValueError, KeyError, StopIteration) as e:
            return self.send_json({"error": str(e)}, 400)
        return self.send_json({"error": "unknown endpoint"}, 404)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not os.path.exists(PROGRESS):
        save_progress(empty_progress())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = "http://127.0.0.1:{}".format(args.port)
    print("ML From Scratch course running at", url)
    print("Checks run with:", os.path.relpath(python_exe(), ROOT))
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Remember to commit your progress!")


if __name__ == "__main__":
    main()
