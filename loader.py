"""Load an implementation file by path, so each example.py can run against any attempt.

    python 01_knn/example.py                       # runs solution.py
    python 01_knn/example.py practice              # runs practice.py
    python 01_knn/example.py attempts/2026-10-04   # runs a dated attempt
"""
import importlib.util
import os
import sys


def load_module(path, name=None):
    name = name or os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_impl(example_file, default="solution"):
    folder = os.path.dirname(os.path.abspath(example_file))
    target = sys.argv[1] if len(sys.argv) > 1 else default
    if not target.endswith(".py"):
        target += ".py"
    if not os.path.isabs(target):
        target = os.path.join(folder, target)
    print("Running against", os.path.relpath(target, folder))
    return load_module(target, "impl")
