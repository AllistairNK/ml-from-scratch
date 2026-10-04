# B2: Python basics

Only the Python this course actually uses. Type every example into a Python shell (`.venv\Scripts\python`).
Anything after a `#` is a **comment**, a note for humans that Python ignores.

---

## 1. Variables and numbers

A variable is a name that holds a value. `=` means "store the value on the right under the name on the left".

```python
weight = 150
smoothness = 9.5
name = "apple"          # text is called a "string" and goes in quotes
```

| Operator | Meaning | Example | Result |
|----------|---------|---------|--------|
| `+ - *` | add, subtract, multiply | `7 * 2` | `14` |
| `/` | divide | `7 / 2` | `3.5` |
| `//` | divide and round down | `7 // 2` | `3` |
| `%` | remainder | `7 % 2` | `1` |
| `**` | power | `3 ** 2` | `9` |
| `==` | is equal? (gives True/False) | `3 == 3` | `True` |
| `!=` `<` `>` `<=` `>=` | comparisons | `3 > 5` | `False` |

`=` **stores** a value; `==` **asks** whether two values are equal. Mixing them up is a classic bug.

`x += 1` is shorthand for `x = x + 1`. Likewise `x -= 0.1` and `x /= n`. The course uses `self.w -= self.lr * dw` all the time.

## 2. Lists and indexes

A list holds several values in order:

```python
weights = [150, 170, 140]
weights[0]        # 150   (positions start at 0!)
weights[2]        # 140
weights[-1]       # 140   (negative = count from the end)
len(weights)      # 3     (how many items)
weights[0:2]      # [150, 170]   a slice: from 0 up to, but NOT including, 2
weights.append(160)   # add to the end -> [150, 170, 140, 160]
```

![Slicing](../00_foundations/diagrams/slicing.svg)

*A slice start:stop includes the start but stops just BEFORE the stop index.*

The position number is called the **index**. Python counts from 0 because the index means "how many steps from the start".

## 3. If statements

Run code only when a condition is true. **The indentation (4 spaces) is part of the code.** It shows what belongs inside the `if`.

```python
smoothness = 8
if smoothness > 6:
    print("apple")
else:
    print("orange")
```

## 4. For loops

Repeat code once for each item:

```python
for w in [150, 170, 140]:
    print(w * 2)          # prints 300, then 340, then 280
```

`range(n)` gives the numbers 0, 1, …, n−1. It's handy for "repeat n times":

```python
for i in range(3):
    print(i)              # 0, 1, 2
```

`_` is used as the loop variable when you don't need it: `for _ in range(1000):` just means "do this 1000 times".

**List comprehension** is a one-line loop that builds a list. You'll see it a lot:

```python
[w * 2 for w in [150, 170, 140]]      # [300, 340, 280]
[w for w in [150, 170, 140] if w > 145]   # [150, 170]  (keep only some)
```

Read it as "**w × 2**, for each **w** in the list".

## 5. Functions

A function is a named recipe: it takes inputs and **returns** an output.

```python
def average(values):              # def = define; values = the input
    total = sum(values)
    return total / len(values)    # return = hand the answer back

average([2, 4, 9])                # 5.0
```

- The names in brackets in the definition (`values`) are the **parameters**. The actual values you pass in (`[2, 4, 9]`) are the **arguments**.
- **Default values** make an argument optional: `def __init__(self, k=3):` means k is 3 unless you say otherwise.
- **Keyword arguments** name the input when calling: `KNN(k=5)`, `X.sum(axis=0)`.

## 6. Dictionaries and tuples

A **dictionary** stores values under names (keys). The neural network returns its gradients this way:

```python
grads = {"W1": 0.5, "b1": -0.1}
grads["W1"]        # 0.5
```

A **tuple** is a fixed list in round brackets, e.g. an array's shape `(3, 2)`. A function can return several values as a tuple,
and you can unpack them into separate names:

```python
n_samples, n_features = (3, 2)   # n_samples = 3, n_features = 2
```

That's what `n_samples, n_features = X.shape` does in the course code.

## 7. Classes: the pattern every model uses

A **class** bundles data and the functions that use it. Every model in this course looks like this:

```python
class MajorityGuesser:
    """Always predicts the most common label it saw during training."""

    def __init__(self, name="guesser"):     # runs when you create the object
        self.name = name                   # store a setting on the object

    def fit(self, y):                      # learn from data
        self.answer = max(set(y), key=y.count)   # the most frequent label
        return self

    def predict(self, how_many):           # use what was learned
        return [self.answer] * how_many

model = MajorityGuesser()       # create an object: this runs __init__
model.fit([0, 1, 0, 0])         # call a method: it learns that 0 is most common
model.predict(3)                # [0, 0, 0]
model.answer                    # 0  (an attribute stored by fit)
```

The key ideas:

- **`class Name:`** defines the blueprint. **`Name()`** makes an object from it.
- **Methods** are functions defined inside the class. Their **first parameter is always `self`**.
- **`self`** is the object itself. `self.answer = ...` **saves** a value on the object, so another method (here `predict`) can read it later.
  Forgetting `self.` is the most common class bug: a plain `answer = ...` disappears when the method ends.
- **`__init__`** (two underscores on each side) runs automatically when the object is created. Use it to store the settings.
- **`return self`** at the end of `fit` lets you chain calls: `MajorityGuesser().fit(y).predict(3)`.
- When you call `model.fit(y)`, Python passes `model` in as `self` automatically, so you don't write it.

## 8. Imports

```python
import numpy as np      # load the NumPy library and call it np
np.array([1, 2, 3])     # use something from it with np.
```

## 9. Reading error messages

When code crashes, Python prints a **traceback**. Read it **from the bottom up**:

```
Traceback (most recent call last):
  File "example.py", line 30, in <module>
    model = KNN(k=k).fit(X_train, y_train)
  File "my_practice.py", line 9, in fit
    self.X_train = X_trian
NameError: name 'X_trian' is not defined
```

1. **The last line** says what went wrong: `NameError`, a name Python doesn't know. It's a typo: `X_trian`.
2. **The line above it** is the exact place in **your** file: `my_practice.py`, line 9.
3. Everything higher up is the chain of calls that led there. You can usually ignore it.

The errors you'll meet most:

| Error | Usually means |
|-------|---------------|
| `NotImplementedError` | the practice file still has a `raise NotImplementedError(...)` line, so that part isn't written yet |
| `NameError` | a typo, or you forgot `self.` |
| `IndentationError` | the spaces at the start of a line are wrong (use 4 spaces per level) |
| `SyntaxError` | a missing `:` or bracket, or a typo in the structure |
| `ValueError: operands could not be broadcast` | the array shapes don't fit together (see [F1](../00_foundations/F1_numpy.md)) |
| `IndexError` | an index past the end of the array or list |
| `AttributeError: ... has no attribute 'w'` | you used `self.w` before setting it (forgot to store it in `fit`?) |
| `AssertionError` | a check in example.py failed: your code runs, but gives the wrong answer |

**The best debugging tool is `print`.** Print values and shapes to see what's actually happening:

```python
print("dists shape:", dists.shape)
```

## Check yourself

1. What does `[10, 20, 30, 40][1:3]` give?
2. What is `[x ** 2 for x in range(4)]`?
3. In the `MajorityGuesser` class, why does `predict` use `self.answer` and not just `answer`?
4. You get `AttributeError: 'KNN' object has no attribute 'X_train'`. What did you probably forget?

<details>
<summary>Answers</summary>

1. `[20, 30]`: the indexes 1 and 2 (the stop index 3 is not included).
2. `[0, 1, 4, 9]`.
3. `answer` would only exist inside `fit` and vanish when `fit` ends. `self.answer` is stored on the object, so `predict` can read it.
4. To store `self.X_train = X` in `fit` (or `fit` was never called).

</details>

**Next:** [B3: Your first model](B3_first_model.md)
