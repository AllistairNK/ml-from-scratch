# B3: Your first model, end to end

Let's put all the words together. We'll build the simplest possible classifier, test it, and see the
same `fit` / `predict` pattern that every algorithm in this course uses.

Type this in a Python shell as you read.

## 1. The data

Six fruit. **Features**: weight and smoothness. **Labels**: 0 = apple, 1 = orange.

```python
import numpy as np

X = np.array([[150, 9],    # sample 0: apple
              [170, 3],    # sample 1: orange
              [140, 8],    # sample 2: apple
              [180, 2],    # sample 3: orange
              [155, 7],    # sample 4: apple
              [165, 4]])   # sample 5: orange
y = np.array([0, 1, 0, 1, 0, 1])

X.shape    # (6, 2) -> 6 samples, 2 features
y.shape    # (6,)   -> one label per sample
```

## 2. Split into training and test data

The model learns from the first 4 fruit, and we keep the last 2 hidden to test it fairly.

```python
X_train, y_train = X[:4], y[:4]     # rows 0-3
X_test,  y_test  = X[4:], y[4:]     # rows 4-5
```

`X[:4]` is a slice: the rows from the start up to (not including) index 4.

## 3. The idea: a "nearest average" model

To learn: compute the **average fruit of each class** (its centroid). To predict: say whichever class average the new fruit is **closer** to.

Here are the steps by hand, so you can see each word in action:

```python
apples  = X_train[y_train == 0]      # mask: keep the rows whose label is class 0
oranges = X_train[y_train == 1]
apples                               # [[150, 9], [140, 8]]

apple_centre  = apples.mean(axis=0)   # average down each column -> [145. , 8.5]
orange_centre = oranges.mean(axis=0)  # [175. , 2.5]
```

- `y_train == 0` gives `[True, False, True, False]`, a mask saying which rows are apples.
- `axis=0` averages **down** the rows, giving one average per feature (column).

Now predict the new fruit `[155, 7]` (test sample 0):

```python
new = X_test[0]                                         # [155, 7]
dist_apple  = np.sqrt(((new - apple_centre) ** 2).sum())   # about 10.1
dist_orange = np.sqrt(((new - orange_centre) ** 2).sum())  # about 20.5
```

It's closer to the apple centre, so the **prediction** is class 0 (apple). That's correct!

## 4. The same thing as a model class

The course packages every algorithm like this, so the steps above become `fit` and `predict`:

```python
class NearestCentroid:
    def fit(self, X, y):
        self.classes = np.unique(y)                       # [0, 1]
        self.centres = np.array([X[y == c].mean(axis=0) for c in self.classes])
        return self

    def predict(self, X):
        preds = []
        for x in X:                                       # one sample at a time
            dists = np.sqrt(((self.centres - x) ** 2).sum(axis=1))   # distance to each centre
            preds.append(self.classes[np.argmin(dists)])  # index of the smallest distance
        return np.array(preds)

model = NearestCentroid().fit(X_train, y_train)    # training
y_pred = model.predict(X_test)                     # prediction -> [0, 1]
```

- `fit` **learns parameters** (the centres) and stores them on `self`.
- `predict` **uses** them on new data.
- `np.argmin` gives the **index** of the smallest value: 0 if the first centre is closest, 1 if the second.

## 5. How good is it? Accuracy

Compare the predictions with the true labels:

```python
y_pred == y_test             # [True, True]  (a mask of correct answers)
np.mean(y_pred == y_test)    # 1.0  -> 100% accuracy
```

`np.mean` of True/False values counts True as 1 and False as 0, so it gives the fraction that are correct.

## 6. What you just did

| Step | ML word | Code |
|------|---------|------|
| Measured fruit | dataset, samples, features, labels | `X`, `y` |
| Hid some fruit | train/test split | `X[:4]`, `X[4:]` |
| Computed class averages | training / fitting, learning parameters | `model.fit(X_train, y_train)` |
| Guessed new fruit | prediction | `model.predict(X_test)` |
| Counted correct guesses | accuracy | `np.mean(y_pred == y_test)` |

Every algorithm in this course follows this exact shape. Only the inside of `fit` and `predict` changes.
KNN (Week 1) is a close cousin of this model: instead of comparing with class averages, it compares with the nearest individual fruit.

## Check yourself

1. Predict the fruit `[172, 3]` by hand with the centres above. Which class?
2. Why do we test on fruit the model didn't train on?
3. What does `X_train[y_train == 1]` return?

<details>
<summary>Answers</summary>

1. The distance to the apple centre (145, 8.5) is about 27.6, and to the orange centre (175, 2.5) about 3.0, so it's **class 1 (orange)**.
2. To check that it learned the pattern rather than memorising the answers. Scoring on the training fruit would be like marking an exam whose answers you already gave the student.
3. The training rows whose label is 1 (the oranges): `[[170, 3], [180, 2]]`.

</details>

**Next:** the lab ([practice.py](practice.py)), then the [Foundations unit](../00_foundations/README.md).
