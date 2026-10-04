# F5: ML basics: the vocabulary

**Learning objectives.** After this lecture you can:

- classify a problem as supervised (regression or classification) or unsupervised
- explain the fit/predict pattern and why we hold out test data
- explain overfitting vs underfitting and the bias–variance trade-off
- pick a sensible evaluation metric

---

## 1. Kinds of problems

| Type | Given | Goal | Algorithms in this course |
|------|-------|------|---------------------------|
| **Regression** (supervised) | features `X`, numeric target `y` | predict a number | Linear Regression |
| **Classification** (supervised) | features `X`, class labels `y` | predict a category | KNN, Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Neural Network |
| **Clustering** (unsupervised) | features `X` only | find groups | K-Means |
| **Dimensionality reduction** (unsupervised) | features `X` only | compress to fewer features | PCA |
| **Sequence modelling** | sequences of vectors | mix information between positions | Attention |

- **Feature:** an input column (e.g. age, income).
- **Label / target:** what we want to predict.
- **Sample / example:** one row.

## 2. The fit / predict pattern

Every model in this course has the same interface (borrowed from scikit-learn):

```python
model = SomeModel(hyperparameters)   # choose settings
model.fit(X_train, y_train)          # learn parameters from data
y_pred = model.predict(X_test)       # use them on new data
```

- **Parameters** are learned by `fit` (weights, centroids, tree splits).
- **Hyperparameters** are chosen by you before training (k in KNN, the learning rate, the max depth).

## 3. Train / validation / test

A model can **memorise** the training data. So we always measure it on data it hasn't seen:

- **Training set** is used to fit the parameters.
- **Validation set** is used to choose the hyperparameters (or use cross-validation).
- **Test set** is touched **once**, at the end, for an honest final score.

```python
idx = rng.permutation(len(X))       # shuffle first! (the data is often sorted by class)
X, y = X[idx], y[idx]
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]
```

**Data leakage** means test information sneaks into training (e.g. computing the
standardisation mean over the full dataset). It gives fake-good scores. Fit any preprocessing on
the training set only.

## 4. Overfitting, underfitting, bias and variance

| | Train score | Test score | Cause | Fix |
|---|---|---|---|---|
| **Underfitting** (high bias) | bad | bad | the model is too simple | a more flexible model, more features |
| **Good fit** | good | good | | |
| **Overfitting** (high variance) | great | bad | the model memorised the noise | more data, a simpler model, regularisation, ensembles |

Knobs you'll meet: a **small k** in KNN overfits and a large k underfits. A **deep tree** overfits.
A **Random Forest** reduces variance by averaging many trees.

## 5. Loss functions vs metrics

- The **loss** is what training minimises. It must be smooth enough to differentiate: MSE, cross-entropy.
- A **metric** is what you report. It's human-readable: accuracy, precision, RMSE.

**Regression metrics:** MSE $= \frac1n\sum (\hat y - y)^2$; RMSE $=\sqrt{\text{MSE}}$ (in the units of y); $R^2$.

**Classification metrics** (for the positive class):

|  | predicted + | predicted − |
|---|---|---|
| **actually +** | TP | FN |
| **actually −** | FP | TN |

- **Accuracy** $= \frac{TP+TN}{\text{all}}$. This misleads with imbalanced classes: 99% accuracy is easy if 99% of the samples are negative.
- **Precision** $= \frac{TP}{TP+FP}$: "when I say positive, how often am I right?"
- **Recall** $= \frac{TP}{TP+FN}$: "of all the real positives, how many did I catch?"
- **F1** is the harmonic mean of the two.

## 6. Feature scaling

Distance-based (KNN, K-Means) and gradient-based (linear/logistic regression, neural networks)
methods are sensitive to feature scale. **Standardise**: `X = (X - mean) / std`, with the mean and
std computed on the **training** set. Trees and Naive Bayes don't need it.

## Check yourself

1. Predicting house prices: which type of problem is it? Grouping customers with no labels?
2. Train accuracy 100%, test accuracy 62%. What is happening, and name two fixes.
3. A fraud dataset is 0.5% fraud. A model predicts "not fraud" for everything. What are its accuracy and its recall?
4. Is the learning rate a parameter or a hyperparameter? And the weights?

<details>
<summary>Answers</summary>

1. Regression; clustering.
2. Overfitting. Fixes: more data, a simpler model or regularisation (e.g. limit the tree depth), or an ensemble.
3. Accuracy is 99.5%, but recall is 0%: it catches no fraud at all. That's why accuracy alone is dangerous.
4. The learning rate is a hyperparameter; the weights are parameters.

</details>

**Next:** the lab [practice.py](practice.py) (the NumPy drills), then start [Week 1: KNN](../01_knn/LECTURE.md).
