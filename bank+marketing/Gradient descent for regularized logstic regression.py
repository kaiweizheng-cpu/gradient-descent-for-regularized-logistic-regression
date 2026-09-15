import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split


class LogisticLoss:
    def __init__(self, lam=0.01):
        self.lam = lam

    def predict(self, X, w, b):
        return expit(X @ w + b)

    def compute_loss(self, X, w, b, y):
        p = self.predict(X, w, b)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        ce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        reg = 0.5 * self.lam * np.dot(w, w)
        return ce + reg

    def compute_grad(self, X, w, b, y):
        n = X.shape[0]
        p = self.predict(X, w, b)
        dw = X.T @ (p - y) / n + self.lam * w
        db = np.sum(p - y) / n
        return dw, db


class GradientDescent:
    def __init__(self, lr):
        self.lr = lr

    def step(self, w, b, dw, db):
        w -= self.lr * dw
        b -= self.lr * db
        return w, b


def load_bank_data(path):
    df = pd.read_csv(path, sep=";")

    y = (df["y"] == "yes").astype(int).to_numpy()

    num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    cat_cols = ["job", "marital", "education", "default", "housing",
                "loan", "contact", "month", "poutcome"]

    X_num = df[num_cols].to_numpy(dtype=float)
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True).to_numpy(dtype=float)

    feature_names = num_cols + list(pd.get_dummies(df[cat_cols], drop_first=True).columns)
    return X_num, X_cat, y, feature_names


def accuracy(X, w, b, y):
    p = expit(X @ w + b)
    return np.mean((p >= 0.5) == y)


if __name__ == "__main__":
    X_num, X_cat, y, feature_names = load_bank_data("bank/bank-full.csv")

    X = np.hstack([X_num, X_cat])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # standardize numeric features using training-set statistics only
    num_w = X_num.shape[1]
    mu = X_tr[:, :num_w].mean(axis=0)
    std = X_tr[:, :num_w].std(axis=0)
    X_tr[:, :num_w] = (X_tr[:, :num_w] - mu) / std
    X_te[:, :num_w] = (X_te[:, :num_w] - mu) / std

    n, d = X_tr.shape
    print(f"train: n={n}, test: n={X_te.shape[0]}, attributes: d={d}")

    w = np.zeros(d)
    b = 0.0
    lam = 0.01             # L2 regularization strength
    loss_fn = LogisticLoss(lam=lam)
    opt = GradientDescent(lr=0.5)

    for i in range(2000):
        loss = loss_fn.compute_loss(X_tr, w, b, y_tr)
        dw, db = loss_fn.compute_grad(X_tr, w, b, y_tr)
        w, b = opt.step(w, b, dw, db)

        if i % 200 == 0:
            acc = accuracy(X_tr, w, b, y_tr)
            print(f"iter {i:4d} | loss={loss:.4f} | train acc={acc:.4f}")

    print(f"\nfinish: train loss={loss_fn.compute_loss(X_tr, w, b, y_tr):.4f}")
    print(f"train accuracy={accuracy(X_tr, w, b, y_tr):.4f}")
    print(f"test  accuracy={accuracy(X_te, w, b, y_te):.4f}")
    print(f"bias b={b:.4f}")

    print("\ntop |w| features:")
    order = np.argsort(-np.abs(w))
    for k in order[:10]:
        print(f"  {feature_names[k]:24s} = {w[k]:+.4f}")
