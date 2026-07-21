"""Classic ML estimators (Sec 4.A) + factory.  All support predict_proba."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def make_estimator(kind: str, seed: int, class_weight=True):
    cw = "balanced" if class_weight else None
    if kind == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                   class_weight=cw, random_state=seed)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=400, class_weight=cw,
                                      random_state=seed, n_jobs=-1)
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                             random_state=seed, n_jobs=-1)
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, class_weight=cw, random_state=seed)
    if kind == "knn":
        return KNeighborsClassifier(n_neighbors=7)
    raise ValueError(f"unknown estimator: {kind}")


def make_fit_predict(kind: str, class_weight=True):
    """Return a runner-compatible fit_predict fn (train-only scaler + estimator)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fit_predict(Xtr, ytr, Xte, meta):
        est = make_estimator(kind, meta["seed"], class_weight)
        clf = make_pipeline(StandardScaler(), est)
        clf.fit(Xtr, ytr)
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(Xte)[:, 1]
        import numpy as np
        d = clf.decision_function(Xte)
        return 1.0 / (1.0 + np.exp(-d))

    return fit_predict
