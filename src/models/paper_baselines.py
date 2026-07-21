"""Re-implemented prior-work baselines (Sec 4.B, [C2]).

Each is run on THIS corpus under the shared subject-wise protocol.  Provenance:

  little2009   : Little et al. 2009, dysphonia measures + SVM(RBF).
                 RE-IMPLEMENT. Original code n/a; features = jitter/shimmer/HNR/F0.
  tsanas2012   : Tsanas et al. 2012, dysphonia + feature selection (LASSO) + RF.
                 RE-IMPLEMENT of the "feature-selection + ensemble" recipe.
  vasquez2018  : Vasquez-Correa et al. 2018, DisVoice articulation/phonation + NN.
                 RE-IMPLEMENT; DisVoice vector -> MLP (CNN in original; see caveat).
  moro2019     : Moro-Velazquez et al. 2019, MFCC + GMM likelihood-ratio.
                 RE-IMPLEMENT of the GMM-UBM-style phonemic modelling (simplified).
  neurovoz2024 : mel-spectrogram + ResNet-18 -> handled by the CNN family (deep.py).

Caveats (reported in docs/related_work.md): original papers used different corpora,
tasks and, for CNN/GMM, more elaborate front-ends; numbers here are re-implementations
on THIS dataset, not the authors' originals.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _proba(clf, Xte):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(Xte)[:, 1]
    d = clf.decision_function(Xte)
    return 1.0 / (1.0 + np.exp(-d))


def little2009_fit_predict(Xtr, ytr, Xte, meta):
    seed = meta["seed"]
    clf = make_pipeline(StandardScaler(),
                        SVC(kernel="rbf", probability=True, class_weight="balanced",
                            random_state=seed))
    clf.fit(Xtr, ytr)
    return _proba(clf, Xte)


def tsanas2012_fit_predict(Xtr, ytr, Xte, meta):
    seed = meta["seed"]
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    # LASSO feature selection
    try:
        lasso = LassoCV(cv=3, random_state=seed, max_iter=5000).fit(Xtr_s, ytr)
        sel = np.abs(lasso.coef_) > 1e-8
        if sel.sum() == 0:
            sel = np.ones(Xtr_s.shape[1], dtype=bool)
    except Exception:
        sel = np.ones(Xtr_s.shape[1], dtype=bool)
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                random_state=seed, n_jobs=-1)
    rf.fit(Xtr_s[:, sel], ytr)
    return rf.predict_proba(Xte_s[:, sel])[:, 1]


def vasquez2018_fit_predict(Xtr, ytr, Xte, meta):
    seed = meta["seed"]
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                      early_stopping=True, random_state=seed))
    clf.fit(Xtr, ytr)
    return _proba(clf, Xte)


def moro2019_fit_predict(Xtr, ytr, Xte, meta):
    """MFCC-sequence GMM likelihood-ratio.  Xtr/Xte are [n, C, T] MFCC sequences."""
    from sklearn.mixture import GaussianMixture
    seed = meta["seed"]
    # frame-level: transpose to [n, T, C] then stack frames per class
    def frames(X, idx):
        sub = X[idx]                                   # [m, C, T]
        return np.concatenate([s.T for s in sub], axis=0)  # [m*T, C]
    Xtr3 = Xtr if Xtr.ndim == 3 else Xtr.reshape(len(Xtr), -1, 1)
    Xte3 = Xte if Xte.ndim == 3 else Xte.reshape(len(Xte), -1, 1)
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    n_comp = 8
    sc = StandardScaler()
    allframes = np.concatenate([f.T for f in Xtr3], axis=0)
    sc.fit(allframes)
    def gmm(idx):
        F = sc.transform(frames(Xtr3, idx))
        g = GaussianMixture(n_components=min(n_comp, max(1, len(F) // 50)),
                            covariance_type="diag", random_state=seed, reg_covar=1e-3)
        g.fit(F); return g
    g_pd, g_hc = gmm(pos), gmm(neg)
    probs = []
    for s in Xte3:
        F = sc.transform(s.T)
        llr = g_pd.score(F) - g_hc.score(F)            # avg log-lik ratio
        probs.append(1.0 / (1.0 + np.exp(-llr)))
    return np.asarray(probs)


REGISTRY = {
    "little2009": little2009_fit_predict,
    "tsanas2012": tsanas2012_fit_predict,
    "vasquez2018": vasquez2018_fit_predict,
    "moro2019": moro2019_fit_predict,
}
