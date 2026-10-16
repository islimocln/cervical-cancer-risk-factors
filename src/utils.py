"""Notebook'larda ortak kullanılan veri yükleme, temizleme ve eksik veri fonksiyonları."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (IterativeImputer için gerekli)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "risk_factors_cervical_cancer.csv"

TARGET = "Biopsy"
LEAKY = ["Hinselmann", "Schiller", "Citology"]  # biyopsiyle aynı muayenenin testleri → girdi olamaz

# 01_veri_inceleme.ipynb'deki karar tablosunda "TUT" denen 15 öznitelik
FEATURES = [
    "Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies",
    "Smokes (years)", "Smokes (packs/year)",
    "Hormonal Contraceptives (years)", "IUD (years)",
    "STDs (number)", "STDs:condylomatosis", "STDs:syphilis", "STDs:HIV",
    "Dx:Cancer", "Dx:CIN", "Dx:HPV",
]
NUMERIC = FEATURES[:9]  # sayısal/sayım sütunları (log1p uygulanır); kalan 6'sı 0/1

# Her eksiklik grubunu temsil eden sütun (grup içindeki sütunlar hep birlikte eksik)
MISSING_GROUPS = {
    "Partner":    "Number of sexual partners",
    "İlk ilişki": "First sexual intercourse",
    "Gebelik":    "Num of pregnancies",
    "Sigara":     "Smokes (years)",
    "HK":         "Hormonal Contraceptives (years)",
    "RİA":        "IUD (years)",
    "CYBH":       "STDs (number)",
}


def load_raw(path=DATA_PATH):
    """Ham veriyi okur; '?' işaretleri NaN olur."""
    return pd.read_csv(path, na_values="?")


def clean(df):
    """01'de bulunan veri hatalarını düzeltir. Doldurma (imputasyon) YAPMAZ.

    - İlk cinsel ilişki yaşı > şimdiki yaş olan değerler → NaN (imkânsız değer)
    - CYBH soruları cevapsızken 0 yazılmış 'STDs: Number of diagnosis' → NaN (gizli eksik)
    """
    df = df.copy()
    df.loc[df["First sexual intercourse"] > df["Age"], "First sexual intercourse"] = np.nan
    df.loc[df["STDs"].isna(), "STDs: Number of diagnosis"] = np.nan
    return df


def missing_indicators(df):
    """Her eksiklik grubu için 0/1 gösterge tablosu döndürür."""
    return pd.DataFrame({g: df[c].isna().astype(int) for g, c in MISSING_GROUPS.items()}, index=df.index)


class GroupMissingIndicator(BaseEstimator, TransformerMixin):
    """Her eksiklik grubu için tek bir 0/1 gösterge üretir (Pipeline içinde kullanılır).

    sklearn'ün add_indicator=True seçeneği her sütuna ayrı gösterge ekler; HK, RİA ve CYBH
    hep birlikte eksik olduğu için bu neredeyse aynı birçok sütun demek. Bu sınıf onun yerine
    grup başına bir gösterge verir. Öğrenilen bir parametre yoktur, sızıntı riski taşımaz.
    """

    def fit(self, X, y=None):
        self.groups_ = {g: c for g, c in MISSING_GROUPS.items() if c in X.columns}
        return self

    def transform(self, X):
        return np.column_stack([X[c].isna().to_numpy(dtype=float) for c in self.groups_.values()])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"eksik_{g}" for g in self.groups_])


def build_pipeline(model, imputer="median", indicator=False, group=False):
    """Ön işleme + model Pipeline'ı kurar. Her şey fit sırasında yalnızca eğitim verisinden öğrenilir.

    Adımlar:  log1p (sayısal) → imputasyon → standartlaştırma  [+ grup eksiklik göstergeleri] → model

    imputer:   "median" (SimpleImputer) veya "mice" (IterativeImputer, çoklu imputasyonun tekli sürümü)
    indicator: True ise sklearn her eksikli sütuna ayrı bir gösterge ekler
    group:     True ise GroupMissingIndicator ile grup başına tek gösterge eklenir
    """
    log = ColumnTransformer(
        [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), NUMERIC)],
        remainder="passthrough", verbose_feature_names_out=False,
    )
    if imputer == "median":
        imp = SimpleImputer(strategy="median", add_indicator=indicator)
    elif imputer == "mice":
        imp = IterativeImputer(max_iter=10, random_state=0, add_indicator=indicator)
    else:
        raise ValueError(imputer)
    veri = Pipeline([("log", log), ("imp", imp), ("scale", StandardScaler())])
    feats = FeatureUnion([("veri", veri), ("grup", GroupMissingIndicator())]) if group else veri
    return Pipeline([("ozellik", feats), ("model", model)])


def corrected_ttest(diff, n_train, n_test):
    """Tekrarlı çapraz doğrulamada iki modeli karşılaştırmak için düzeltilmiş t-testi
    (Nadeau & Bengio, 2003). Katlar bağımsız olmadığı için normal t-testi fazla iyimserdir.

    diff: her kattaki skor farkı (model B − model A). Döndürür: (ortalama fark, p-değeri)
    """
    diff = np.asarray(diff, dtype=float)
    k = len(diff)
    var = diff.var(ddof=1)
    if var == 0:
        return diff.mean(), 1.0
    t = diff.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return diff.mean(), 2 * stats.t.sf(abs(t), k - 1)


def _em_mvn(X, max_iter=1000, tol=1e-8):
    """Eksik veriyle çok değişkenli normal dağılımın ortalama ve kovaryansını EM ile tahmin eder."""
    n, p = X.shape
    obs = ~np.isnan(X)
    mu = np.nanmean(X, axis=0)
    S = np.diag(np.nanvar(X, axis=0))
    patterns, inv = np.unique(obs, axis=0, return_inverse=True)

    for _ in range(max_iter):
        sum_x = np.zeros(p)
        sum_xx = np.zeros((p, p))
        for k, o in enumerate(patterns):
            rows = X[inv.ravel() == k]
            m = ~o
            Xk = rows.copy()
            C = np.zeros((p, p))
            if m.any():
                B = S[np.ix_(m, o)] @ np.linalg.inv(S[np.ix_(o, o)])
                Xk[:, m] = mu[m] + (rows[:, o] - mu[o]) @ B.T
                C[np.ix_(m, m)] = S[np.ix_(m, m)] - B @ S[np.ix_(o, m)]
            sum_x += Xk.sum(axis=0)
            sum_xx += Xk.T @ Xk + len(rows) * C
        mu_new = sum_x / n
        S_new = sum_xx / n - np.outer(mu_new, mu_new)
        converged = np.max(np.abs(mu_new - mu)) < tol and np.max(np.abs(S_new - S)) < tol
        mu, S = mu_new, S_new
        if converged:
            break
    return mu, S


def little_mcar_test(df):
    """Little (1988) MCAR testi.

    H0: veri tamamen rastgele eksik (MCAR). Küçük p-değeri → MCAR reddedilir.
    Varsayım: değişkenler yaklaşık çok değişkenli normal.
    Döndürür: (ki-kare istatistiği, serbestlik derecesi, p-değeri, desen sayısı)
    """
    X = df.to_numpy(dtype=float)
    p = X.shape[1]
    mu, S = _em_mvn(X)
    obs = ~np.isnan(X)
    patterns, inv = np.unique(obs, axis=0, return_inverse=True)

    d2, dof = 0.0, 0
    for k, o in enumerate(patterns):
        if not o.any():
            continue
        rows = X[inv.ravel() == k][:, o]
        diff = rows.mean(axis=0) - mu[o]
        d2 += len(rows) * diff @ np.linalg.inv(S[np.ix_(o, o)]) @ diff
        dof += o.sum()
    dof -= p
    return d2, dof, stats.chi2.sf(d2, dof), len(patterns)


def wilson_ci(k, n, alpha=0.05):
    """Oran için Wilson güven aralığı (küçük örneklem ve 0/n durumunda da çalışır)."""
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half
