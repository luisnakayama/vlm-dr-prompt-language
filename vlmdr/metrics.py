"""Métricas de acurácia diagnóstica.

Com `groups` (patient_id), as imagens do mesmo paciente são tratadas como um agrupamento: bootstrap por paciente
para os ICs, DeLong agrupado (Obuchowski 1997) e McNemar agrupado (Durkalski 2003). Sem `groups`, cada linha é
independente (Wilson, DeLong e McNemar usuais).
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

import config


def wilson(k, n, alpha=config.ALPHA):
    if n == 0:
        return np.nan, np.nan, np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, c - h, c + h


def _codes(groups):
    return pd.factorize(np.asarray(groups))[0]


def proportion(x, groups=None, n=config.BOOTSTRAP, seed=config.SEED, alpha=config.ALPHA):
    """Proporção de acertos em `x` (0/1): Wilson sem `groups`; bootstrap por paciente com `groups`."""
    x = np.asarray(x).astype(int)
    if groups is None or len(x) == 0:
        return wilson(int(x.sum()), len(x), alpha)
    g = _codes(groups)
    k, m = np.bincount(g, weights=x), np.bincount(g)
    draw = np.random.default_rng(seed).integers(0, len(m), (n, len(m)))
    boot = k[draw].sum(axis=1) / m[draw].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return x.mean(), lo, hi


def sens_spec(y, yhat, groups=None):
    """y, yhat em {0,1}. Resposta inválida já deve ter sido convertida (análise principal: inválida = encaminhar)."""
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    tp, fn = int(((y == 1) & (yhat == 1)).sum()), int(((y == 1) & (yhat == 0)).sum())
    tn, fp = int(((y == 0) & (yhat == 0)).sum()), int(((y == 0) & (yhat == 1)).sum())
    g = None if groups is None else np.asarray(groups)
    se, se_lo, se_hi = proportion(yhat[y == 1], None if g is None else g[y == 1])
    sp, sp_lo, sp_hi = proportion(1 - yhat[y == 0], None if g is None else g[y == 0])
    out = {"n": len(y), "positivos": tp + fn, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
           "sens": se, "sens_lo": se_lo, "sens_hi": se_hi, "spec": sp, "spec_lo": sp_lo, "spec_hi": sp_hi}
    if g is not None:
        out["pacientes"] = len(np.unique(g))
    return out


def _resample(groups, rng):
    """Sorteador de reamostras por paciente: cada paciente sorteado entra com todas as suas imagens."""
    g = _codes(groups)
    order, sizes = np.argsort(g, kind="stable"), np.bincount(g)
    starts = np.cumsum(sizes) - sizes

    def draw():
        c = rng.integers(0, len(sizes), len(sizes))
        lens = sizes[c]
        offs = np.cumsum(lens) - lens
        return order[np.arange(lens.sum()) - np.repeat(offs, lens) + np.repeat(starts[c], lens)]
    return draw


def bootstrap_ci(fn, y, s, groups=None, n=config.BOOTSTRAP, seed=config.SEED, alpha=config.ALPHA):
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(y))
    draw = _resample(groups, rng)
    vals = []
    for _ in range(n):
        i = draw()
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], s[i]))
    return np.quantile(vals, [alpha / 2, 1 - alpha / 2])


def auroc(y, s, ci=True, groups=None):
    est = roc_auc_score(y, s)
    lo, hi = bootstrap_ci(roc_auc_score, y, s, groups) if ci else (np.nan, np.nan)
    return est, lo, hi


def auprc(y, s, ci=True, groups=None):
    est = average_precision_score(y, s)
    lo, hi = bootstrap_ci(average_precision_score, y, s, groups) if ci else (np.nan, np.nan)
    return est, lo, hi


# ------------------------------------------------------------------------------------------ grau ICDR
ICDR = [0, 1, 2, 3, 4]


def _kappa(weights):
    return lambda y, yhat: cohen_kappa_score(y, yhat, weights=weights, labels=ICDR)


def kappa(y, yhat, weights="quadratic", groups=None):
    """Kappa de Cohen (ponderado: 'quadratic' | 'linear' | None) entre grau real e previsto, com IC por bootstrap."""
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    fn = _kappa(weights)
    lo, hi = bootstrap_ci(fn, y, yhat, groups)
    return fn(y, yhat), lo, hi


def paired_diff(fn, y, a, b, groups=None, n=config.BOOTSTRAP, seed=config.SEED, alpha=config.ALPHA):
    """fn(y, a) - fn(y, b) com IC e p bilateral por bootstrap pareado (por paciente com `groups`)."""
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    draw = _resample(np.arange(len(y)) if groups is None else groups, np.random.default_rng(seed))
    vals = []
    for _ in range(n):
        i = draw()
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], a[i]) - fn(y[i], b[i]))
    vals = np.asarray(vals)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))
    return fn(y, a) - fn(y, b), lo, hi, p


# ------------------------------------------------------------------------------------------ DeLong
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def _delong_cov(y, scores):
    y = np.asarray(y).astype(int)
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01).reshape(k, k) / m + np.cov(v10).reshape(k, k) / n
    return aucs, cov


def _obuchowski_cov(y, scores, groups):
    """Covariância das AUROCs com imagens agrupadas por paciente (Obuchowski 1997)."""
    y = np.asarray(y).astype(int)
    g = _codes(groups)
    gp, gn = g[y == 1], g[y == 0]
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n, k, n_cl = pos.shape[1], neg.shape[1], scores.shape[0], g.max() + 1
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    v_pos = (tz[:, :m] - tx) / n            # fração de negativos abaixo de cada positivo
    v_neg = 1 - (tz[:, m:] - ty) / m        # fração de positivos acima de cada negativo
    m_i, n_i = np.bincount(gp, minlength=n_cl), np.bincount(gn, minlength=n_cl)
    V10 = np.array([np.bincount(gp, weights=r, minlength=n_cl) for r in v_pos])
    V01 = np.array([np.bincount(gn, weights=r, minlength=n_cl) for r in v_neg])
    aucs = V10.sum(axis=1) / m
    D10, D01 = V10 - m_i * aucs[:, None], V01 - n_i * aucs[:, None]
    f = n_cl / (n_cl - 1)
    S10, S01, S11 = f * D10 @ D10.T / m, f * D01 @ D01.T / n, f * D10 @ D01.T
    return aucs, S10 / m + S01 / n + (S11 + S11.T) / (m * n)


def delong(y, s1, s2, groups=None):
    """Teste de DeLong para duas AUROCs pareadas (agrupado por paciente com `groups`). Retorna (auc1, auc2, diferença, p)."""
    sc = np.vstack([s1, s2]).astype(float)
    aucs, cov = _delong_cov(y, sc) if groups is None else _obuchowski_cov(y, sc, groups)
    d = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    p = 2 * stats.norm.sf(abs(d) / np.sqrt(var)) if var > 0 else np.nan
    return aucs[0], aucs[1], d, p


def mcnemar(correct_a, correct_b, groups=None):
    """McNemar exato (binomial) para acertos pareados; com `groups`, McNemar agrupado por paciente (Durkalski 2003)."""
    a, b = np.asarray(correct_a).astype(bool), np.asarray(correct_b).astype(bool)
    if groups is not None:
        d = np.bincount(_codes(groups), weights=(a & ~b).astype(int) - (~a & b).astype(int))
        den = (d ** 2).sum()
        return 1.0 if den == 0 else stats.chi2.sf(d.sum() ** 2 / den, 1)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    if n01 + n10 == 0:
        return 1.0
    return stats.binomtest(n01, n01 + n10, 0.5).pvalue


# ---------------------------------------------------------------- limiar fixado em metade, avaliado na outra
def split_half(n, seed=config.SEED, groups=None):
    """Metades A e B; com `groups`, as imagens do mesmo paciente ficam na mesma metade."""
    if groups is None:
        idx = np.random.default_rng(seed).permutation(n)
        return idx[: n // 2], idx[n // 2:]
    g = _codes(groups)
    pats = np.random.default_rng(seed).permutation(g.max() + 1)
    in_a = np.isin(g, pats[: len(pats) // 2])
    return np.flatnonzero(in_a), np.flatnonzero(~in_a)


def threshold_for_sens(y, s, target):
    pos = np.sort(np.asarray(s)[np.asarray(y) == 1])
    k = int(np.floor((1 - target) * len(pos)))
    return pos[k] if len(pos) else np.nan


def operating_point(y, s, target_sens, seed=config.SEED, groups=None):
    """Limiar para `target_sens` na metade A; sensibilidade, especificidade e triagem na metade B."""
    y, s = np.asarray(y).astype(int), np.asarray(s).astype(float)
    a, b = split_half(len(y), seed, groups)
    thr = threshold_for_sens(y[a], s[a], target_sens)
    yhat = (s[b] >= thr).astype(int)
    r = sens_spec(y[b], yhat, None if groups is None else np.asarray(groups)[b])
    r.update({"alvo_sens": target_sens, "limiar": thr,
              "triagem_sem_leitura_%": 100 * (1 - yhat.mean()),
              "perdidos_por_1000": 1000 * r["fn"] / len(b)})
    return r


def predictive_values(sens, spec, prev):
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    npv = spec * (1 - prev) / (spec * (1 - prev) + (1 - sens) * prev)
    return ppv, npv


def sens_by_grade(df, yhat_col="yhat", group_col="patient_id"):
    """Sensibilidade por grau real; `group_col=None` trata cada imagem como independente."""
    grp = (lambda d: None) if group_col is None else (lambda d: d[group_col].values)
    rows = []
    for (g, name), d in df[df.referable == 1].groupby(["dr_icdr", "grade"]):
        k, n = int(d[yhat_col].sum()), len(d)
        p, lo, hi = proportion(d[yhat_col].values, grp(d))
        rows.append({"icdr": g, "grau": name, "n": n, "encaminhados": k, "sens": p, "lo": lo, "hi": hi})
    d = df[(df.referable == 1) & (df.macular_edema == 1)]
    if len(d):
        p, lo, hi = proportion(d[yhat_col].values, grp(d))
        rows.append({"icdr": "EM", "grau": "com edema macular", "n": len(d), "encaminhados": int(d[yhat_col].sum()),
                     "sens": p, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def fmt(p, lo, hi, pct=True):
    f = 100 if pct else 1
    d = 1 if pct else 3
    return f"{p * f:.{d}f} ({lo * f:.{d}f}–{hi * f:.{d}f})"
