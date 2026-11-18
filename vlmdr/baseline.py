"""Linha de base supervisionada: embeddings congelados + regressão logística.

Três cenários, sempre sem paciente em comum entre treino e teste:
  brset_crossfit  : BRSET, validação cruzada em 5 partes por paciente. Treino = todas as imagens graduáveis
                    do BRSET fora da parte (com e sem diabetes); predição = imagens da coorte naquela parte.
  mbrset_crossfit : mBRSET, mesma lógica com todas as fotos maculares graduáveis (os dois olhos).
  brset_to_mbrset : treino em todo o BRSET, teste no mBRSET (validação externa entre câmeras).
A regularização (C) é escolhida por validação interna agrupada por paciente.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from . import cohort as C
from . import embeddings as E

CS = [1e-3, 1e-2, 1e-1, 1.0]


def _fit(X, y, groups, seed=config.SEED):
    inner = list(GroupKFold(n_splits=3).split(X, y, groups))
    clf = make_pipeline(StandardScaler(),
                        LogisticRegressionCV(Cs=CS, cv=inner, scoring="roc_auc", max_iter=5000, random_state=seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return clf.fit(X, y)


def _crossfit(pool, target, emb, n_splits=5):
    """Predição fora da parte para as imagens de `target`, treinando em `pool` (agrupado por paciente)."""
    pats = pool["patient_id"].unique()
    rng = np.random.default_rng(config.SEED)
    fold_of = dict(zip(pats, rng.permutation(len(pats)) % n_splits))
    out = []
    for k in range(n_splits):
        tr = pool[pool.patient_id.map(fold_of) != k]
        te = target[target.patient_id.map(fold_of) == k]
        if te.empty:
            continue
        clf = _fit(E.take(emb, tr.image_id), tr.referable.values, tr.patient_id.values)
        out.append(pd.DataFrame({"image_id": te.image_id.values, "p": clf.predict_proba(E.take(emb, te.image_id))[:, 1],
                                 "fold": k, "C": float(clf[-1].C_[0])}))
    return pd.concat(out, ignore_index=True)


def run(model, cohorts=None):
    """Roda os três cenários para um modelo de embedding e grava results/preds/baseline_{model}.csv."""
    cohorts = cohorts or {ds: C.load(ds) for ds in ("brset", "mbrset")}
    pool_b, pool_m = C.brset_training_pool(), C.mbrset_training_pool()
    emb_b, emb_m = E.load("brset", model), E.load("mbrset", model)
    tb, tm = cohorts["brset"], cohorts["mbrset"]
    missing = set(tb.patient_id) - set(pool_b.patient_id)
    assert not missing, f"{len(missing)} pacientes da coorte BRSET fora do pool de treino"
    parts = []
    r = _crossfit(pool_b, tb, emb_b)
    r["setting"], r["dataset"] = "brset_crossfit", "brset"
    parts.append(r)
    r = _crossfit(pool_m, tm, emb_m)
    r["setting"], r["dataset"] = "mbrset_crossfit", "mbrset"
    parts.append(r)
    clf = _fit(E.take(emb_b, pool_b.image_id), pool_b.referable.values, pool_b.patient_id.values)
    parts.append(pd.DataFrame({"image_id": tm.image_id.values, "p": clf.predict_proba(E.take(emb_m, tm.image_id))[:, 1],
                               "fold": -1, "C": float(clf[-1].C_[0]), "setting": "brset_to_mbrset", "dataset": "mbrset"}))
    out = pd.concat(parts, ignore_index=True)
    out["embedding"] = model
    config.PREDS.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.PREDS / f"baseline_{model}.csv", index=False)
    return out
