"""Teste rápido do pipeline com dados SINTÉTICOS (não usa nenhum dado do PhysioNet nem modelo de IA).

    python tests/smoke_test.py

Cria rótulos, imagens e embeddings falsos numa pasta temporária, monta a coorte, roda a linha de base
e as métricas. Serve para checar o código depois de mudanças, sem tocar nos dados reais.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

tmp = Path(tempfile.mkdtemp(prefix="vlmdr_smoke_"))
os.environ.update({"LABELS_DIR": str(tmp), "BRSET_IMAGES": str(tmp / "fundus_photos"),
                   "MBRSET_IMAGES": str(tmp / "mBRSET"), "EMB_NPY_DIR": str(tmp / "emb"),
                   "DINO_DIR": str(tmp / "x"), "RETFOUND_DIR": str(tmp / "y")})
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

config.DATA, config.RESULTS = tmp / "data", tmp / "results"
config.PREDS, config.TABLES = config.RESULTS / "preds", config.RESULTS / "tables"
config.BOOTSTRAP = 200

from vlmdr import baseline, cohort, embeddings, metrics, prompts  # noqa: E402

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- BRSET falso: 600 pacientes, 2 imagens cada
n = 600
pid = np.repeat(np.arange(1, n + 1), 2)
dm = np.repeat(rng.random(n) < 0.4, 2)
icdr = np.where(dm, rng.choice([0, 1, 2, 3, 4], size=2 * n, p=[.6, .1, .15, .05, .1]), 0)
b = pd.DataFrame({
    "image_id": [f"img{i:05d}" for i in range(1, 2 * n + 1)], "patient_id": pid, "camera": "Canon CR",
    "patient_age": np.repeat(rng.integers(30, 80, n), 2), "comorbidities": np.where(dm, "diabetes1, sah", ""),
    "diabetes": np.where(dm, "yes", "no"), "patient_sex": np.repeat(rng.integers(1, 3, n), 2),
    "exam_eye": np.tile([1, 2], n), "DR_ICDR": icdr, "macular_edema": (icdr >= 3).astype(int),
    "focus": rng.choice([1, 2], 2 * n, p=[.95, .05]), "Illuminaton": 1, "image_field": 1,
})
b.loc[b.sample(5, random_state=1).index, "comorbidities"] = np.nan
b.to_csv(tmp / "labels_brset.csv", index=False)

# --------------------------------------------------------------- mBRSET falso: 500 pacientes, 4 fotos cada
m_n = 500
mp = np.repeat(np.arange(1, m_n + 1), 4)
m = pd.DataFrame({
    "patient": mp, "age": np.repeat(rng.integers(30, 80, m_n), 4).astype(str), "sex": np.repeat(rng.integers(0, 2, m_n), 4),
    "file": [f"{p}.{k}.jpg" for p in range(1, m_n + 1) for k in (1, 2, 3, 4)],
    "laterality": np.tile(["right", "right", "left", "left"], m_n),
    "final_quality": rng.choice(["yes", "no"], 4 * m_n, p=[.95, .05]),
    "final_icdr": rng.choice([0, 1, 2, 3, 4, np.nan], 4 * m_n, p=[.6, .1, .15, .05, .05, .05]),
    "final_edema": rng.choice(["yes", "no"], 4 * m_n, p=[.08, .92]),
})
m.to_csv(tmp / "labels_mbrset.csv", index=False)

# ------------------------------------------------- embeddings falsos com sinal do desfecho + imagens vazias
(tmp / "emb").mkdir()
for ds, ids, ref in (("brset", b.image_id, ((b.DR_ICDR >= 2) | (b.macular_edema == 1))),
                     ("mbrset", m.file.str.replace(".jpg", "", regex=False),
                      (m.final_icdr >= 2) | (m.final_edema == "yes"))):
    X = rng.normal(size=(len(ids), 16)).astype(np.float32)
    X[:, 0] += 1.5 * ref.values
    np.save(tmp / "emb" / f"{ds}_dinov3_vits16.npy", X)
    np.save(tmp / "emb" / f"{ds}_dinov3_vits16_ids.npy", np.array(ids, dtype=object))
(tmp / "fundus_photos").mkdir()
(tmp / "mBRSET").mkdir()

# --------------------------------------------------------------------------------------------- testes
parts, flow, summ = cohort.build()
print(flow.to_string(index=False), "\n", summ.to_string(index=False), sep="")
br, mb = parts["brset"], parts["mbrset"]
assert br.image_id.is_unique and mb.image_id.is_unique, "imagem repetida na coorte"
assert config.ONE_PER_PATIENT or br.patient_id.duplicated().any(), "esperava os dois olhos de alguns pacientes"
assert br.groupby("patient_id").split.nunique().max() == 1, "paciente dividido entre piloto e avaliação"
assert set(br.patient_id) <= set("BR_" + b.patient_id[b.comorbidities.fillna("").str.contains("diab")].astype(str))
assert mb.image_id.str.match(r"^\d+\.[13]$").all(), "mBRSET com foto não macular"
assert br[br.split == "pilot"].patient_id.nunique() == config.N_PILOT
assert mb[mb.split == "pilot"].patient_id.nunique() == config.N_PILOT
assert ((br.referable == 1) == ((br.dr_icdr >= 2) | (br.macular_edema == 1))).all()
assert br[br.split == "pilot"].robust.sum() == 0 and br[br.split == "pilot"].repro.sum() == 0

print(embeddings.available().query("encontrado").to_string(index=False))
bl = baseline.run("dinov3_vits16")
for (s, ds), g in bl.groupby(["setting", "dataset"]):
    c = parts[ds].merge(g, on="image_id")
    print(f"{s:16s} AUROC {metrics.roc_auc_score(c.referable, c.p):.3f}  n={len(c)}")
    assert metrics.roc_auc_score(c.referable, c.p) > 0.7

# métricas
y = rng.integers(0, 2, 400)
s1, s2 = y + rng.normal(0, .8, 400), y + rng.normal(0, 1.5, 400)
a1, a2, diff, p = metrics.delong(y, s1, s2)
assert a1 > a2 and p < 0.05, (a1, a2, p)
r = metrics.operating_point(y, s1, 0.9)
assert 0.75 < r["sens"] <= 1
assert abs(metrics.wilson(80, 100)[1] - 0.711) < 0.01

# métricas agrupadas: com um paciente por linha, Obuchowski ~ DeLong; com dois olhos, IC mais largo que o ingênuo
solo = np.arange(400)
_, _, _, p_cl = metrics.delong(y, s1, s2, groups=solo)
assert abs(p_cl - p) < 0.01, (p_cl, p)
pid2 = np.repeat(np.arange(200), 2)
y2 = np.repeat(rng.integers(0, 2, 200), 2)
s2a = y2 + np.repeat(rng.normal(0, .8, 200), 2) + rng.normal(0, .2, 400)
lo_n, hi_n = metrics.bootstrap_ci(metrics.roc_auc_score, y2, s2a)
lo_c, hi_c = metrics.bootstrap_ci(metrics.roc_auc_score, y2, s2a, groups=pid2)
assert hi_c - lo_c > hi_n - lo_n, ((lo_n, hi_n), (lo_c, hi_c))
r = metrics.operating_point(y2, s2a, 0.9, groups=pid2)
a_idx, b_idx = metrics.split_half(400, groups=pid2)
assert not set(pid2[a_idx]) & set(pid2[b_idx])
assert 0 <= metrics.mcnemar(y2 == 1, s2a > .5, groups=pid2) <= 1

# grau: kappa ponderado e diferença pareada
g_true = rng.integers(0, 5, 400)
g_good = np.clip(g_true + rng.integers(-1, 2, 400), 0, 4)
g_bad = rng.integers(0, 5, 400)
k, lo, hi = metrics.kappa(g_true, g_good, groups=pid2)
assert lo < k < hi and k > 0.6, (k, lo, hi)
est, lo, hi, pk = metrics.paired_diff(metrics._kappa("quadratic"), g_true, g_good, g_bad, groups=pid2)
assert est > 0 and lo > 0 and pk < 0.05, (est, lo, hi, pk)

# parser
pr = prompts.load_prompts()
en, pt = pr["main_en"], pr["main_pt"]
assert prompts.parse("2; Macular edema: No", en) == {"grade": 2, "edema": "no", "answer": "yes"}
assert prompts.parse("0; Macular edema: No", en)["answer"] == "no"
assert prompts.parse("1; Macular edema: Yes", en)["answer"] == "yes"          # leve + edema = referenciável
assert prompts.parse("**1**; Macular edema: **No**", en)["answer"] == "no"
assert prompts.parse("4", en) == {"grade": 4, "edema": None, "answer": "yes"}  # grau >= 2 dispensa o edema
assert prompts.parse("1", en)["answer"] == "invalid"                            # grau < 2 sem edema legível
assert prompts.parse("I cannot determine", en)["answer"] == "invalid"
assert prompts.parse("2.5; Macular edema: No", en)["grade"] is None
assert prompts.parse("0; Edema macular: Não", pt)["answer"] == "no"
assert prompts.parse("1; Edema macular: Sim", pt)["answer"] == "yes"
print("\nOK: coorte, linha de base, métricas e parser passaram no teste sintético.")
