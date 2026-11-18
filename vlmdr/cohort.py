"""Coorte do estudo.

Critérios:
  BRSET : só pacientes com diabetes (texto de `comorbidities`; ver config.DIABETES_RULE);
          imagens com grau ICDR; foco e iluminação adequados (2 = insatisfatório).
  mBRSET: só fotos centradas na mácula (arquivos terminados em .1 = olho direito e .3 = olho esquerdo);
          imagens com grau ICDR; final_quality == "yes".
  Ambas : todas as imagens elegíveis (os dois olhos; config.ONE_PER_PATIENT = True sorteia um olho);
          50 pacientes por base vão para o piloto. ICs e testes são agrupados por paciente (vlmdr.metrics).
Desfecho primário: grau ICDR (0-4). Secundário: RD referenciável = ICDR >= 2 e/ou edema macular.

Saídas em data/ (não versionar: dados de acesso controlado do PhysioNet).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config

GRADE_NAMES = {0: "sem RD", 1: "leve", 2: "moderada", 3: "grave", 4: "proliferativa"}


def _step(flow, ds, label, df, pid):
    flow.append({"dataset": ds, "etapa": label, "imagens": len(df), "pacientes": int(df[pid].nunique())})


def _yn(s):
    return s.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})


def image_key(x):
    """Chave comum a rótulos e embeddings: sem extensão, sem espaços ('1.1.jpg' -> '1.1')."""
    return re.sub(r"\.(jpe?g|png)$", "", str(x).strip(), flags=re.I)


def find_image(folder, name):
    for cand in (folder / name, folder / "images" / name, folder / "fundus_photos" / name):
        if cand.exists():
            return cand
    return folder / name


# ------------------------------------------------------------------------------------------------ BRSET
def read_brset():
    return pd.read_csv(config.BRSET_LABELS, encoding="utf-8-sig", low_memory=False)


def brset_diabetes(b):
    """Indicador de diabetes por paciente (qualquer imagem do paciente com a informação)."""
    by_text = b["comorbidities"].fillna("").astype(str).str.lower().str.contains("diab")
    if config.DIABETES_RULE == "comorbidities_or_column" and "diabetes" in b:
        by_col = b["diabetes"].astype(str).str.strip().str.lower().isin(["yes", "1", "1.0", "sim"])
        by_text = by_text | by_col
    return by_text.groupby(b["patient_id"]).transform("max").astype(bool)


def brset_quality_ok(b):
    cols = [c for c in config.BRSET_QUALITY_COLS if c in b]
    missing = set(config.BRSET_QUALITY_COLS) - set(cols)
    if missing:
        print("Aviso: colunas de qualidade ausentes no CSV:", sorted(missing))
    ok = pd.Series(True, index=b.index)
    for c in cols:
        ok &= pd.to_numeric(b[c], errors="coerce") != 2
    return ok


def _brset_frame(b):
    return pd.DataFrame({
        "image_id": b["image_id"].map(image_key).values,
        "patient_id": ("BR_" + b["patient_id"].astype(str)).values,
        "dataset": "brset",
        "eye": b["exam_eye"].map({1: "right", 2: "left"}).values,     # BRSET: 1 = OD, 2 = OE
        "camera": b["camera"].values,
        "age": pd.to_numeric(b["patient_age"], errors="coerce").values,
        "sex_female": b["patient_sex"].map({2: 1, 1: 0}).values,      # BRSET: 1 = masc., 2 = fem.
        "dr_icdr": b["DR_ICDR"].astype(int).values,
        "macular_edema": pd.to_numeric(b["macular_edema"], errors="coerce").fillna(0).astype(int).values,
        "image_path": [str(find_image(config.BRSET_IMAGES, f"{image_key(i)}.jpg")) for i in b["image_id"]],
    })


def brset(flow):
    b = read_brset()
    _step(flow, "brset", "Registros totais", b, "patient_id")
    b = b[brset_diabetes(b)]
    _step(flow, "brset", f"Pacientes com diabetes ({config.DIABETES_RULE})", b, "patient_id")
    b = b[b["DR_ICDR"].notna()]
    _step(flow, "brset", "Com grau ICDR", b, "patient_id")
    b = b[brset_quality_ok(b)]
    _step(flow, "brset", "Foco e iluminação adequados", b, "patient_id")
    return _brset_frame(b)


def brset_training_pool():
    """Todas as imagens graduáveis do BRSET (com e sem diabetes): treino da linha de base."""
    b = read_brset()
    b = b[b["DR_ICDR"].notna() & brset_quality_ok(b)]
    out = _brset_frame(b)
    out["diabetes"] = brset_diabetes(b).astype(int).values
    return finish(out, select=False)


# ----------------------------------------------------------------------------------------------- mBRSET
def read_mbrset():
    return pd.read_csv(config.MBRSET_LABELS, low_memory=False)


def _mbrset_filter(m, flow=None):
    if flow is not None:
        _step(flow, "mbrset", "Registros totais", m, "patient")
    suf = m["file"].astype(str).str.extract(r"^\d+\.(\d+)")[0]
    m = m[suf.isin(["1", "3"])]
    if flow is not None:
        _step(flow, "mbrset", "Centrada na mácula (.1 e .3)", m, "patient")
    m = m[m["final_icdr"].notna()]
    if flow is not None:
        _step(flow, "mbrset", "Com grau ICDR", m, "patient")
    m = m[_yn(m["final_quality"]) == 1]
    if flow is not None:
        _step(flow, "mbrset", "Qualidade adequada (final_quality)", m, "patient")
    age = pd.to_numeric(m["age"].astype(str).str.replace(">=", "", regex=False).str.strip(), errors="coerce")
    return pd.DataFrame({
        "image_id": m["file"].map(image_key).values,
        "patient_id": ("MB_" + m["patient"].astype(str)).values,
        "dataset": "mbrset",
        "eye": m["laterality"].values,
        "camera": "Phelcom Eyer",
        "age": age.values,
        "sex_female": m["sex"].map({0: 1, 1: 0}).values,              # mBRSET: 0 = fem., 1 = masc.
        "dr_icdr": m["final_icdr"].astype(int).values,
        "macular_edema": _yn(m["final_edema"]).fillna(0).astype(int).values,
        "image_path": [str(find_image(config.MBRSET_IMAGES, str(f).strip())) for f in m["file"]],
    })


def mbrset(flow):
    return _mbrset_filter(read_mbrset(), flow)


def mbrset_training_pool():
    """Todas as fotos maculares graduáveis do mBRSET (os dois olhos): validação cruzada da linha de base."""
    return finish(_mbrset_filter(read_mbrset()), select=False)


# ---------------------------------------------------------------------------------------------- comum
def finish(df, select=True, flow=None, seed=config.SEED):
    df = df.copy()
    df["referable"] = ((df["dr_icdr"] >= 2) | (df["macular_edema"] == 1)).astype(int)
    df["grade"] = df["dr_icdr"].map(GRADE_NAMES)
    if not select:
        return df.reset_index(drop=True)
    ds = df["dataset"].iloc[0]
    if config.ONE_PER_PATIENT:
        # uma imagem por paciente: embaralha com semente fixa e fica com a primeira de cada paciente
        df = df.sample(frac=1, random_state=seed).drop_duplicates("patient_id").reset_index(drop=True)
        _step(flow, ds, "Uma imagem por paciente (olho sorteado)", df, "patient_id")
    else:
        df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    pilot = set(rng.choice(df["patient_id"].unique(), size=min(config.N_PILOT, len(df)), replace=False))
    df["split"] = np.where(df["patient_id"].isin(pilot), "pilot", "eval")
    ev = df[df["split"] == "eval"]
    if config.SAMPLE_MODE == "stratified":
        keep = pd.concat([
            ev[ev.referable == 1].sample(n=min(config.N_REF, int(ev.referable.sum())), random_state=seed),
            ev[ev.referable == 0].sample(n=min(config.N_NONREF, int((ev.referable == 0).sum())), random_state=seed),
        ]).index
        df.loc[ev.index.difference(keep), "split"] = "not_sampled"
        ev = df[df["split"] == "eval"]
    _step(flow, ds, f"Piloto (excluído das análises): {len(pilot)} pacientes", df[df.split == "pilot"], "patient_id")
    _step(flow, ds, "Amostra de avaliação", ev, "patient_id")
    # subamostras estratificadas pelo desfecho
    for col, n in (("robust", config.N_ROBUST), ("repro", config.N_REPRO)):
        frac = min(1.0, n / max(len(ev), 1))
        idx = ev.groupby("referable", group_keys=False).sample(frac=frac, random_state=seed + len(col)).index
        df[col] = df.index.isin(idx).astype(int)
    df = df[df["split"] != "not_sampled"].reset_index(drop=True)
    lead = ["image_id", "patient_id", "dataset", "split", "robust", "repro", "eye", "referable", "dr_icdr",
            "grade", "macular_edema"]
    return df[lead + [c for c in df.columns if c not in lead]]


def build():
    """Monta e grava a coorte. Retorna (dict de coortes, fluxo de exclusão, resumo)."""
    config.DATA.mkdir(exist_ok=True)
    flow = []
    parts = {"brset": finish(brset(flow), flow=flow), "mbrset": finish(mbrset(flow), flow=flow)}
    flow = pd.DataFrame(flow)
    flow.to_csv(config.DATA / "fluxo_exclusao.csv", index=False)
    rows = []
    for ds, d in parts.items():
        d.to_csv(config.DATA / f"cohort_{ds}.csv", index=False)
        ev = d[d.split == "eval"]
        rows.append({"dataset": ds, "imagens_avaliacao": len(ev), "pacientes_avaliacao": ev.patient_id.nunique(),
                     "referenciaveis": int(ev.referable.sum()), "prevalencia_%": round(100 * ev.referable.mean(), 1),
                     "piloto_pacientes": int(d[d.split == "pilot"].patient_id.nunique()),
                     "subamostra_robustez": int(d.robust.sum()), "subamostra_repro": int(d.repro.sum())})
    summ = pd.DataFrame(rows)
    summ.to_csv(config.DATA / "resumo_coorte.csv", index=False)
    return parts, flow, summ


def load(ds):
    """Coorte gravada por build(). image_path é refeito a partir de config.*_IMAGES (a coorte pode vir de outra máquina)."""
    df = pd.read_csv(config.DATA / f"cohort_{ds}.csv", dtype={"image_id": str})
    folder = config.BRSET_IMAGES if ds == "brset" else config.MBRSET_IMAGES
    df["image_path"] = [str(find_image(folder, Path(p).name)) for p in df["image_path"]]
    return df
