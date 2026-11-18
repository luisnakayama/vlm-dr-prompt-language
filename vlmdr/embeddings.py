"""Leitura dos embeddings congelados (cache .npy do bilateral_leakage ou CSVs do PhysioNet)."""
import numpy as np
import pandas as pd

import config
from .cohort import image_key

_ID_COLS = ["image_id", "file", "ImageName", "image", "filename"]


def _from_npy(ds, model):
    X = config.EMB_NPY_DIR / f"{ds}_{model}.npy"
    ids = config.EMB_NPY_DIR / f"{ds}_{model}_ids.npy"
    if X.exists() and ids.exists():
        return np.load(ids, allow_pickle=True).astype(str), np.load(X).astype(np.float32), X
    return None


def _from_csv(ds, model):
    for d in config.EMB_CSV_DIRS:
        p = d / f"Embeddings_{ds}_{model}.csv"
        if p.exists():
            df = pd.read_csv(p)
            idc = next((c for c in _ID_COLS if c in df.columns), df.columns[0])
            feats = [c for c in df.columns if c != idc and pd.api.types.is_numeric_dtype(df[c])]
            return df[idc].astype(str).values, df[feats].to_numpy(np.float32), p
    return None


def load(ds, model):
    """Retorna DataFrame indexado pela chave da imagem (linhas = imagens, colunas = dimensões)."""
    got = _from_npy(ds, model) or _from_csv(ds, model)
    if got is None:
        raise FileNotFoundError(f"Embeddings de {ds}/{model} não encontrados em {config.EMB_NPY_DIR} "
                                f"nem em {[str(d) for d in config.EMB_CSV_DIRS]}")
    ids, X, src = got
    df = pd.DataFrame(X, index=[image_key(i) for i in ids])
    df = df[~df.index.duplicated()]
    df.attrs["source"] = str(src)
    return df


def take(emb, image_ids):
    keys = [image_key(i) for i in image_ids]
    missing = [k for k in keys if k not in emb.index]
    if missing:
        raise KeyError(f"{len(missing)} imagens sem embedding (ex.: {missing[:5]})")
    return emb.loc[keys].to_numpy()


def available():
    rows = []
    for ds in ("brset", "mbrset"):
        for m in config.EMB_MODELS:
            got = _from_npy(ds, m) or _from_csv(ds, m)
            rows.append({"dataset": ds, "modelo": m, "encontrado": got is not None,
                         "fonte": str(got[2]) if got else "", "imagens": len(got[0]) if got else 0,
                         "dimensoes": got[1].shape[1] if got else 0})
    return pd.DataFrame(rows)
