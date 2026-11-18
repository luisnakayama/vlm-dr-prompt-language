"""Inferência local (Hugging Face transformers; MPS no Mac, CUDA ou CPU como alternativa).

O modelo responde o grau ICDR e o edema macular; a resposta começa já escrita com o prefill do prompt ("ICDR: "),
então o 1º token gerado é o grau. Para cada imagem grava: resposta gerada, grau e edema lidos, referenciável
(yes/no/invalid), probabilidades de cada grau (p_g0..p_g4, normalizadas nos dígitos 0-4), P(edema) no passo em
que a palavra sim/não é gerada, escore contínuo p_yes = P(referenciável) = 1 - P(grau 0 ou 1) x [1 - P(edema)],
tokens de entrada e saída e tempo.
O arquivo de saída é gravado a cada imagem: se o processo parar, rodar de novo continua de onde parou.
"""
import gc
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config
from .prompts import GRADES, load_models, load_prompts, parse

COLUMNS = ["image_id", "patient_id", "dataset", "model", "hf_id", "revision", "prompt", "run", "raw_text",
           "grade_pred", "edema_pred", "answer", "p_yes", "p_g0", "p_g1", "p_g2", "p_g3", "p_g4", "p_edema",
           "grade_mass", "edema_mass", "n_tokens_in", "n_tokens_out", "seconds", "device", "dtype", "timestamp"]


def device_name():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class VLM:
    def __init__(self, key, device=None, dtype=None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageTextToText, AutoProcessor

        self.key = key
        self.cfg = load_models()[key]
        self.device = device or device_name()
        self.dtype_name = dtype or self.cfg.get("dtype", "bfloat16")
        self.dtype = getattr(torch, self.dtype_name)
        rev = {"revision": self.cfg["revision"]} if self.cfg.get("revision") else {}
        pkw = {k: self.cfg[k] for k in ("min_pixels", "max_pixels") if k in self.cfg}
        self.processor = AutoProcessor.from_pretrained(self.cfg["hf_id"], **rev, **pkw)
        if "use_fast" in self.cfg:   # só o processador de imagem (no AutoProcessor, use_fast trocaria também o tokenizador)
            self.processor.image_processor = AutoImageProcessor.from_pretrained(
                self.cfg["hf_id"], use_fast=self.cfg["use_fast"], **rev, **pkw)
        mkw = {"dtype": self.dtype, **rev}
        if self.cfg.get("attn_implementation"):
            mkw["attn_implementation"] = self.cfg["attn_implementation"]
        self.model = AutoModelForImageTextToText.from_pretrained(self.cfg["hf_id"], **mkw).to(self.device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or ""
        self.tok = self.processor.tokenizer
        self._ids = {}

    # -------------------------------------------------------------------------------- tokens de resposta
    def _firsts(self, words):
        ids = set()
        for w in words:
            for v in (w, " " + w):
                t = self.tok.encode(v, add_special_tokens=False)
                if t:
                    ids.add(t[0])
        return ids

    def answer_ids(self, prompt):
        """Primeiro token de cada palavra de edema sim/não (com e sem espaço antes); ambíguos são descartados."""
        key = tuple(prompt["yes_words"] + ["|"] + prompt["no_words"])
        if key not in self._ids:
            y, n = self._firsts(prompt["yes_words"]), self._firsts(prompt["no_words"])
            self._ids[key] = (sorted(y - n), sorted(n - y))
        return self._ids[key]

    def grade_ids(self):
        """Tokens de cada dígito 0-4 (com e sem espaço antes); tokens comuns a mais de um dígito (ex.: o espaço) saem."""
        if "grades" not in self._ids:
            ids = [self._firsts([g]) for g in GRADES]
            ids = [s - set().union(*(o for o in ids if o is not s)) for s in ids]
            assert all(ids), "dígito sem token próprio"
            self._ids["grades"] = [sorted(s) for s in ids]
        return self._ids["grades"]

    # -------------------------------------------------------------------------------- uma imagem
    def predict(self, image_path, prompt):
        import torch

        img = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt["text"]}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        text += prompt.get("prefill", "")
        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        inputs = {k: (v.to(self.device, self.dtype) if torch.is_floating_point(v) else v.to(self.device))
                  for k, v in inputs.items()}
        n_in = int(inputs["input_ids"].shape[1])
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False,
                                      output_scores=True, return_dict_in_generate=True)
        if self.device == "mps":
            torch.mps.synchronize()
        secs = time.perf_counter() - t0
        gen = out.sequences[0, n_in:].tolist()
        # grau: distribuição no 1º token gerado, normalizada nos dígitos 0-4
        logp = torch.log_softmax(out.scores[0][0].float(), dim=-1)
        lg = torch.stack([torch.logsumexp(logp[ids], 0) for ids in self.grade_ids()])
        p_g = torch.softmax(lg, 0).tolist()
        grade_mass = float(torch.exp(torch.logsumexp(lg, 0)))
        # edema: passo em que o modelo gera a palavra sim/não (condicionado ao grau que ele escreveu)
        yes_ids, no_ids = self.answer_ids(prompt)
        yn = set(yes_ids) | set(no_ids)
        step = next((k for k in range(1, len(gen)) if gen[k] in yn), None)
        p_edema = edema_mass = np.nan
        if step is not None:
            le = torch.log_softmax(out.scores[step][0].float(), dim=-1)
            ly, ln = torch.logsumexp(le[yes_ids], 0), torch.logsumexp(le[no_ids], 0)
            p_edema = float(torch.sigmoid(ly - ln))
            edema_mass = float(torch.exp(torch.logsumexp(torch.stack([ly, ln]), 0)))
        p_low = p_g[0] + p_g[1]
        p_ref = 1 - p_low * (1 - p_edema) if step is not None else 1 - p_low
        stop = {self.tok.eos_token_id, self.tok.pad_token_id}
        n_out = int(sum(1 for t in gen if t not in stop))
        raw = self.tok.decode(gen, skip_special_tokens=True).strip()
        r = parse(raw, prompt)
        return {"raw_text": raw, "grade_pred": r["grade"], "edema_pred": r["edema"], "answer": r["answer"],
                "p_yes": p_ref, **{f"p_g{k}": v for k, v in enumerate(p_g)}, "p_edema": p_edema,
                "grade_mass": grade_mass, "edema_mass": edema_mass,
                "n_tokens_in": n_in, "n_tokens_out": n_out, "seconds": secs}

    def close(self):
        import torch
        del self.model
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()


def out_path(model_key, ds, prompt_key, run="r1"):
    return config.PREDS / f"vlm_{model_key}_{ds}_{prompt_key}_{run}.csv"


def _predict_retry(vlm, image_path, prompt, tries=3):
    """predict com novas tentativas: no MPS, falhas momentâneas aparecem depois de horas rodando (ex.: 'Image features
    and image tokens do not match' numa imagem que funciona ao repetir). Libera a memória entre as tentativas."""
    import torch
    for k in range(tries):
        try:
            return vlm.predict(image_path, prompt)
        except (RuntimeError, ValueError) as e:
            if k == tries - 1:
                raise
            print(f"falha em {Path(image_path).name} ({type(e).__name__}: {str(e)[:80]}); tentando de novo", flush=True)
            gc.collect()
            if vlm.device == "mps":
                torch.mps.empty_cache()
            elif vlm.device == "cuda":
                torch.cuda.empty_cache()


def run(vlm, df, prompt_key, run_tag="r1", limit=None, progress=True):
    """Roda `vlm` nas linhas de `df` (uma base) com o prompt indicado; retoma se o arquivo já existir."""
    prompt = load_prompts()[prompt_key]
    ds = df["dataset"].iloc[0]
    path = out_path(vlm.key, ds, prompt_key, run_tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(path, dtype={"image_id": str}).image_id) if path.exists() else set()
    todo = df[~df.image_id.astype(str).isin(done)]
    if limit is not None:
        todo = todo.head(max(0, limit - len(done)))
    it = todo.itertuples(index=False)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, total=len(todo), desc=f"{vlm.key} · {ds} · {prompt_key} · {run_tag}")
    for r in it:
        if not Path(r.image_path).exists():
            raise FileNotFoundError(r.image_path)
        res = _predict_retry(vlm, r.image_path, prompt)
        row = {"image_id": r.image_id, "patient_id": r.patient_id, "dataset": ds, "model": vlm.key,
               "hf_id": vlm.cfg["hf_id"], "revision": vlm.revision, "prompt": prompt_key, "run": run_tag,
               "device": vlm.device, "dtype": vlm.dtype_name,
               "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **res}
        pd.DataFrame([row])[COLUMNS].to_csv(path, mode="a", header=not path.exists(), index=False)
    return pd.read_csv(path, dtype={"image_id": str})


def peak_memory_gb(device):
    import torch
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    return np.nan


def load_preds(pattern="vlm_*.csv"):
    files = sorted(config.PREDS.glob(pattern))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_csv(f, dtype={"image_id": str}) for f in files], ignore_index=True)
