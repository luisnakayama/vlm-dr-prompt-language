"""Execução da inferência por modelo (notebooks 03): checagem no piloto, principal, robustez e reprodutibilidade.

Ordem das etapas: principal nas duas bases primeiro, depois robustez, depois reprodutibilidade; se o tempo acabar,
o desfecho primário já está completo. Grava a cada imagem e retoma de onde parou. O log vai para
results/execucao_<tag>.csv.

Etapas do artigo 2 (notebooks 05), com run = "a2" para não misturar com a subamostra de robustez do artigo 1:
  pt_completo   : main_pt na amostra de avaliação inteira;
  dois_por_dois : pt_instr_en_fmt e en_instr_pt_fmt na subamostra de robustez (as outras duas células do 2x2 são
                  main_en/r1 e main_pt/a2, já cobertas).
"""
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config
from . import infer


class Runner:
    def __init__(self, cohorts, tag, etapas=("piloto", "principal", "robustez", "repro"), limite=None, n_piloto=20,
                 limpar_cache=False):
        self.cohorts, self.etapas, self.limite, self.n_piloto = cohorts, etapas, limite, n_piloto
        self.limpar_cache = limpar_cache
        config.RESULTS.mkdir(parents=True, exist_ok=True)
        self.path = config.RESULTS / f"execucao_{tag}.csv"
        self.log = []

    def registra(self, **kw):
        self.log.append({"hora_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **kw})
        pd.DataFrame(self.log).to_csv(self.path, index=False)
        print(self.log[-1], flush=True)

    def checa_piloto(self, vlm):
        """Poucas imagens do piloto (run = piloto_<dispositivo>, fora das análises); lista de problemas encontrados."""
        r = pd.concat([infer.run(vlm, c[c.split == "pilot"].head(self.n_piloto), "main_en", f"piloto_{vlm.device}",
                                 progress=False) for c in self.cohorts.values()])
        problemas = []
        if r.p_yes.isna().any():
            problemas.append(f"{int(r.p_yes.isna().sum())} P(referenciável) nan")
        if (r.answer == "invalid").mean() > 0.10:
            problemas.append(f"{(r.answer == 'invalid').mean():.0%} inválidas")
        if r.grade_mass.mean() < 0.9:
            problemas.append(f"massa média nos graus {r.grade_mass.mean():.2f}")
        return problemas, r.seconds.median()

    def tarefas(self):
        ev = {ds: c[c.split == "eval"] for ds, c in self.cohorts.items()}
        out = []
        if "principal" in self.etapas:
            out += [("principal", ds, e, "main_en", "r1") for ds, e in ev.items()]
        if "robustez" in self.etapas:
            out += [("robustez", ds, e[e.robust == 1], pk, "r1") for pk in ("nodef_en", "main_pt") for ds, e in ev.items()]
        if "repro" in self.etapas:
            out += [("repro", ds, e[e.repro == 1], "main_en", "r2") for ds, e in ev.items()]
        if "pt_completo" in self.etapas:
            out += [("pt_completo", ds, e, "main_pt", "a2") for ds, e in ev.items()]
        if "dois_por_dois" in self.etapas:
            out += [("dois_por_dois", ds, e[e.robust == 1], pk, "a2")
                    for pk in ("pt_instr_en_fmt", "en_instr_pt_fmt") for ds, e in ev.items()]
        return out

    def roda(self, m, etapas=None):
        """Roda o modelo `m`; `etapas` substitui as etapas do Runner só para este modelo."""
        if etapas is not None:
            guardadas, self.etapas = self.etapas, etapas
            try:
                return self.roda(m)
            finally:
                self.etapas = guardadas
        t0 = time.time()
        vlm = infer.VLM(m)
        self.registra(modelo=m, etapa="carregado", revisao=vlm.revision, dtype=vlm.dtype_name, dispositivo=vlm.device)
        try:
            if "piloto" in self.etapas:
                problemas, seg = self.checa_piloto(vlm)
                self.registra(modelo=m, etapa="piloto", seg_por_imagem=round(seg, 2), problemas="; ".join(problemas) or "ok")
                if problemas:
                    return
            for etapa, ds, df, pk, run in self.tarefas():
                t = time.time()
                r = infer.run(vlm, df, pk, run, limit=self.limite, progress=False)
                self.registra(modelo=m, etapa=etapa, base=ds, prompt=pk, run=run, n=len(r),
                              invalidas=int((r.answer == "invalid").sum()), minutos=round((time.time() - t) / 60, 1))
        finally:
            self.registra(modelo=m, etapa="fim", memoria_GB=round(infer.peak_memory_gb(vlm.device), 1),
                          horas=round((time.time() - t0) / 3600, 2))
            vlm.close()
            if self.limpar_cache:
                from huggingface_hub import constants
                shutil.rmtree(Path(constants.HF_HUB_CACHE) / ("models--" + vlm.cfg["hf_id"].replace("/", "--")),
                              ignore_errors=True)
