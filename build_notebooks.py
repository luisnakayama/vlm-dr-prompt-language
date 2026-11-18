"""Gera os notebooks em notebooks/ (edite aqui para mudanças estruturais).

    python build_notebooks.py                 # só gera
    python build_notebooks.py --run 00 01     # gera e executa os indicados (grava as saídas)
    python build_notebooks.py --only 05 05b   # gera só os indicados (não mexe nos já executados)
"""
import sys
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parent / "notebooks"
NB.mkdir(exist_ok=True)
SETUP = r'''%load_ext autoreload
%autoreload 2
import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import config
pd.set_option("display.max_columns", 50, "display.width", 250, "display.max_rows", 200)'''
books = {}

# ------------------------------------------------------------------------------------------- 00
books["00_coorte"] = [
("md", """# 00 · Coorte

Monta a coorte do estudo e confere os caminhos das imagens.

**Critérios**
- **BRSET**: só pacientes com diabetes (texto de `comorbidities`; a coluna `diabetes` codifica dado ausente como "No"); imagens com grau ICDR; foco e iluminação adequados.
- **mBRSET**: só fotos centradas na mácula (arquivos `.1` = olho direito e `.3` = olho esquerdo); imagens com grau ICDR; `final_quality == "yes"`.
- **Ambas**: todas as imagens elegíveis (os dois olhos; ICs e testes agrupados por paciente); 50 pacientes por base vão para o piloto (só ajuste de prompt e parser).
- **Desfecho primário**: grau ICDR (0–4) previsto pelo modelo vs. o de referência (kappa ponderado quadrático).
- **Desfecho secundário**: RD referenciável = ICDR ≥ 2 e/ou edema macular.

Caminhos em `config.py`. Nada de linhas individuais é exibido: os dados são de acesso controlado."""),
("code", SETUP + "\nfrom vlmdr import cohort"),
("md", "## Arquivos de entrada"),
("code", r'''for name in ["BRSET_LABELS", "MBRSET_LABELS", "BRSET_IMAGES", "MBRSET_IMAGES", "EMB_NPY_DIR"]:
    p = getattr(config, name)
    print(f"{name:14s} {'ok ' if p.exists() else 'NÃO ENCONTRADO'}  {p}")'''),
("md", "## Regra de diabetes no BRSET\n\nCompara o texto de `comorbidities` com a coluna `diabetes`, por paciente. Confira antes de seguir."),
("code", r'''b = cohort.read_brset()
by_text = b.comorbidities.fillna("").str.lower().str.contains("diab").groupby(b.patient_id).max()
col = b.diabetes.astype(str).str.strip().str.lower().groupby(b.patient_id).first()
pd.crosstab(col.rename("coluna diabetes"), by_text.rename("comorbidities contém 'diab'"), margins=True)'''),
("md", "## Fluxo de exclusão e resumo"),
("code", "parts, flow, summ = cohort.build()\ndisplay(flow, summ)"),
("md", "## Distribuição do desfecho e dos graus (amostra de avaliação)"),
("code", r'''for ds, d in parts.items():
    ev = d[d.split == "eval"]
    print(f"\n{ds}: {len(ev)} imagens de {ev.patient_id.nunique()} pacientes, {ev.referable.sum()} referenciáveis "
          f"({100 * ev.referable.mean():.1f}%)")
    display(pd.crosstab(ev.grade, ev.macular_edema.rename("edema"), margins=True))'''),
("md", "## Imagens no disco"),
("code", r'''for ds, d in parts.items():
    ok = d.image_path.map(lambda p: Path(p).exists())
    print(f"{ds}: {ok.sum()}/{len(d)} imagens encontradas")
    if not ok.all():
        print("  exemplo de caminho ausente:", d.loc[~ok, "image_path"].iloc[0])'''),
]

# ------------------------------------------------------------------------------------------- 01
books["01_baseline_embeddings"] = [
("md", """# 01 · Linha de base supervisionada (embeddings + regressão logística)

Usa os embeddings congelados já extraídos (cache `.npy` do projeto `bilateral_leakage` ou os CSVs do PhysioNet).

| Cenário | Treino | Teste |
|---|---|---|
| `brset_crossfit` | BRSET graduável (com e sem diabetes), 5 partes por paciente | coorte BRSET (diabéticos) |
| `mbrset_crossfit` | mBRSET macular graduável (os dois olhos), 5 partes por paciente | coorte mBRSET |
| `brset_to_mbrset` | todo o BRSET | coorte mBRSET (validação externa entre câmeras) |

Referência anterior (`embeddings_experiments`, mBRSET macular, ICDR ≥ 2, sem edema): AUROC 0,880 (DINOv3 ConvNeXt-Tiny) e 0,855 (DINOv3 ViT-S/16)."""),
("code", SETUP + "\nfrom vlmdr import baseline, cohort, embeddings, metrics"),
("code", "embeddings.available()"),
("code", r'''cohorts = {ds: cohort.load(ds) for ds in ("brset", "mbrset")}
rows = []
for m in config.EMB_MODELS:
    try:
        pr = baseline.run(m, cohorts)
    except FileNotFoundError as e:
        print("pulando", m, "->", e)
        continue
    for (setting, ds), d in pr.groupby(["setting", "dataset"]):
        c = cohorts[ds].merge(d, on="image_id")
        c = c[c.split == "eval"]
        auc = metrics.auroc(c.referable, c.p, groups=c.patient_id)
        ap = metrics.auprc(c.referable, c.p, groups=c.patient_id)
        rows.append({"embedding": m, "cenario": setting, "n": len(c), "pacientes": c.patient_id.nunique(),
                     "positivos": int(c.referable.sum()),
                     "AUROC": metrics.fmt(*auc, pct=False), "AUPRC": metrics.fmt(*ap, pct=False), "_auc": auc[0]})
res = pd.DataFrame(rows).sort_values(["cenario", "_auc"], ascending=[True, False])
config.TABLES.mkdir(parents=True, exist_ok=True)
res.drop(columns="_auc").to_csv(config.TABLES / "baseline_auroc.csv", index=False)
res.drop(columns="_auc")'''),
("md", "Escolha **um** embedding para a linha de base do artigo (registre a escolha antes de olhar os resultados dos VLMs)."),
]

# ------------------------------------------------------------------------------------------- 02
books["02_teste_local"] = [
("md", """# 02 · Teste local (piloto, MPS)

Roda poucas imagens do **piloto** (excluídas das análises) em cada modelo local para conferir: o modelo carrega no MPS,
a resposta (grau ICDR + edema macular) é lida corretamente, o escore P(referenciável) não sai `nan`
e quanto tempo leva cada imagem.
Os modelos são carregados um de cada vez e liberados antes do próximo.

Antes: `huggingface-cli login` e aceite as licenças do MedGemma e do Gemma 3 no Hugging Face."""),
("code", SETUP + "\nfrom vlmdr import cohort, infer, prompts"),
("code", r'''MODELOS = [k for k, v in prompts.load_models().items() if v["local"]]   # ou uma lista, ex.: ["gemma3_4b"]
N = 20                      # imagens do piloto por base
PROMPT = "main_en"
print("modelos:", MODELOS, "| dispositivo:", infer.device_name())'''),
("code", r'''p = prompts.load_prompts()[PROMPT]
cohorts = {ds: cohort.load(ds) for ds in ("brset", "mbrset")}
out, info = [], []
for m in MODELOS:
    try:
        vlm = infer.VLM(m)
    except Exception as e:
        print(f"{m}: falhou ao carregar -> {type(e).__name__}: {e}")
        continue
    y_ids, n_ids = vlm.answer_ids(p)
    print(f"\n{m}: {vlm.cfg['hf_id']} | revisão {vlm.revision} | dtype {vlm.dtype_name}")
    print("  tokens dos graus:", [[vlm.tok.decode([i]) for i in ids] for ids in vlm.grade_ids()])
    print("  tokens edema sim:", [vlm.tok.decode([i]) for i in y_ids], "| não:", [vlm.tok.decode([i]) for i in n_ids])
    for ds, c in cohorts.items():
        pil = c[c.split == "pilot"].head(N)
        r = infer.run(vlm, pil, PROMPT, run_tag="piloto")
        out.append(r.merge(c[["image_id", "referable", "dr_icdr", "macular_edema"]], on="image_id"))
    info.append({"modelo": m, "memoria_GB": round(infer.peak_memory_gb(vlm.device), 1)})
    vlm.close()
out = pd.concat(out, ignore_index=True)
info = pd.DataFrame(info)'''),
("md", "## Respostas por imagem"),
("code", r'''out[["model", "dataset", "referable", "dr_icdr", "macular_edema", "raw_text", "grade_pred", "edema_pred",
     "answer", "p_yes", "grade_mass", "edema_mass", "seconds"]]'''),
("md", "## Grau previsto × grau real (linhas = real, colunas = previsto)"),
("code", r'''for m, g in out.groupby("model"):
    print(m)
    display(pd.crosstab(g.dr_icdr, g.grade_pred.fillna(-1).astype(int).rename("previsto (-1 = ilegível)")))'''),
("md", "## Resumo por modelo (piloto: só confere o funcionamento, não estima desempenho)"),
("code", r'''n_total = sum(len(c.query("split == 'eval'")) for c in cohorts.values())
out["yhat"] = (out.answer != "no").astype(int)       # inválida = encaminhar
rows = []
for m, g in out.groupby("model"):
    pos, neg = g[g.referable == 1], g[g.referable == 0]
    rows.append({"modelo": m, "n": len(g), "invalidas": int((g.answer == "invalid").sum()),
                 "p_yes_nan": int(g.p_yes.isna().sum()), "massa_graus": round(g.grade_mass.mean(), 3),
                 "massa_edema": round(g.edema_mass.mean(), 3), "edema_ilegivel": int(g.edema_pred.isna().sum()),
                 "seg_mediana": round(g.seconds.median(), 2),
                 "referenciaveis_encaminhados": f"{pos.yhat.sum()}/{len(pos)}",
                 "nao_ref_sem_encaminhar": f"{(neg.yhat == 0).sum()}/{len(neg)}"})
res = pd.DataFrame(rows).merge(info, on="modelo")
res["horas_por_prompt"] = (res.seg_mediana * n_total / 3600).round(1)
res'''),
("md", """**Se P(sim) sair `nan` ou as respostas vierem estranhas no MPS** para um modelo, troque o tipo numérico só dele:
`vlm = infer.VLM("<modelo>", dtype="float16")` (ou `"float32"`, que usa o dobro de memória), rode a célula de inferência para ele e `vlm.close()`.

Os arquivos `*_piloto.csv` ficam em `results/preds/` e **não** entram na análise. A inferência retoma de onde parou:
para rodar de novo com outro prompt, mova os `*_piloto.csv` antigos para outra pasta."""),
]

# ------------------------------------------------------------------------------------------- 03
books["03_inferencia_gpu"] = [
("md", """# 03 · Inferência na GPU (nuvem): MedGemma 27B

Só o modelo que não cabe no Mac (MedGemma 27B, ~54 GB em bf16). Os outros três rodam no Mac (`03b_inferencia_local`).
Etapas (`vlmdr.runner`):
0. **checagem no piloto**: 20 imagens por base (`run = piloto_cuda`, fora das análises). Se houver P(referenciável) `nan`,
   mais de 10% de respostas inválidas ou massa média nos graus < 0,9, o modelo é pulado e o motivo vai para o log;
1. prompt principal (`main_en`) nas **duas bases primeiro** (desfecho primário);
2. prompts alternativos (`nodef_en`, `main_pt`) na subamostra de robustez;
3. segunda execução do prompt principal na subamostra de reprodutibilidade (`run = r2`).

Grava a cada imagem e retoma de onde parou. **Rode pelo `rodar_nuvem.sh`** (veja o README): ele executa este notebook
sem depender do navegador, para no limite `MAX_HORAS`, empacota os resultados em `resultados_nuvem.tar.gz` e para a
máquina pela API da GPU Brasil (o disco é mantido)."""),
("code", SETUP + "\nfrom vlmdr import cohort, infer, prompts\nfrom vlmdr.runner import Runner"),
("code", r'''import os, shutil, subprocess, tarfile, traceback
import torch, transformers
from huggingface_hub import constants as hf_constants, whoami

MODELOS = os.environ.get("MODELOS", "medgemma_27b").split(",")
LIMITE = int(os.environ["LIMITE"]) if os.environ.get("LIMITE") else None   # ex.: 5 para um ensaio rápido
ENCERRAR = os.environ.get("ENCERRAR", "1") == "1"   # o rodar_nuvem.sh põe 0 e para a máquina ele mesmo
ENCERRAR_CMD = os.environ.get("ENCERRAR_CMD", "python3 ../gpubrasil.py parar")   # shutdown NÃO para a cobrança

assert torch.cuda.is_available(), "sem GPU CUDA: este notebook é só para a máquina da nuvem"
print("GPU:", torch.cuda.get_device_name(0), "| VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9))
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| transformers", transformers.__version__)
print("disco livre (GB):", round(shutil.disk_usage(config.ROOT).free / 1e9), "| cache dos modelos:", hf_constants.HF_HUB_CACHE)
print("Hugging Face:", whoami()["name"], "| modelos:", MODELOS, "| limite:", LIMITE)
cohorts = {ds: cohort.load(ds) for ds in ("brset", "mbrset")}
for ds, c in cohorts.items():
    falta = int((~c.image_path.map(lambda p: Path(p).exists())).sum())
    print(f"{ds}: {len(c)} imagens, {falta} não encontradas")
    assert falta == 0, f"imagens do {ds} não encontradas: confira {ds.upper()}_IMAGES"
runner = Runner(cohorts, "gpu", limite=LIMITE, limpar_cache=len(MODELOS) > 1)'''),
("md", "## Execução (empacota os resultados e para a máquina no fim, mesmo se der erro)"),
("code", r'''try:
    for m in MODELOS:
        try:
            runner.roda(m)
        except Exception as e:
            runner.registra(modelo=m, etapa="erro", problemas=f"{type(e).__name__}: {e}")
            traceback.print_exc()
finally:
    destino = config.ROOT / "resultados_nuvem.tar.gz"
    with tarfile.open(destino, "w:gz") as tar:
        tar.add(config.RESULTS, arcname="results")
    print("resultados em", destino, flush=True)
    if ENCERRAR:
        print("parando a máquina:", ENCERRAR_CMD, flush=True)
        subprocess.run(ENCERRAR_CMD, shell=True)
pd.DataFrame(runner.log)'''),
]

books["03b_inferencia_local"] = [
("md", """# 03b · Inferência local (Mac, MPS): MedGemma 4B, Gemma 3 4B, Qwen2.5-VL 7B

Mesmas etapas e mesma ordem do notebook 03 (`vlmdr.runner`): checagem no piloto (`run = piloto_mps`), principal nas
duas bases, robustez, reprodutibilidade. Grava a cada imagem e retoma de onde parou.
Rode com o Mac na tomada e `caffeinate -dims` num Terminal."""),
("code", SETUP + "\nfrom vlmdr import cohort, infer\nfrom vlmdr.runner import Runner"),
("code", r'''MODELOS = ["medgemma_4b", "gemma3_4b", "qwen25vl_7b"]
LIMITE = None          # ex.: 30 para um ensaio rápido; None = amostra inteira
print("dispositivo:", infer.device_name())
cohorts = {ds: cohort.load(ds) for ds in ("brset", "mbrset")}
runner = Runner(cohorts, "local", limite=LIMITE)'''),
("code", r'''for m in MODELOS:
    runner.roda(m)
pd.DataFrame(runner.log)'''),
]

# ------------------------------------------------------------------------------------------- 04
books["04_analise"] = [
("md", """# 04 · Análise

O modelo dá o grau ICDR (0–4) e diz se há edema macular.

**Desfecho primário: grau ICDR.** Concordância entre o grau previsto e o de referência: kappa ponderado quadrático
(principal), kappa linear, acerto exato e erro de até 1 grau; matriz de confusão; comparação pareada entre modelos
(H1: MedGemma 4B vs. Gemma 3 4B) pela diferença de kappa. Respostas sem grau legível ficam fora e são contadas.

**Desfecho secundário: RD referenciável** = ICDR ≥ 2 e/ou edema macular (derivado da resposta; inválida = encaminhar).
Sensibilidade, especificidade, AUROC/AUPRC de P(referenciável) = 1 − P(grau 0 ou 1) × [1 − P(edema)] (coluna `p_yes`),
especificidade com sensibilidade fixa (limiar na metade A, avaliação na metade B), sensibilidade por grau, DeLong e McNemar.
Também: concordância entre prompts e entre execuções, linha de base supervisionada e eficiência.

Os dois olhos do paciente entram na análise: ICs por bootstrap por paciente, DeLong agrupado (Obuchowski),
McNemar agrupado (Durkalski) e metades A/B divididas por paciente."""),
("code", SETUP + "\nfrom vlmdr import cohort, infer, metrics\nfrom sklearn.metrics import cohen_kappa_score"),
("code", r'''cohorts = {ds: cohort.load(ds).query("split == 'eval'") for ds in ("brset", "mbrset")}
p = infer.load_preds()
p = p[~p.run.astype(str).str.startswith("piloto")]
p["yhat"] = (p.answer != "no").astype(int)           # inválida = encaminhar
p["yhat_valid"] = p.answer.map({"yes": 1, "no": 0})  # análise de sensibilidade (exclui inválidas)
main = p[(p.prompt == "main_en") & (p.run == "r1")]
d = pd.concat([cohorts[ds].merge(g, on=["image_id", "patient_id", "dataset"]) for ds, g in main.groupby("dataset")])
d.groupby(["dataset", "model"]).size().unstack()'''),
("md", "## Desfecho primário: grau ICDR (0–4)"),
("code", r'''rows = []
for (ds, m), g in d.groupby(["dataset", "model"]):
    ok = g[g.grade_pred.notna()]
    yt, yp, pid = ok.dr_icdr.astype(int), ok.grade_pred.astype(int), ok.patient_id
    rows.append({"base": ds, "modelo": m, "n": len(g), "pacientes": g.patient_id.nunique(),
                 "sem grau legível": int(g.grade_pred.isna().sum()),
                 "kappa quadrático": metrics.fmt(*metrics.kappa(yt, yp, "quadratic", groups=pid), pct=False),
                 "kappa linear": metrics.fmt(*metrics.kappa(yt, yp, "linear", groups=pid), pct=False),
                 "acerto exato %": metrics.fmt(*metrics.proportion((yt == yp).astype(int), groups=pid)),
                 "erro de até 1 grau %": metrics.fmt(*metrics.proportion(((yt - yp).abs() <= 1).astype(int), groups=pid))})
grau = pd.DataFrame(rows)
config.TABLES.mkdir(parents=True, exist_ok=True)
grau.to_csv(config.TABLES / "primario_grau.csv", index=False)
grau'''),
("md", "### Matriz de confusão (linhas = grau real; colunas = previsto; -1 = sem grau legível) e % de acerto por grau real"),
("code", r'''for (ds, m), g in d.groupby(["dataset", "model"]):
    print(ds, m)
    cm = pd.crosstab(g.dr_icdr.rename("real"), g.grade_pred.fillna(-1).astype(int).rename("previsto"))
    cm["acerto %"] = [round(100 * cm.loc[k, k] / cm.loc[k].sum(), 1) if k in cm.columns else 0.0 for k in cm.index]
    display(cm)'''),
("md", "### Comparações pareadas do grau (diferença de kappa quadrático; bootstrap pareado por paciente)"),
("code", r'''modelos = sorted(d.model.unique())
pares = [(x, y) for i, x in enumerate(modelos) for y in modelos[i + 1:]]
kq = lambda yt, yp: cohen_kappa_score(yt, yp, weights="quadratic", labels=metrics.ICDR)

def pareado_grau(ds, a, b):
    w = d[d.dataset == ds].pivot_table(index=["image_id", "patient_id", "dr_icdr"], columns="model",
                                       values="grade_pred").dropna()
    y = w.index.get_level_values("dr_icdr").values.astype(int)
    pid = w.index.get_level_values("patient_id").values
    est, lo, hi, p = metrics.paired_diff(kq, y, w[a].values.astype(int), w[b].values.astype(int), groups=pid)
    return {"base": ds, "A": a, "B": b, "n": len(w), "pacientes": len(set(pid)),
            "kappa A": round(kq(y, w[a].values.astype(int)), 3), "kappa B": round(kq(y, w[b].values.astype(int)), 3),
            "diferença (IC 95%)": metrics.fmt(est, lo, hi, pct=False), "p": p}

comp_grau = pd.DataFrame([pareado_grau(ds, a, b) for ds in d.dataset.unique() for a, b in pares])
comp_grau.to_csv(config.TABLES / "primario_grau_pareado.csv", index=False)
comp_grau'''),
("md", "## Desfecho secundário: RD referenciável"),
("code", r'''rows = []
for (ds, m), g in d.groupby(["dataset", "model"]):
    r = metrics.sens_spec(g.referable, g.yhat, groups=g.patient_id)
    auc = metrics.auroc(g.referable, g.p_yes, groups=g.patient_id)
    rows.append({"base": ds, "modelo": m, "n": r["n"], "pacientes": r["pacientes"], "positivos": r["positivos"],
                 "sensibilidade %": metrics.fmt(r["sens"], r["sens_lo"], r["sens_hi"]),
                 "especificidade %": metrics.fmt(r["spec"], r["spec_lo"], r["spec_hi"]),
                 "AUROC": metrics.fmt(*auc, pct=False),
                 "inválidas": int((g.answer == "invalid").sum())})
sec = pd.DataFrame(rows)
sec.to_csv(config.TABLES / "secundario_referenciavel.csv", index=False)
sec'''),
("md", "## Comparações pareadas do referenciável dentro de cada base (McNemar e DeLong agrupados)"),
("code", r'''def pareado(ds, a, b):
    w = d[d.dataset == ds].pivot_table(index=["image_id", "patient_id", "referable"], columns="model",
                                       values=["yhat", "p_yes"]).dropna()
    y = w.index.get_level_values("referable").values
    pid = w.index.get_level_values("patient_id").values
    pos, neg = y == 1, y == 0
    p_sens = metrics.mcnemar(w["yhat"][a].values[pos] == 1, w["yhat"][b].values[pos] == 1, groups=pid[pos])
    p_spec = metrics.mcnemar(w["yhat"][a].values[neg] == 0, w["yhat"][b].values[neg] == 0, groups=pid[neg])
    a1, a2, diff, p_auc = metrics.delong(y, w["p_yes"][a].values, w["p_yes"][b].values, groups=pid)
    return {"base": ds, "A": a, "B": b, "n": len(w), "pacientes": len(set(pid)),
            "p McNemar sens": p_sens, "p McNemar espec": p_spec,
            "AUROC A": a1, "AUROC B": a2, "p DeLong": p_auc}

comp = pd.DataFrame([pareado(ds, a, b) for ds in d.dataset.unique() for a, b in pares])
comp'''),
("md", "## Ponto de operação do rastreio e triagem (limiar fixado na metade A, avaliado na metade B)"),
("code", r'''rows = []
for (ds, m), g in d.groupby(["dataset", "model"]):
    for alvo in (0.90, 0.95):
        r = metrics.operating_point(g.referable, g.p_yes, alvo, groups=g.patient_id)
        prev = g.referable.mean()
        ppv, npv = metrics.predictive_values(r["sens"], r["spec"], prev)
        rows.append({"base": ds, "modelo": m, "alvo": alvo,
                     "sens %": metrics.fmt(r["sens"], r["sens_lo"], r["sens_hi"]),
                     "espec %": metrics.fmt(r["spec"], r["spec_lo"], r["spec_hi"]),
                     "VPP": round(ppv, 3), "VPN": round(npv, 3),
                     "sem leitura %": round(r["triagem_sem_leitura_%"], 1), "perdidos/1000": round(r["perdidos_por_1000"], 1)})
pd.DataFrame(rows)'''),
("md", "## Edema macular"),
("code", r'''rows = []
for (ds, m), g in d.groupby(["dataset", "model"]):
    e = g.edema_pred.notna()
    r = metrics.sens_spec(g.macular_edema[e], (g.edema_pred[e] == "yes").astype(int), groups=g.patient_id[e])
    rows.append({"base": ds, "modelo": m, "edema legível": int(e.sum()), "com edema (real)": r["positivos"],
                 "sensibilidade %": metrics.fmt(r["sens"], r["sens_lo"], r["sens_hi"]),
                 "especificidade %": metrics.fmt(r["spec"], r["spec_lo"], r["spec_hi"]),
                 "kappa": round(cohen_kappa_score(g.macular_edema[e], (g.edema_pred[e] == "yes").astype(int)), 3)})
pd.DataFrame(rows)'''),
("md", "## Sensibilidade para referenciável por grau real"),
("code", r'''pd.concat([metrics.sens_by_grade(g).assign(base=ds, modelo=m) for (ds, m), g in d.groupby(["dataset", "model"])])'''),
("md", "## Robustez ao prompt e reprodutibilidade"),
("code", r'''def concordancia(p1, r1, p2, r2):
    cols = ["model", "dataset", "image_id", "grade_pred", "answer"]
    a = p[(p.prompt == p1) & (p.run == r1)][cols]
    b = p[(p.prompt == p2) & (p.run == r2)][cols]
    m = a.merge(b, on=["model", "dataset", "image_id"], suffixes=("_a", "_b"))
    out = []
    for (mod, ds), g in m.groupby(["model", "dataset"]):
        gg = g.dropna(subset=["grade_pred_a", "grade_pred_b"])
        out.append({"comparação": f"{p1}/{r1} vs {p2}/{r2}", "modelo": mod, "base": ds, "n": len(g),
                    "grau igual %": round(100 * (gg.grade_pred_a == gg.grade_pred_b).mean(), 1),
                    "kappa quadrático (grau)": round(cohen_kappa_score(gg.grade_pred_a.astype(int), gg.grade_pred_b.astype(int),
                                                                       weights="quadratic", labels=metrics.ICDR), 3),
                    "referenciável igual %": round(100 * (g.answer_a == g.answer_b).mean(), 1),
                    "kappa (referenciável)": round(cohen_kappa_score(g.answer_a, g.answer_b), 3)})
    return out

pd.DataFrame(concordancia("main_en", "r1", "nodef_en", "r1") + concordancia("main_en", "r1", "main_pt", "r1")
             + concordancia("main_en", "r1", "main_en", "r2"))'''),
("md", "## Linha de base supervisionada (escolhida no notebook 01; só o desfecho secundário, referenciável)"),
("code", r'''EMBEDDING = "dinov3_convnext_tiny"   # troque pela escolha registrada
bl = pd.read_csv(config.PREDS / f"baseline_{EMBEDDING}.csv", dtype={"image_id": str})
rows = []
for (setting, ds), g in bl.groupby(["setting", "dataset"]):
    c = cohorts[ds].merge(g, on="image_id")
    rows.append({"cenário": setting, "base": ds, "n": len(c), "pacientes": c.patient_id.nunique(),
                 "AUROC": metrics.fmt(*metrics.auroc(c.referable, c.p, groups=c.patient_id), pct=False)})
pd.DataFrame(rows)'''),
("md", "## Eficiência (tempo, tokens)"),
("code", r'''main.groupby(["model", "dataset"]).agg(seg_mediana=("seconds", "median"), tokens_in=("n_tokens_in", "median"),
                                      tokens_out=("n_tokens_out", "median"), dispositivo=("device", "first"))'''),
]


# ------------------------------------------------------------------------------- 05 (artigo 2 + checagem)
ART2_MD = """**Experimentos extras** (run = `a2`, separados dos arquivos do artigo 1):
- **Artigo 2, prompt em português completo:** `main_pt` na amostra de avaliação inteira das duas bases.
- **Artigo 2, 2×2 de língua** (subamostra de robustez, 300 imagens por base): instrução em português com formato de
  resposta em inglês (`pt_instr_en_fmt`) e instrução em inglês com formato em português (`en_instr_pt_fmt`). As outras
  duas células do 2×2 são `main_en`/`r1` (já rodado) e `main_pt`/`a2` (acima). O prefill `ICDR: ` é o mesmo nas quatro.
"""

books["05_extra_gpu"] = [
("md", """# 05 · Experimentos extras na GPU (nuvem)

1. **Checagem MPS × CUDA do artigo 1:** MedGemma 4B nas 40 imagens do piloto em CUDA (`run = piloto_cuda`), para
   comparar com as mesmas imagens já rodadas no Mac (`piloto_mps`). Só a checagem; nada da amostra de avaliação.
2. **MedGemma 27B:** checagem no piloto, depois os experimentos do artigo 2.

""" + ART2_MD + """
Rode pelo `rodar_nuvem.sh` com `NOTEBOOK=notebooks/05_extra_gpu.ipynb` (veja o README). Grava a cada imagem e retoma
de onde parou; no fim empacota `results/` e para a máquina. Tempo estimado na A100: ~2,5 h (≈1 s por imagem no 27B)."""),
("code", SETUP + "\nfrom vlmdr import cohort, infer, prompts\nfrom vlmdr.runner import Runner"),
("code", r'''import os, shutil, subprocess, tarfile, traceback
import torch, transformers
from huggingface_hub import constants as hf_constants, whoami

# etapas por modelo (o 4B só faz a checagem no piloto)
PLANO = {
    "medgemma_4b": ("piloto",),
    "medgemma_27b": ("piloto", "pt_completo", "dois_por_dois"),
}
MODELOS = os.environ.get("MODELOS", ",".join(PLANO)).split(",")
LIMITE = int(os.environ["LIMITE"]) if os.environ.get("LIMITE") else None   # ex.: 5 para um ensaio rápido
ENCERRAR = os.environ.get("ENCERRAR", "1") == "1"   # o rodar_nuvem.sh põe 0 e para a máquina ele mesmo
ENCERRAR_CMD = os.environ.get("ENCERRAR_CMD", "python3 ../gpubrasil.py parar")   # shutdown NÃO para a cobrança

assert torch.cuda.is_available(), "sem GPU CUDA: este notebook é só para a máquina da nuvem"
print("GPU:", torch.cuda.get_device_name(0), "| VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9))
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| transformers", transformers.__version__)
print("disco livre (GB):", round(shutil.disk_usage(config.ROOT).free / 1e9), "| cache dos modelos:", hf_constants.HF_HUB_CACHE)
print("Hugging Face:", whoami()["name"], "| modelos:", MODELOS, "| limite:", LIMITE)
pr = prompts.load_prompts()
assert {"main_pt", "pt_instr_en_fmt", "en_instr_pt_fmt"} <= set(pr), "prompts do artigo 2 ausentes em settings/prompts.yaml"
cohorts = {ds: cohort.load(ds) for ds in ("brset", "mbrset")}
for ds, c in cohorts.items():
    falta = int((~c.image_path.map(lambda p: Path(p).exists())).sum())
    print(f"{ds}: {len(c)} imagens, {falta} não encontradas")
    assert falta == 0, f"imagens do {ds} não encontradas: confira {ds.upper()}_IMAGES"
runner = Runner(cohorts, "gpu_extra", limite=LIMITE, limpar_cache=len(MODELOS) > 1)'''),
("md", "## Execução (empacota os resultados e para a máquina no fim, mesmo se der erro)"),
("code", r'''try:
    for m in MODELOS:
        try:
            runner.roda(m, etapas=PLANO[m])
        except Exception as e:
            runner.registra(modelo=m, etapa="erro", problemas=f"{type(e).__name__}: {e}")
            traceback.print_exc()
finally:
    destino = config.ROOT / "resultados_nuvem.tar.gz"
    with tarfile.open(destino, "w:gz") as tar:
        tar.add(config.RESULTS, arcname="results")
    print("resultados em", destino, flush=True)
    if ENCERRAR:
        print("parando a máquina:", ENCERRAR_CMD, flush=True)
        subprocess.run(ENCERRAR_CMD, shell=True)
pd.DataFrame(runner.log)'''),
]

books["05b_extra_local"] = [
("md", """# 05b · Experimentos extras no Mac (MPS): MedGemma 4B

""" + ART2_MD + """
Tempo estimado: ~2,5 h para o `main_pt` completo (≈1,9 s por imagem) e ~40 min para o 2×2.
Rode com o Mac na tomada e `caffeinate -dims` num Terminal. Grava a cada imagem e retoma de onde parou.

No fim, a última seção compara a checagem do piloto feita na nuvem (`piloto_cuda`) com a do Mac (`piloto_mps`),
depois que os resultados da nuvem forem copiados para `results/preds/`."""),
("code", SETUP + "\nfrom vlmdr import cohort, infer, prompts\nfrom vlmdr.runner import Runner"),
("code", r'''MODELOS = ["medgemma_4b"]
ETAPAS = ("pt_completo", "dois_por_dois")
LIMITE = None          # ex.: 20 para um ensaio rápido; None = amostra inteira
print("dispositivo:", infer.device_name())
pr = prompts.load_prompts()
assert {"main_pt", "pt_instr_en_fmt", "en_instr_pt_fmt"} <= set(pr), "prompts do artigo 2 ausentes em settings/prompts.yaml"
cohorts = {ds: cohort.load(ds) for ds in ("brset", "mbrset")}
runner = Runner(cohorts, "local_extra", etapas=ETAPAS, limite=LIMITE)'''),
("code", r'''for m in MODELOS:
    runner.roda(m)
pd.DataFrame(runner.log)'''),
("md", """## Conferência: `main_pt` no artigo 1 (`r1`, 300 imagens) × artigo 2 (`a2`)

Mesmas imagens, mesmo prompt: os graus devem ser idênticos (a execução é determinística)."""),
("code", r'''p = infer.load_preds("vlm_medgemma_4b_*_main_pt_*.csv")
w = p.pivot_table(index=["dataset", "image_id"], columns="run", values="grade_pred", aggfunc="first").dropna()
print("imagens nas duas execuções:", len(w), "| graus idênticos:", f"{(w['r1'] == w['a2']).mean():.1%}")'''),
("md", """## Checagem MPS × CUDA (MedGemma 4B, piloto)

Precisa dos arquivos `vlm_medgemma_4b_*_main_en_piloto_cuda.csv` vindos da nuvem (notebook 05)."""),
("code", r'''rows = []
for ds in ("brset", "mbrset"):
    a, b = (config.PREDS / f"vlm_medgemma_4b_{ds}_main_en_piloto_{d}.csv" for d in ("mps", "cuda"))
    if not (a.exists() and b.exists()):
        print("faltando:", b.name if a.exists() else a.name)
        continue
    x = pd.read_csv(a, dtype={"image_id": str}).merge(pd.read_csv(b, dtype={"image_id": str}), on="image_id", suffixes=("_mps", "_cuda"))
    rows.append({"base": ds, "n": len(x), "grau idêntico %": 100 * (x.grade_pred_mps == x.grade_pred_cuda).mean(),
                 "edema idêntico %": 100 * (x.edema_pred_mps == x.edema_pred_cuda).mean(),
                 "max |Δ P(referenciável)|": (x.p_yes_mps - x.p_yes_cuda).abs().max()})
pd.DataFrame(rows)'''),
]


# ------------------------------------------------------------------------------- 06 (erros, vieses, língua)
books["06_erros_vieses"] = [
("md", """# 06 · Tipos de erro, vieses entre grupos e efeito da língua

**Parte A, artigo 1 (exploratória): para que lado o modelo erra.** Superestima × subestima o grau, erro médio com
sinal por grau real e erros de impacto clínico: encaminhamento perdido (falso negativo), doença grave (ICDR 3–4)
classificada como 0–1, encaminhamento desnecessário (falso positivo) e olho normal classificado como referenciável.

**Parte B, artigo 1 (exploratória): vieses entre grupos.** Sexo e faixa etária (duas bases); escolaridade e plano
de saúde (só mBRSET; códigos do PhysioNet: 1 analfabeto … 7 superior completo; 0/1 sem/com plano). Diferenças brutas
podem vir só da mistura de gravidade; por isso a regressão logística (GEE por paciente) é **ajustada pelo grau real e
pelo edema**. Grupos com menos de 50 referenciáveis entram só como descrição.

**Parte C, artigo 2: a língua do prompt muda os tipos de erro?** `main_en` × `main_pt` nas mesmas imagens: taxa de cada
erro, diferença pareada (IC por bootstrap por paciente, McNemar agrupado), transição de graus e erro com sinal por grau.
Depois o 2×2 (instrução × formato) e se o efeito do português é igual entre grupos de escolaridade.

Inválida = encaminhar (como no artigo 1). ICs: bootstrap por paciente, 2.000 reamostras."""),
("code", SETUP + "\nfrom vlmdr import errors, metrics"),
("code", r'''MODELOS = ["medgemma_4b", "medgemma_27b"]   # Gemma 3 e Qwen colapsam numa resposta só: tipos de erro sem sentido
GRUPOS = ["sexo", "faixa_etaria", "escolaridade", "plano_saude"]
d = errors.load(MODELOS)
config.TABLES.mkdir(parents=True, exist_ok=True)
print(d.groupby(["dataset", "model"]).size().unstack())
for v in GRUPOS:
    if v in d:
        print(f"\n{v} (pacientes):", d.drop_duplicates(["dataset", "patient_id"]).groupby("dataset")[v].value_counts().to_dict())'''),
("md", "## A1 · Taxas de cada tipo de erro (prompt principal)"),
("code", r'''erros = errors.error_table(d)
erros.to_csv(config.TABLES / "erros_tipos.csv", index=False)
erros[[c for c in erros.columns if not c.endswith(" n")]]'''),
("md", "## A2 · Erro médio com sinal por grau real (negativo = subestima a doença)"),
("code", r'''se = errors.signed_error_by_grade(d)
se.to_csv(config.TABLES / "erros_com_sinal_por_grau.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, (ds, g) in zip(axes, se.groupby("dataset")):
    for m, h in g.groupby("model"):
        ax.errorbar(h.icdr + (0.08 if m.endswith("27b") else -0.08), h["erro médio"],
                    yerr=[h["erro médio"] - h.lo, h.hi - h["erro médio"]], marker="o", capsize=3, label=m)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(title=ds, xlabel="grau ICDR real", xticks=range(5))
axes[0].set_ylabel("erro médio (previsto − real)")
axes[0].legend(frameon=False)
config.FIGURES.mkdir(exist_ok=True)
fig.tight_layout(); fig.savefig(config.FIGURES / "erro_com_sinal_por_grau.png", dpi=200)
se.pivot_table(index=["dataset", "model"], columns="icdr", values="erro médio").round(2)'''),
("md", "## B1 · Desempenho e erros por grupo"),
("code", r'''sub = []
for ds in ("brset", "mbrset"):
    for m in MODELOS:
        for v in GRUPOS:
            if v in d and d.loc[d.dataset == ds, v].notna().any():
                t = errors.subgroup_table(d, v, m, ds)
                sub.append(t.assign(base=ds, modelo=m, variável=v))
sub = pd.concat(sub, ignore_index=True)
sub = sub[["base", "modelo", "variável"] + [c for c in sub.columns if c not in ("base", "modelo", "variável")]]
sub.to_csv(config.TABLES / "vieses_por_grupo.csv", index=False)
sub'''),
("md", """## B2 · Regressão ajustada pelo grau real (GEE, agrupada por paciente)

OR > 1 = o grupo tem mais desse erro do que o grupo de referência (o mais frequente), **com a mesma gravidade**."""),
("code", r'''adj = []
for ds in ("brset", "mbrset"):
    for m in MODELOS:
        for v in GRUPOS:
            if v in d and d.loc[d.dataset == ds, v].notna().any():
                for e in ("encaminhamento_perdido", "encaminhamento_desnecessario"):
                    adj.append(errors.adjusted_odds(d, v, m, ds, e))
adj = pd.concat(adj, ignore_index=True)
adj.to_csv(config.TABLES / "vieses_or_ajustada.csv", index=False)
adj.round(3)'''),
("md", """## C1 · Inglês × português nas mesmas imagens: muda o tipo de erro?

Usa `main_pt`/`a2` (amostra completa, notebooks 05); se ainda não houver, cai para `main_pt`/`r1` (300 por base)."""),
("code", r'''w = errors.paired_languages(MODELOS)
print("prompts comparados:", w.attrs["a"], "×", w.attrs["b"], "| imagens:", w.groupby(["dataset", "model"]).size().to_dict())
ling = errors.language_error_shift(w)
ling.to_csv(config.TABLES / "lingua_tipos_de_erro.csv", index=False)
ling'''),
("md", "### Transição de graus (linhas = inglês, colunas = português; -1 = sem grau legível)"),
("code", r'''for ds in ("brset", "mbrset"):
    for m in MODELOS:
        print(ds, m)
        display(errors.grade_transitions(w, m, ds))'''),
("md", "### Erro com sinal por grau real: inglês × português"),
("code", r'''def por_lingua(w, lado):
    x = w.rename(columns={f"grade_pred_{lado}": "grade_pred", f"erro_{lado}": "erro"})
    return errors.signed_error_by_grade(x).assign(prompt=w.attrs[lado][0])
se_l = pd.concat([por_lingua(w, "a"), por_lingua(w, "b")])
se_l.to_csv(config.TABLES / "lingua_erro_com_sinal_por_grau.csv", index=False)
fig, axes = plt.subplots(len(MODELOS), 2, figsize=(10, 3.4 * len(MODELOS)), sharey=True, squeeze=False)
for i, m in enumerate(MODELOS):
    for j, ds in enumerate(("brset", "mbrset")):
        ax = axes[i, j]
        for pk, h in se_l[(se_l.model == m) & (se_l.dataset == ds)].groupby("prompt"):
            ax.errorbar(h.icdr + (0.08 if pk.endswith("pt") else -0.08), h["erro médio"],
                        yerr=[h["erro médio"] - h.lo, h.hi - h["erro médio"]], marker="o", capsize=3, label=pk)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set(title=f"{m} · {ds}", xlabel="grau ICDR real", xticks=range(5))
    axes[i, 0].set_ylabel("erro médio (previsto − real)")
axes[0, 0].legend(frameon=False)
fig.tight_layout(); fig.savefig(config.FIGURES / "lingua_erro_com_sinal_por_grau.png", dpi=200)
se_l.pivot_table(index=["model", "dataset", "prompt"], columns="icdr", values="erro médio").round(2)'''),
("md", """## C2 · 2×2: o efeito vem da instrução ou do formato da resposta?

Subamostra de robustez, imagens com as quatro células. Efeito principal = média, sobre o outro fator, da mudança
para o português (kappa: em pontos de kappa; erros: em pontos percentuais). Precisa dos notebooks 05."""),
("code", r'''cel, efeitos = errors.two_by_two(MODELOS)
if len(cel):
    cel.to_csv(config.TABLES / "lingua_2x2_celulas.csv", index=False)
    efeitos.to_csv(config.TABLES / "lingua_2x2_efeitos.csv", index=False)
    display(cel, efeitos)'''),
("md", """## C3 · O efeito do português é igual entre grupos de escolaridade? (mBRSET)

Diferença PT − EN na taxa de encaminhamento perdido e no erro médio dos ICDR ≥ 2, por grupo. Exploratório."""),
("code", r'''rows = []
if "escolaridade" in w:
    fa = errors.flags(w.rename(columns=lambda c: c[:-2] if c.endswith("_a") else c))
    fb = errors.flags(w.rename(columns=lambda c: c[:-2] if c.endswith("_b") else c))
    for (m, grp), g in w[(w.dataset == "mbrset") & w.escolaridade.notna()].groupby(["model", "escolaridade"]):
        x = (fb.loc[g.index, "encaminhamento_perdido"] - fa.loc[g.index, "encaminhamento_perdido"]).dropna()
        e = g[g.dr_icdr >= 2]
        est, lo, hi = errors._mean_ci(x.values, g.loc[x.index, "patient_id"].values) if len(x) else (np.nan,) * 3
        rows.append({"modelo": m, "escolaridade": grp, "referenciáveis": len(x),
                     "Δ encaminhamento perdido (pontos %)": metrics.fmt(est, lo, hi) if len(x) else "",
                     "Δ erro médio ICDR>=2": round((e.erro_b - e.erro_a).mean(), 2)})
pd.DataFrame(rows)'''),
]

def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s) for t, s in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = NB / f"{name}.ipynb"
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
    if only and "--run" in only:
        only = only[:only.index("--run")]
    # --only 05 05b: gera só esses (não sobrescreve notebooks já executados, como o 04)
    paths = {n.split("_")[0]: build(n, c) for n, c in books.items() if not only or n.split("_")[0] in only}
    print("gerados:", ", ".join(p.name for p in paths.values()))
    if "--run" in sys.argv:
        from nbclient import NotebookClient
        for key in sys.argv[sys.argv.index("--run") + 1:]:
            nb = nbf.read(paths[key], as_version=4)
            NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
            nbf.write(nb, paths[key])
            print("executado:", paths[key].name)
