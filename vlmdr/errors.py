"""Tipos de erro, vieses entre grupos e efeito da língua do prompt (notebook 06).

Unidade = imagem; ICs por bootstrap por paciente (as imagens do mesmo paciente são reamostradas juntas).

Categorias de erro (todas 0/1 por imagem):
  superestima / subestima      : grau previsto acima / abaixo do real (sem grau legível fica fora das duas)
  sem_grau                     : resposta sem grau legível
  encaminhamento_perdido       : RD referenciável real e o modelo não encaminha (falso negativo)  [só referenciáveis]
  grave_como_leve              : ICDR real 3-4 e grau previsto 0-1 (erro clinicamente grave)      [só ICDR 3-4]
  encaminhamento_desnecessario : não referenciável real e o modelo encaminha (falso positivo)     [só não referenciáveis]
  normal_como_referenciavel    : ICDR real 0 sem edema e grau previsto >= 2                        [só ICDR 0 sem edema]
"""
import numpy as np
import pandas as pd

import config
from . import cohort, infer, metrics

EDU = {1: "até fundamental incompleto", 2: "até fundamental incompleto", 3: "fundamental completo a médio incompleto",
       4: "fundamental completo a médio incompleto", 5: "médio completo ou mais", 6: "médio completo ou mais",
       7: "médio completo ou mais"}          # mBRSET: 1 analfabeto … 7 superior completo (documentação do PhysioNet)
INSURANCE = {0: "sem plano (SUS)", 1: "com plano de saúde"}
AGE_BINS, AGE_LABELS = [0, 50, 65, 200], ["<50", "50-64", "65+"]

# categoria -> (denominador, numerador)
CATEGORIES = {
    "superestima": (lambda d: d.grade_pred.notna(), lambda d: d.grade_pred > d.dr_icdr),
    "subestima": (lambda d: d.grade_pred.notna(), lambda d: d.grade_pred < d.dr_icdr),
    "sem_grau": (lambda d: pd.Series(True, index=d.index), lambda d: d.grade_pred.isna()),
    "encaminhamento_perdido": (lambda d: d.referable == 1, lambda d: d.yhat == 0),
    "grave_como_leve": (lambda d: d.dr_icdr >= 3, lambda d: d.grade_pred <= 1),
    "encaminhamento_desnecessario": (lambda d: d.referable == 0, lambda d: d.yhat == 1),
    "normal_como_referenciavel": (lambda d: (d.dr_icdr == 0) & (d.macular_edema == 0), lambda d: d.grade_pred >= 2),
}


# ------------------------------------------------------------------------------------------- dados
def add_groups(c):
    """Grupos para a análise de vieses: sexo e faixa etária (as duas bases); escolaridade e plano (só mBRSET)."""
    c = c.copy()
    c["sexo"] = c.sex_female.map({1: "feminino", 0: "masculino"})
    c["faixa_etaria"] = pd.cut(c.age, AGE_BINS, labels=AGE_LABELS, right=False).astype(str).replace("nan", np.nan)
    if (c.dataset == "mbrset").any():
        try:
            m = cohort.read_mbrset()
            m = pd.DataFrame({"image_id": m["file"].map(cohort.image_key),
                              "educational_level": pd.to_numeric(m["educational_level"], errors="coerce"),
                              "insurance": pd.to_numeric(m["insurance"], errors="coerce")})
            c = c.merge(m, on="image_id", how="left")
            c["escolaridade"] = c.educational_level.map(EDU)
            c["plano_saude"] = c.insurance.map(INSURANCE)
        except FileNotFoundError:
            print("Aviso: labels_mbrset.csv não encontrado; escolaridade e plano de saúde ficam de fora")
    return c


def load(models, prompt="main_en", run="r1"):
    """Coorte de avaliação (com grupos) + predições de um prompt/execução, para os modelos indicados."""
    c = pd.concat([cohort.load(ds).query("split == 'eval'") for ds in ("brset", "mbrset")], ignore_index=True)
    c = add_groups(c)
    p = infer.load_preds()
    p = p[(p.prompt == prompt) & (p.run == run) & p.model.isin(models)]
    d = c.merge(p, on=["image_id", "patient_id", "dataset"])
    d["yhat"] = (d.answer != "no").astype(int)          # inválida = encaminhar (como no artigo 1)
    d["erro"] = d.grade_pred - d.dr_icdr                # com sinal; nan sem grau legível
    return d


def flags(d):
    """Uma coluna 0/1 por categoria (nan fora do denominador)."""
    out = pd.DataFrame(index=d.index)
    for k, (den, num) in CATEGORIES.items():
        out[k] = np.where(den(d), num(d).astype(float), np.nan)
    return out


# ------------------------------------------------------------------------------------------- resumos
def _rate(x, g):
    ok = ~np.isnan(x)
    if ok.sum() == 0:
        return np.nan, np.nan, np.nan, 0
    p, lo, hi = metrics.proportion(x[ok].astype(int), groups=g[ok])
    return p, lo, hi, int(ok.sum())


def error_table(d, by=("model", "dataset")):
    """Taxa de cada categoria de erro (IC 95% por bootstrap por paciente)."""
    f = flags(d)
    rows = []
    for key, idx in d.groupby(list(by), observed=True).groups.items():
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        g = d.loc[idx, "patient_id"].values
        for k in CATEGORIES:
            p, lo, hi, n = _rate(f.loc[idx, k].values, g)
            row[f"{k} %"] = metrics.fmt(p, lo, hi) if n else ""
            row[f"{k} n"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def signed_error_by_grade(d, by=("model", "dataset")):
    """Erro médio com sinal (previsto - real) por grau real; negativo = subestima a doença."""
    rows = []
    for key, g in d[d.grade_pred.notna()].groupby(list(by) + ["dr_icdr"], observed=True):
        m, lo, hi = _mean_ci(g.erro.values, g.patient_id.values)
        rows.append({**dict(zip(list(by) + ["icdr"], key)), "n": len(g), "erro médio": m, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def _mean_ci(x, groups, n=config.BOOTSTRAP, seed=config.SEED, alpha=config.ALPHA):
    gi = metrics._codes(groups)
    s, k = np.bincount(gi, weights=x), np.bincount(gi)
    draw = np.random.default_rng(seed).integers(0, len(k), (n, len(k)))
    boot = s[draw].sum(axis=1) / k[draw].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(x)), lo, hi


# ------------------------------------------------------------------------------------------- vieses
def subgroup_table(d, var, model, dataset):
    """Por grupo: kappa quadrático, sensibilidade e especificidade (referenciável) e taxas de erro, com IC."""
    x = d[(d.model == model) & (d.dataset == dataset) & d[var].notna()]
    f = flags(x)
    rows = []
    for lvl, g in x.groupby(var, observed=True):
        ok = g[g.grade_pred.notna()]
        k = metrics.kappa(ok.dr_icdr, ok.grade_pred, "quadratic", groups=ok.patient_id)
        r = metrics.sens_spec(g.referable, g.yhat, groups=g.patient_id)
        row = {"grupo": lvl, "imagens": len(g), "pacientes": g.patient_id.nunique(), "referenciáveis": r["positivos"],
               "kappa quadrático": metrics.fmt(*k, pct=False),
               "sensibilidade %": metrics.fmt(r["sens"], r["sens_lo"], r["sens_hi"]),
               "especificidade %": metrics.fmt(r["spec"], r["spec_lo"], r["spec_hi"]),
               "erro médio nos ICDR>=2": round(g.loc[g.dr_icdr >= 2, "erro"].mean(), 2)}
        for c in ("encaminhamento_perdido", "grave_como_leve", "encaminhamento_desnecessario"):
            p, lo, hi, n = _rate(f.loc[g.index, c].values, g.patient_id.values)
            row[c + " %"] = metrics.fmt(p, lo, hi) if n else ""
        rows.append(row)
    return pd.DataFrame(rows)


def adjusted_odds(d, var, model, dataset, outcome="encaminhamento_perdido"):
    """Regressão logística (GEE, agrupada por paciente) da categoria de erro em função do grupo,
    ajustada pelo grau real (e pelo edema): compara pacientes com a mesma gravidade."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    x = d[(d.model == model) & (d.dataset == dataset) & d[var].notna()].copy()
    x["y"] = flags(x)[outcome]
    x = x[x.y.notna()].rename(columns={var: "grupo"})
    x["grau"] = x.dr_icdr.astype(str)
    if x.grupo.nunique() < 2 or x.y.sum() < 10:
        return pd.DataFrame([{"modelo": model, "base": dataset, "variável": var, "erro": outcome,
                              "nota": "poucos eventos ou um grupo só"}])
    ref = x.grupo.value_counts().index[0]               # grupo mais frequente = referência
    adj = " + C(grau)" if x.grau.nunique() > 1 else ""
    adj += " + macular_edema" if x.macular_edema.nunique() > 1 else ""
    import warnings
    formula = f"y ~ C(grupo, Treatment(reference={ref!r})){adj}"
    # correlação exchangeable entre os olhos; se não convergir (IC nan), independence (os EPs continuam robustos)
    for cs in (sm.cov_struct.Exchangeable(), sm.cov_struct.Independence()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.gee(formula, groups="patient_id", data=x, family=sm.families.Binomial(), cov_struct=cs).fit()
        if np.isfinite(fit.bse).all():
            break
    ci = fit.conf_int()
    rows = []
    for t in fit.params.index:
        if t.startswith("C(grupo"):
            rows.append({"modelo": model, "base": dataset, "variável": var, "erro": outcome, "referência": ref,
                         "grupo": t.split("[T.")[-1].rstrip("]"), "OR ajustada": np.exp(fit.params[t]),
                         "IC 95%": f"{np.exp(ci.loc[t, 0]):.2f}–{np.exp(ci.loc[t, 1]):.2f}", "p": fit.pvalues[t],
                         "eventos": int(x.y.sum()), "n": len(x), "correlação": type(fit.cov_struct).__name__})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- língua
def paired_languages(models, a=("main_en", "r1"), b=("main_pt", "a2")):
    """Mesmas imagens com dois prompts (a = inglês, b = português), lado a lado. Se main_pt/a2 (artigo 2, amostra
    completa) ainda não existir, cai para main_pt/r1 (subamostra de robustez do artigo 1) e avisa."""
    p = infer.load_preds()
    if b == ("main_pt", "a2") and not ((p.prompt == "main_pt") & (p.run == "a2") & p.model.isin(models)).any():
        print("Aviso: main_pt/a2 ainda não rodado; usando main_pt/r1 (subamostra de robustez, 300 por base)")
        b = ("main_pt", "r1")
    da, db = load(models, *a), load(models, *b)
    keep = ["image_id", "model", "grade_pred", "edema_pred", "answer", "yhat", "erro", "p_yes"]
    w = da.merge(db[keep], on=["image_id", "model"], suffixes=("_a", "_b"))
    w.attrs["a"], w.attrs["b"] = a, b
    return w


def language_error_shift(w):
    """Para cada categoria de erro: taxa com o prompt A e com o B nas mesmas imagens, diferença (B - A) com IC por
    bootstrap pareado por paciente e p do McNemar agrupado."""
    fa = flags(w.rename(columns=lambda c: c[:-2] if c.endswith("_a") else c))
    fb = flags(w.rename(columns=lambda c: c[:-2] if c.endswith("_b") else c))
    rows = []
    for (m, ds), g in w.groupby(["model", "dataset"]):
        for k in CATEGORIES:
            xa, xb = fa.loc[g.index, k].values, fb.loc[g.index, k].values
            ok = ~np.isnan(xa) & ~np.isnan(xb)
            if ok.sum() < 5:
                continue
            pid = g.patient_id.values[ok]
            est, lo, hi = _mean_ci(xb[ok] - xa[ok], pid)        # diferença pareada (B - A), bootstrap por paciente
            rows.append({"modelo": m, "base": ds, "erro": k, "n": int(ok.sum()),
                         f"{w.attrs['a'][0]} %": round(100 * xa[ok].mean(), 1),
                         f"{w.attrs['b'][0]} %": round(100 * xb[ok].mean(), 1),
                         "diferença (pontos %)": metrics.fmt(est, lo, hi),
                         "p McNemar": metrics.mcnemar(xa[ok] == 0, xb[ok] == 0, groups=pid)})
    return pd.DataFrame(rows)


def grade_transitions(w, model, dataset):
    """Grau com o prompt A (linhas) × grau com o prompt B (colunas), nas mesmas imagens."""
    g = w[(w.model == model) & (w.dataset == dataset)]
    return pd.crosstab(g.grade_pred_a.fillna(-1).astype(int).rename(w.attrs["a"][0]),
                       g.grade_pred_b.fillna(-1).astype(int).rename(w.attrs["b"][0]))


def two_by_two(models):
    """Experimento 2x2 (subamostra de robustez): instrução EN/PT × formato EN/PT. Taxas de erro por célula e os
    efeitos principais (média sobre o outro fator) de trocar a instrução e de trocar o formato para português."""
    cells = {("EN", "EN"): ("main_en", "r1"), ("PT", "PT"): ("main_pt", "a2"),
             ("PT", "EN"): ("pt_instr_en_fmt", "a2"), ("EN", "PT"): ("en_instr_pt_fmt", "a2")}
    p = infer.load_preds()
    have = {k: ((p.prompt == v[0]) & (p.run == v[1]) & p.model.isin(models)).any() for k, v in cells.items()}
    if not all(have.values()):
        print("Aviso: células do 2x2 ainda não rodadas:", [cells[k] for k, ok in have.items() if not ok])
        return pd.DataFrame(), pd.DataFrame()
    parts = []
    for (ins, fmt), (pk, run) in cells.items():
        x = load(models, pk, run)
        x = x[x.robust == 1]
        parts.append(x.assign(instrucao=ins, formato=fmt))
    d = pd.concat(parts, ignore_index=True)
    common = d.groupby(["model", "image_id"]).instrucao.transform("size") == 4    # imagens com as quatro células
    d = d[common]
    f = flags(d)
    d = pd.concat([d, f], axis=1)
    k_rows = []
    for (m, ds, ins, fmt), g in d.groupby(["model", "dataset", "instrucao", "formato"]):
        ok = g[g.grade_pred.notna()]
        k_rows.append({"modelo": m, "base": ds, "instrução": ins, "formato": fmt, "n": len(g),
                       "kappa quadrático": round(metrics._kappa("quadratic")(ok.dr_icdr, ok.grade_pred), 3),
                       **{f"{c} %": round(100 * g[c].mean(), 1) for c in CATEGORIES}})
    cell_tab = pd.DataFrame(k_rows)
    eff = []
    for (m, ds), g in cell_tab.groupby(["modelo", "base"]):
        t = g.set_index(["instrução", "formato"])
        for col in ["kappa quadrático"] + [f"{c} %" for c in CATEGORIES]:
            v = t[col]
            eff.append({"modelo": m, "base": ds, "medida": col,
                        "efeito da instrução em PT": ((v["PT", "EN"] - v["EN", "EN"]) + (v["PT", "PT"] - v["EN", "PT"])) / 2,
                        "efeito do formato em PT": ((v["EN", "PT"] - v["EN", "EN"]) + (v["PT", "PT"] - v["PT", "EN"])) / 2,
                        "interação": (v["PT", "PT"] - v["PT", "EN"]) - (v["EN", "PT"] - v["EN", "EN"])})
    return cell_tab, pd.DataFrame(eff).round(3)
