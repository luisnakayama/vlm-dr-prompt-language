"""Prompts, modelos e leitura das respostas."""
import re
import unicodedata

import yaml

import config


def load_prompts():
    return yaml.safe_load(config.PROMPTS_YAML.read_text(encoding="utf-8"))


def load_models():
    return yaml.safe_load(config.MODELS_YAML.read_text(encoding="utf-8"))


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]+", " ", s).split()


GRADES = ["0", "1", "2", "3", "4"]


def parse(text, prompt):
    """Lê a continuação depois do prefill (ex.: '2; Macular edema: No').

    Retorna {"grade": 0-4 ou None, "edema": "yes" | "no" | None, "answer": "yes" | "no" | "invalid"}, em que
    answer = referenciável (ICDR >= 2 e/ou edema). Grau ausente, ou grau < 2 sem edema legível, = inválida.
    """
    m = re.match(r"^\W*([0-4])(?!\d|[.,]\d)", str(text))
    grade = int(m.group(1)) if m else None
    words = _norm(text[m.end():] if m else text)
    if "edema" in words:
        words = words[words.index("edema") + 1:]
    yes = {w for v in prompt["yes_words"] for w in _norm(v)}
    no = {w for v in prompt["no_words"] for w in _norm(v)}
    edema = next((("yes" if w in yes else "no") for w in words if (w in yes) != (w in no)), None)
    if grade is None:
        answer = "invalid"
    elif grade >= 2 or edema == "yes":
        answer = "yes"
    else:
        answer = "no" if edema == "no" else "invalid"
    return {"grade": grade, "edema": edema, "answer": answer}
