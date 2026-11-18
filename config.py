"""Caminhos e parâmetros do projeto. Sobrescreva por variáveis de ambiente se necessário.

Os padrões seguem os outros projetos do grupo (bilateral_leakage, retinal_age_vrs):
rótulos em ~/Downloads, imagens do BRSET em ~/Desktop/fundus_photos, imagens do mBRSET em ~/Downloads/mBRSET.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOME = Path.home()

# ----------------------------------------------------------------------------- dados (PhysioNet)
LABELS_DIR = Path(os.environ.get("LABELS_DIR", HOME / "Downloads"))
BRSET_LABELS = Path(os.environ.get("BRSET_LABELS", LABELS_DIR / "labels_brset.csv"))
MBRSET_LABELS = Path(os.environ.get("MBRSET_LABELS", LABELS_DIR / "labels_mbrset.csv"))
BRSET_IMAGES = Path(os.environ.get("BRSET_IMAGES", HOME / "Desktop" / "fundus_photos"))
MBRSET_IMAGES = Path(os.environ.get("MBRSET_IMAGES", LABELS_DIR / "mBRSET"))

# ----------------------------------------------------------------------------- embeddings (linha de base)
# 1º) cache .npy já gerado no projeto bilateral_leakage ({base}_{modelo}.npy + {base}_{modelo}_ids.npy)
EMB_NPY_DIR = Path(os.environ.get("EMB_NPY_DIR", ROOT.parent / "bilateral_leakage" / "data" / "emb"))
# 2º) CSVs Embeddings_{base}_{modelo}.csv (PhysioNet "embedding-based-representations" e retinal_age_vrs)
EMB_CSV_DIRS = [
    Path(os.environ.get("DINO_DIR", LABELS_DIR / "embedding-based-representations-for-brset-and-mbrset-1.0.0")),
    Path(os.environ.get("RETFOUND_DIR", ROOT.parent / "retinal_age_vrs" / "embeddings")),
    LABELS_DIR / "Embeddings",
]
EMB_MODELS = ["retfound_gap", "retfound_cls", "dinov3_vits16", "dinov3_convnext_tiny"]

# ----------------------------------------------------------------------------- saídas (não versionar data/ nem preds/)
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PREDS = RESULTS / "preds"
TABLES = RESULTS / "tables"
FIGURES = ROOT / "figures"
MODELS_YAML = ROOT / "settings" / "models.yaml"
PROMPTS_YAML = ROOT / "settings" / "prompts.yaml"

# ----------------------------------------------------------------------------- coorte e amostra
SEED = 42
# Diabetes no BRSET: a coluna `diabetes` codifica dado ausente como "No"; por isso o padrão usa o texto
# de `comorbidities` (mesma regra do bilateral_leakage). "comorbidities_or_column" também aceita diabetes == yes.
DIABETES_RULE = os.environ.get("DIABETES_RULE", "comorbidities")
# Qualidade BRSET: 2 = insatisfatório em focus e Illuminaton (grafia do CSV); image_field não exclui
BRSET_QUALITY_COLS = ["focus", "Illuminaton"]
ONE_PER_PATIENT = False      # False = todas as imagens elegíveis (os dois olhos; análise agrupada por paciente)
N_PILOT = 50                 # por base, só para ajustar prompts e o parser; fora de todas as análises
SAMPLE_MODE = "all"          # "all" = todos os pacientes elegíveis; "stratified" = N_REF + N_NONREF
N_REF, N_NONREF = 300, 700
N_ROBUST = 300               # subamostra para os prompts alternativos
N_REPRO = 100                # subamostra para a segunda execução (reprodutibilidade)

# ----------------------------------------------------------------------------- inferência
MAX_NEW_TOKENS = 16         # "2; Macular edema: No" + folga
BOOTSTRAP = 2000
ALPHA = 0.05
