"""Monta envio_nuvem.tar para a máquina com GPU: código, coorte e só as imagens da coorte.

    python preparar_nuvem.py

Na máquina: tar -xf envio_nuvem.tar && cd vlm_dr_brset (passos no README, seção "GPU na nuvem").
Não inclui rótulos completos, embeddings nem results/: só o necessário para os notebooks 03 e 05.
"""
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CODIGO = ["config.py", "vlmdr", "settings", "notebooks/03_inferencia_gpu.ipynb", "notebooks/05_extra_gpu.ipynb", "rodar_nuvem.sh", "gpubrasil.py",
          "requirements-gpu.txt", "README.md"]


def main():
    out = ROOT / "envio_nuvem.tar"
    sem_cache = lambda ti: None if "__pycache__" in ti.name else ti
    n = 0
    with tarfile.open(out, "w") as tar:      # sem compressão: as imagens já são JPEG
        for c in CODIGO:
            tar.add(ROOT / c, arcname=f"vlm_dr_brset/{c}", filter=sem_cache)
        for ds in ("brset", "mbrset"):
            csv = ROOT / "data" / f"cohort_{ds}.csv"
            tar.add(csv, arcname=f"vlm_dr_brset/data/{csv.name}")
            for p in pd.read_csv(csv)["image_path"]:
                tar.add(p, arcname=f"vlm_dr_brset/imagens/{ds}/{Path(p).name}")
                n += 1
    print(f"{out}: {n} imagens, {out.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
