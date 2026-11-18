#!/usr/bin/env bash
# Roda o notebook 03 na GPU sem depender do navegador; no fim (ou se der erro ou travar) empacota os resultados,
# espera o vigiar_nuvem.sh (no Mac) baixá-los e destruir a máquina; se ninguém baixar em ESPERA_MIN minutos,
# para a máquina pela API da GPU Brasil (reserva: a máquina pode voltar da hibernação sem IP público).
#
#   export HF_TOKEN=hf_...                       # conta que aceitou as licenças do MedGemma e do Gemma 3
#   export GPUB_API_KEY=gpub_live_...            # chave da API da GPU Brasil
#   export GPUB_INSTANCIA=name:<nome>            # ou o id da máquina (veja: python3 gpubrasil.py info)
#   nohup bash rodar_nuvem.sh > /dev/null 2>&1 &
#   tail -f results/execucao_nuvem.log           # acompanhar
#
# Variáveis opcionais:
#   NOTEBOOK=notebooks/05_extra_gpu.ipynb   notebook a executar (padrão: notebooks/03_inferencia_gpu.ipynb)
#   MAX_HORAS=7    interrompe no limite, mesmo se não terminou ou travou (o que já foi gravado fica; rodar de novo retoma)
#   MODELOS=medgemma_27b   modelos (separados por vírgula)
#   LIMITE=5       ensaio rápido: no máximo 5 imagens por etapa
#   ESPERA_MIN=30  minutos esperando o vigia baixar os resultados antes de parar a máquina
#   ENCERRAR=0     não para a máquina no fim
#   VENV=~/vlm_env ambiente Python a ativar
#
# Desligar pelo sistema (shutdown) NÃO para a cobrança na GPU Brasil: por isso este script usa a API.
set -u
cd "$(dirname "$0")"

VENV="${VENV:-$HOME/vlm_env}"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export BRSET_IMAGES="${BRSET_IMAGES:-$PWD/imagens/brset}"
export MBRSET_IMAGES="${MBRSET_IMAGES:-$PWD/imagens/mbrset}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/hf_cache}"   # pesos (só o cache: HF_HOME mudaria também onde fica o token)
ENCERRAR="${ENCERRAR:-1}"

mkdir -p results
LOG=results/execucao_nuvem.log
echo "início: $(date -u)" > "$LOG"
nvidia-smi >> "$LOG" 2>&1

# ENCERRAR=0 só para o notebook: quem para a máquina é este script, depois de empacotar tudo
NOTEBOOK="${NOTEBOOK:-notebooks/03_inferencia_gpu.ipynb}"
EXECUTADO="$(basename "$NOTEBOOK" .ipynb)_executado.ipynb"
echo "notebook: $NOTEBOOK" >> "$LOG"
ENCERRAR=0 timeout "${MAX_HORAS:-7}h" jupyter nbconvert --to notebook --execute "$NOTEBOOK" \
    --output "$EXECUTADO" --ExecutePreprocessor.timeout=-1 >> "$LOG" 2>&1
echo "fim: código de saída $? em $(date -u)" >> "$LOG"

rm -f BAIXADO
tar -czf resultados_nuvem.tar.gz results "notebooks/$EXECUTADO" >> "$LOG" 2>&1
sync
echo "resultados: $PWD/resultados_nuvem.tar.gz" >> "$LOG"

# o vigia cria o arquivo BAIXADO depois de baixar e conferir os resultados (e então destrói a máquina)
for _ in $(seq "${ESPERA_MIN:-30}"); do
    [ -f BAIXADO ] && { echo "resultados baixados pelo vigia: $(date -u)" >> "$LOG"; exit 0; }
    sleep 60
done
echo "ninguém baixou os resultados em ${ESPERA_MIN:-30} min" >> "$LOG"

if [ "$ENCERRAR" = "1" ]; then
    python3 gpubrasil.py parar >> "$LOG" 2>&1
    echo "parar: código $?" >> "$LOG"
fi
