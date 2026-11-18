#!/usr/bin/env bash
# Roda no Mac enquanto a nuvem trabalha: a cada 5 min confere a máquina pelo SSH; quando a execução termina,
# baixa resultados_nuvem.tar.gz, confere o arquivo, avisa a máquina (arquivo BAIXADO) e destrói a máquina pela API
# (encerra a cobrança de vez). Se o vigia não baixar em 30 min, a própria máquina se para pela API.
#
#   export GPUB_INSTANCIA=name:<nome>      # chave da API em ~/.gpubrasil_key (ou GPUB_API_KEY)
#   export NUVEM=ubuntu@<ip>               # e, se precisar, CHAVE_SSH=~/.ssh/id_ed25519
#   caffeinate -i nohup bash vigiar_nuvem.sh > vigia.log 2>&1 &
#
# DESTRUIR=0 só baixa os resultados (não destrói). O Mac precisa ficar ligado e com internet.
set -u
cd "$(dirname "$0")"
: "${NUVEM:?defina NUVEM=ubuntu@<ip>}"
: "${GPUB_INSTANCIA:?defina GPUB_INSTANCIA}"
SSH=(ssh -i "${CHAVE_SSH:-$HOME/.ssh/id_ed25519}" -o BatchMode=yes -o ConnectTimeout=20)
SCP=(scp -i "${CHAVE_SSH:-$HOME/.ssh/id_ed25519}" -o BatchMode=yes -o ConnectTimeout=20)
DIR="${DIR_NUVEM:-vlm_dr_brset}"
DESTINO="results/nuvem_$(date +%Y%m%d)"

while true; do
    if "${SSH[@]}" "$NUVEM" "grep -q '^resultados:' $DIR/results/execucao_nuvem.log" 2>/dev/null; then
        echo "$(date) execução terminou; baixando resultados"
        mkdir -p "$DESTINO"
        if "${SCP[@]}" "$NUVEM:$DIR/resultados_nuvem.tar.gz" "$DESTINO/" && tar -tzf "$DESTINO/resultados_nuvem.tar.gz" > /dev/null; then
            echo "$(date) resultados em $DESTINO/resultados_nuvem.tar.gz"
            "${SSH[@]}" "$NUVEM" "touch $DIR/BAIXADO"
            if [ "${DESTRUIR:-1}" = "1" ]; then
                python3 gpubrasil.py destruir --sim
            fi
            exit 0
        fi
        echo "$(date) falha ao baixar ou conferir; tento de novo em 5 min"
    elif ! "${SSH[@]}" "$NUVEM" true 2>/dev/null; then
        echo "$(date) máquina inacessível (parada ou sem rede):"
        python3 gpubrasil.py info
    else
        echo "$(date) rodando: $("${SSH[@]}" "$NUVEM" "tail -c 300 \$(ls -t $DIR/results/execucao_gpu*.csv 2>/dev/null | head -1) 2>/dev/null | tail -1")"
    fi
    sleep 300
done
