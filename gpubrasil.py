"""Chamadas à API da GPU Brasil (https://gpubrasil.com.br/br/api.html) para consultar, parar ou destruir a máquina.

    python gpubrasil.py info               # máquinas da conta (ou só GPUB_INSTANCIA): status, supports_pause, disk_persists_on_stop
    python gpubrasil.py parar              # stop (hiberna) SÓ se a máquina aceita parar e mantém o disco; senão não faz nada
    python gpubrasil.py destruir --sim     # DELETE: encerra a cobrança e APAGA o disco

Chave: GPUB_API_KEY ou o arquivo ~/.gpubrasil_key. Máquina: GPUB_INSTANCIA (name:<nome> ou instance_id; não o `id` curto da listagem).
Desligar pelo sistema (shutdown) NÃO para a cobrança: a plataforma continua contando a máquina como ligada.
Saída de `parar`: 0 = parada; 2 = não é seguro parar (disco seria apagado ou a máquina não hiberna); 1 = erro.
"""
import json
import os
import ssl
import sys
import urllib.request
from pathlib import Path

API = "https://gpubrasil.com.br/api/instances"


def _key():
    k = os.environ.get("GPUB_API_KEY") or (Path.home() / ".gpubrasil_key").read_text().strip()
    if not k:
        sys.exit("sem chave: defina GPUB_API_KEY ou crie ~/.gpubrasil_key")
    return k


def _ssl():
    """Certificados do certifi, se houver (o Python do python.org no Mac vem sem certificados instalados)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _call(method, path=""):
    req = urllib.request.Request(API + path, method=method, headers={"Authorization": f"Bearer {_key()}"})
    try:
        with urllib.request.urlopen(req, timeout=60, context=_ssl()) as r:
            body = r.read().decode()
            return r.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        return e.code, {"erro": e.read().decode()[:500]}


def _find(obj, key):
    """Procura `key` em qualquer nível da resposta (o formato exato do JSON não está documentado)."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            r = _find(v, key)
            if r is not None:
                return r
    return None


def _ref():
    ref = os.environ.get("GPUB_INSTANCIA")
    if not ref:
        sys.exit("defina GPUB_INSTANCIA (id ou name:<nome>); veja `python gpubrasil.py info`")
    return ref


def _instancia(ref):
    """Registro da máquina na listagem (só a listagem traz supports_pause e disk_persists_on_stop)."""
    code, d = _call("GET", "?limit=200")
    if code != 200:
        return None
    nome = ref[5:] if ref.startswith("name:") else None
    for it in d.get("data", []):
        if (nome and it.get("name") == nome) or str(it.get("instance_id")) == ref:
            return it
    return None


CAMPOS = ("name", "instance_id", "status", "tier", "supports_pause", "disk_persists_on_stop",
          "price_per_hour_brl", "daily_stopped_brl", "total_charged_brl")


def info():
    ref = os.environ.get("GPUB_INSTANCIA")
    if ref:
        it = _instancia(ref)
        print({k: it.get(k) for k in CAMPOS} if it else f"máquina {ref} não encontrada")
        code, d = _call("GET", f"/{ref}")
        print("estado da VM:", _find(d, "status"), "| conexão:", _find(d, "sshCommand"))
        return 0
    code, d = _call("GET", "?limit=200")
    for it in d.get("data", []):
        print({k: it.get(k) for k in CAMPOS})
    return 0 if code == 200 else 1


def parar():
    it = _instancia(_ref())
    if it is None:
        print("não consegui consultar a máquina", _ref())
        return 1
    pausa, disco = it.get("supports_pause"), it.get("disk_persists_on_stop")
    if pausa is not True or disco is not True:
        print(f"não vou parar: supports_pause={pausa}, disk_persists_on_stop={disco} (parar perderia os resultados)")
        return 2
    code, d = _call("POST", f"/{_ref()}/stop")
    print("stop:", code, d)
    return 0 if code < 300 else 1


def destruir():
    if "--sim" not in sys.argv:
        sys.exit("DELETE apaga a máquina e o disco: confirme com --sim")
    code, d = _call("DELETE", f"/{_ref()}")
    print("destruir:", code, d)
    return 0 if code < 300 else 1


if __name__ == "__main__":
    acoes = {"info": info, "parar": parar, "destruir": destruir}
    if len(sys.argv) < 2 or sys.argv[1] not in acoes:
        sys.exit(__doc__)
    r = acoes[sys.argv[1]]()
    sys.exit(r or 0)
