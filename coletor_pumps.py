#!/usr/bin/env python3
"""
Coletor de pumps - Binance Futuros USDT-M (perpétuos)

Encontra moedas que tiveram altas fortes e mede o que aconteceu antes e depois
do topo. Gera dois arquivos para análise:
  pumps_resumo.csv   -> uma linha por pump, com as métricas principais
  pumps_candles.csv  -> candles de 1h ao redor de cada pump (para testar regras)

Usa só dados públicos: não precisa de chave de API.

Instalação:  pip install requests pandas numpy
Uso:         python coletor_pumps.py
Opções:      python coletor_pumps.py --dias 365 --alta 0.4 --janela 24
             python coletor_pumps.py --limite 20      (teste rápido com 20 moedas)
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import requests

BASE = "https://fapi.binance.com"
HORA = 3_600_000
S = requests.Session()


def get(path, params=None, tentativas=5):
    for i in range(tentativas):
        try:
            r = S.get(BASE + path, params=params, timeout=20)
            if r.status_code in (418, 429):
                espera = int(r.headers.get("Retry-After", 60))
                print(f"  limite de requisições, aguardando {espera}s")
                time.sleep(espera)
                continue
            if r.status_code == 451:
                sys.exit("A Binance bloqueou o acesso a partir da sua localização (erro 451).")
            r.raise_for_status()
            time.sleep(0.25)
            return r.json()
        except requests.RequestException as e:
            print(f"  erro ({e}), tentando de novo...")
            time.sleep(2 * (i + 1))
    return None


def candles(sym, ini, fim):
    linhas, t = [], ini
    while t < fim:
        d = get("/fapi/v1/klines", {"symbol": sym, "interval": "1h",
                                    "startTime": t, "endTime": fim, "limit": 1000})
        if not d:
            break
        linhas += d
        if len(d) < 1000:
            break
        t = d[-1][0] + HORA
    if not linhas:
        return None
    df = pd.DataFrame(linhas).iloc[:, [0, 1, 2, 3, 4, 7, 10]]
    df.columns = ["t", "o", "h", "l", "c", "vol_usdt", "vol_compra_usdt"]
    df = df.astype(float)
    df["t"] = df["t"].astype("int64")
    return df.drop_duplicates("t").sort_values("t").reset_index(drop=True)


def funding(sym, ini, fim):
    out, t = [], ini
    while t < fim:
        d = get("/fapi/v1/fundingRate", {"symbol": sym, "startTime": t,
                                         "endTime": fim, "limit": 1000})
        if not d:
            break
        out += d
        if len(d) < 1000:
            break
        t = d[-1]["fundingTime"] + 1
    if not out:
        return pd.DataFrame({"t": pd.Series(dtype="int64"), "funding": pd.Series(dtype=float)})
    return pd.DataFrame({"t": [int(x["fundingTime"]) for x in out],
                         "funding": [float(x["fundingRate"]) for x in out]})


def rsi(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    p = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / p)


def detectar(sym, df, fr, onboard, btc, a):
    n = len(df)
    df["ret"] = df["c"] / df["c"].shift(a.janela) - 1
    df["rsi"] = rsi(df["c"])
    if len(fr):
        df = pd.merge_asof(df, fr, on="t", direction="backward")
    else:
        df["funding"] = np.nan
    eventos, janelas = [], []
    i = a.janela
    while i < n:
        if not df.at[i, "ret"] >= a.alta:
            i += 1
            continue
        # topo: maior máxima; estende enquanto houver máxima maior nas 24h seguintes
        pico = int(df["h"].iloc[i:min(n, i + 72)].idxmax())
        while True:
            prox = int(df["h"].iloc[pico:min(n, pico + 25)].idxmax())
            if prox == pico:
                break
            pico = prox
        base = int(df["l"].iloc[max(0, pico - 96):pico + 1].idxmin())
        P, B = df.at[pico, "h"], df.at[base, "l"]
        tp = int(df.at[pico, "t"])

        def depois(col, h, f):
            j = min(n, pico + 1 + h)
            return f(df[col].iloc[pico + 1:j]) if j > pico + 1 else np.nan

        def fech(h):
            return df.at[pico + h, "c"] if pico + h < n else np.nan

        vol_pump = df["vol_usdt"].iloc[base:pico + 1].mean()
        vol_antes = df["vol_usdt"].iloc[max(0, base - 168):base].mean()
        f24 = df["funding"].iloc[max(0, pico - 24):pico + 1]
        f_pos = fr[(fr.t > tp) & (fr.t <= tp + 72 * HORA)]["funding"].sum() if len(fr) else np.nan

        ev = {
            "id": f"{sym}_{tp}", "moeda": sym,
            "data_topo": pd.to_datetime(tp, unit="ms"),
            "dias_desde_listagem": round((tp - onboard) / 86_400_000, 1) if onboard else np.nan,
            "alta_pct": round((P / B - 1) * 100, 1),
            "duracao_alta_h": pico - base,
            "vol_ratio": round(vol_pump / vol_antes, 2) if vol_antes else np.nan,
            "vol_usdt_24h_topo": round(df["vol_usdt"].iloc[max(0, pico - 23):pico + 1].sum()),
            "pct_compra_agressiva": round(df["vol_compra_usdt"].iloc[base:pico + 1].sum()
                                          / df["vol_usdt"].iloc[base:pico + 1].sum() * 100, 1),
            "rsi_topo": round(df["rsi"].iloc[max(0, pico - 3):pico + 1].max(), 1),
            "funding_max_24h_pct": round(f24.max() * 100, 4),
            "funding_medio_24h_pct": round(f24.mean() * 100, 4),
            "funding_acum_3d_pos_pct": round(f_pos * 100, 4),
            "btc_ret_24h_pct": np.nan,
        }
        if btc is not None:
            bt = btc.set_index("t")["c"]
            if tp in bt.index and tp - 24 * HORA in bt.index:
                ev["btc_ret_24h_pct"] = round((bt[tp] / bt[tp - 24 * HORA] - 1) * 100, 2)
        for h, nome in [(24, "1d"), (72, "3d"), (168, "7d"), (336, "14d")]:
            ev[f"queda_max_{nome}_pct"] = round((1 - depois("l", h, np.min) / P) * 100, 1)
            ev[f"nova_max_{nome}_pct"] = round((depois("h", h, np.max) / P - 1) * 100, 1)
            ev[f"fech_{nome}_vs_topo_pct"] = round((fech(h) / P - 1) * 100, 1)
        # entrada simulada: 1º fechamento X% abaixo do topo, em até 72h
        ent = df.index[(df.index > pico) & (df.index <= pico + 72)
                       & (df["c"] <= P * (1 - a.confirmacao))]
        if len(ent):
            k = int(ent[0])
            E = df.at[k, "c"]
            fim7 = min(n, k + 169)
            ev["entrada_horas_apos_topo"] = k - pico
            ev["entrada_contra_max_7d_pct"] = round((df["h"].iloc[k + 1:fim7].max() / E - 1) * 100, 1)
            ev["entrada_favor_max_7d_pct"] = round((1 - df["l"].iloc[k + 1:fim7].min() / E) * 100, 1)
            ev["entrada_result_7d_pct"] = round((1 - df.at[k + 168, "c"] / E) * 100, 1) if k + 168 < n else np.nan
        ev["dados_completos_14d"] = pico + 336 < n
        eventos.append(ev)

        w = df.iloc[max(0, base - 168):min(n, pico + 337)].copy()
        w.insert(0, "id", ev["id"])
        w["h_rel_topo"] = w.index - pico
        janelas.append(w.drop(columns=["ret"]))
        i = pico + 72  # evita contar o mesmo pump duas vezes
    return eventos, janelas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", type=int, default=180, help="histórico em dias (padrão 180)")
    ap.add_argument("--alta", type=float, default=0.5, help="alta mínima, 0.5 = 50%%")
    ap.add_argument("--janela", type=int, default=24, help="janela da alta em horas")
    ap.add_argument("--confirmacao", type=float, default=0.10,
                    help="queda a partir do topo para a entrada simulada (0.10 = 10%%)")
    ap.add_argument("--limite", type=int, default=0, help="analisar só N moedas (teste)")
    a = ap.parse_args()

    fim = int(time.time() * 1000)
    ini = fim - a.dias * 86_400_000
    info = get("/fapi/v1/exchangeInfo")
    if not info:
        sys.exit("Não consegui acessar a API da Binance.")
    syms = [(s["symbol"], s.get("onboardDate")) for s in info["symbols"]
            if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"]
    if a.limite:
        syms = syms[:a.limite]
    print(f"{len(syms)} moedas | {a.dias} dias | alta mínima {a.alta:.0%} em {a.janela}h")

    btc = candles("BTCUSDT", ini, fim)
    todos, todas_janelas = [], []
    for n_, (sym, onboard) in enumerate(syms, 1):
        df = candles(sym, ini, fim)
        if df is None or len(df) < 200:
            continue
        if (df["h"] / df["l"].rolling(a.janela).min()).max() < 1 + a.alta:
            print(f"[{n_}/{len(syms)}] {sym}: sem pumps")
            continue
        fr = funding(sym, ini, fim)
        ev, jan = detectar(sym, df, fr, onboard, btc, a)
        todos += ev
        todas_janelas += jan
        print(f"[{n_}/{len(syms)}] {sym}: {len(ev)} pump(s)")

    if not todos:
        print("Nenhum pump encontrado com esses parâmetros.")
        return
    pd.DataFrame(todos).sort_values("data_topo").to_csv("pumps_resumo.csv", index=False)
    pd.concat(todas_janelas).to_csv("pumps_candles.csv", index=False)
    print(f"\nPronto: {len(todos)} pumps -> pumps_resumo.csv e pumps_candles.csv")


if __name__ == "__main__":
    main()
