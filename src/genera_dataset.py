"""
genera_dataset.py
-----------------
Simulazione del dataset "Posizioni difensive del libero" (progetto Principi e Modelli della Percezione).

SISTEMA DI COORDINATE (metri, campo intero 9 x 18 m)
  x : 0..9, da sinistra a destra, visto dalla squadra del libero che guarda la rete
  y : 0..18; y = 0 fondo campo del libero, y = 9 rete, y = 18 fondo campo avversario
  Campo del libero (clustering): y in [0, 9]  ->  area 9 x 9 m
  Campo dell'attaccante (contesto): y in [9, 18]

POSIZIONE DI PARTENZA DEL LIBERO
  Il libero difende di norma in POSTO 5 (seconda linea, a sinistra, vista la rete):
  la sua base e' (X_BASE, Y_BASE) e ogni spostamento e' simulato a partire da li'.

CODIFICA DEL MURO
  0 = muro chiude la parallela  -> il libero si accentra e difende la diagonale
  1 = muro chiude la diagonale  -> il libero si sposta lungo la linea laterale
  2 = muro a un uomo / disorganizzato -> il libero assume una posizione di emergenza

NOTE
  - tipo_colpo == "emergenza" marca le compensazioni simulate: serve come verita' di
    riferimento per valutare il DBSCAN, NON come feature del clustering.
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N_AZIONI = 600
P_COMPENSAZIONE = 0.07          # quota di azioni "straordinarie" (outlier)

ZONE = [1, 2, 3, 4, 6]
P_ZONA = [0.15, 0.20, 0.15, 0.30, 0.20]
ZONE_PRIMA_LINEA = {2, 3, 4}
# x medio dell'attaccante, nel sistema del libero: la zona 4 avversaria
# (sua sinistra) cade sulla destra del libero, la 2 e la 1 sulla sinistra.
X_ZONA = {1: 1.8, 2: 1.2, 3: 4.5, 4: 7.8, 6: 4.5}

P_MURO = [0.42, 0.40, 0.18]

# --- posto 5: posizione di partenza del libero ---
X_BASE = 1.8                    # a sinistra, vicino alla linea laterale
DIST_RETE_BASE = 5.4            # distanza dalla rete in posizione di partenza (y = 9 - dist.)
CENTRO = 4.5                    # centro del campo (riferimento per l'effetto dell'attaccante)


def simula_azione(rng: np.random.Generator) -> dict:
    zona = int(rng.choice(ZONE, p=P_ZONA))
    muro = int(rng.choice([0, 1, 2], p=P_MURO))

    # --- attaccante (momento del distacco) ---
    x_att = float(np.clip(rng.normal(X_ZONA[zona], 0.6), 0.3, 8.7))
    if zona in ZONE_PRIMA_LINEA:
        y_att = 9 + float(np.clip(rng.normal(1.4, 0.5), 0.3, 2.8))     # vicino alla rete
    else:
        y_att = 12 + float(np.clip(rng.normal(0.9, 0.4), 0.2, 2.2))    # dietro la linea dei 3 m

    # --- libero: parte da posto 5 ---
    if rng.random() < P_COMPENSAZIONE:
        # compensazione straordinaria: posizione fuori dal pattern standard
        tipo_colpo = "emergenza"
        r = rng.random()
        if r < 0.40:        # buco nel muro: copertura di un pallonetto / tocco vicino alla rete
            x_dif, y_dif = rng.normal(3.5, 1.8), rng.uniform(6.2, 8.2)
        elif r < 0.75:      # sbilanciamento di un compagno: il libero copre la zona destra (posto 1)
            x_dif, y_dif = rng.uniform(6.4, 8.4), rng.uniform(1.0, 5.5)
        else:               # salvataggio profondo, a ridosso del fondo
            x_dif, y_dif = rng.uniform(1.0, 8.0), rng.uniform(0.3, 0.9)
    else:
        tipo_colpo = "forte" if rng.random() < 0.6 else "piazzato"
        dx_att = x_att - CENTRO                      # da che lato arriva l'attacco
        if muro == 0:       # parallela chiusa -> il libero si accentra per la diagonale
            # l'attacco da destra manda la diagonale verso posto 5 (si accentra meno),
            # l'attacco da sinistra la manda verso posto 1 (si accentra di piu')
            x_dif = X_BASE + 1.6 - 0.15 * dx_att + rng.normal(0, 0.45)
            distanza_rete = rng.normal(5.8, 0.5)
        elif muro == 1:     # diagonale chiusa -> il libero va verso la linea laterale
            x_dif = X_BASE - 0.6 + 0.12 * dx_att + rng.normal(0, 0.35)
            distanza_rete = rng.normal(5.0, 0.5)
        else:               # muro disorganizzato -> posizione di emergenza, piu' profonda e dispersa
            x_dif = X_BASE + 1.0 + 0.2 * dx_att + rng.normal(0, 0.8)
            distanza_rete = rng.normal(6.6, 0.7)
        # il colpo forte "tira indietro" il libero, il piazzato lo avvicina alla rete
        distanza_rete += 0.4 if tipo_colpo == "forte" else -0.5
        y_dif = 9 - distanza_rete

    return {
        "zona_attacco": zona,
        "muro": muro,
        "tipo_colpo": tipo_colpo,
        "x_att": x_att,
        "y_att": y_att,
        "x_dif": float(np.clip(x_dif, 0.4, 8.6)),
        "y_dif": float(np.clip(y_dif, 0.4, 8.6)),
    }


def genera(n: int = N_AZIONI, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame([simula_azione(rng) for _ in range(n)])
    df.insert(0, "id_azione", np.arange(1, n + 1))
    return df.round(2)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = genera()
    df.to_csv("data/dataset_volley.csv", index=False)
    print(df.head())
    print(f"\nRighe: {len(df)}")
