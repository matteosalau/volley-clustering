"""
esplora_dataset.py  (Fase 2)
Controlli di qualita', statistiche descrittive e prima mappa del campo.
Eseguire dalla cartella principale del progetto:  python src/esplora_dataset.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

os.makedirs("figures", exist_ok=True)
df = pd.read_csv("data/dataset_volley.csv")

MURO_LABEL = {0: "Muro 0 - parallela chiusa", 1: "Muro 1 - diagonale chiusa", 2: "Muro 2 - disorganizzato"}
MURO_COL = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}

# ------------------------------------------------------------------ 1. controlli di qualita'
print("=== CONTROLLI DI QUALITA' ===")
print("Valori mancanti per colonna:", int(df.isna().sum().sum()))
print("Righe duplicate:", int(df.drop(columns="id_azione").duplicated().sum()))
fuori = (
    (~df.x_dif.between(0, 9)) | (~df.y_dif.between(0, 9)) |
    (~df.x_att.between(0, 9)) | (~df.y_att.between(9, 18))
)
print("Coordinate fuori campo:", int(fuori.sum()))
print("Valori ammessi -> zona:", sorted(df.zona_attacco.unique()),
      "| muro:", sorted(df.muro.unique()), "| tipo_colpo:", sorted(df.tipo_colpo.unique()))

# ------------------------------------------------------------------ 2. statistiche descrittive
print("\n=== STATISTICHE DESCRITTIVE (libero) ===")
print(df[["x_dif", "y_dif"]].describe().round(2).to_string())
for col in ["muro", "zona_attacco", "tipo_colpo"]:
    print(f"\n--- per {col} ---")
    t = df.groupby(col).agg(n=("id_azione", "count"),
                            x_media=("x_dif", "mean"), x_dev=("x_dif", "std"),
                            y_media=("y_dif", "mean"), y_dev=("y_dif", "std")).round(2)
    print(t.to_string())
print("\n--- muro x zona: x_dif medio del libero ---")
print(df[df.tipo_colpo != "emergenza"].pivot_table(index="zona_attacco", columns="muro",
                                                    values="x_dif", aggfunc="mean").round(2).to_string())

# ------------------------------------------------------------------ 3. mappa del campo
fig, ax = plt.subplots(figsize=(6, 10))
ax.add_patch(Rectangle((0, 0), 9, 18, fill=False, lw=2, color="black"))
ax.plot([0, 9], [9, 9], color="black", lw=4)                      # rete
for y in (6, 12):                                                 # linee dei 3 m
    ax.plot([0, 9], [y, y], color="gray", lw=1, ls="--")
ax.text(9.15, 9, "rete", va="center", fontsize=9)

ax.scatter(df.x_att, df.y_att, marker="^", s=14, color="lightgray", edgecolor="gray",
           lw=0.3, label="Attaccante (contesto)")
for m in (0, 1, 2):
    s = df[(df.muro == m) & (df.tipo_colpo != "emergenza")]
    ax.scatter(s.x_dif, s.y_dif, s=18, alpha=0.7, color=MURO_COL[m], label=MURO_LABEL[m])
e = df[df.tipo_colpo == "emergenza"]
ax.scatter(e.x_dif, e.y_dif, marker="x", s=40, color="red", label="Emergenza (compensazione)")

ax.set_xlim(-0.5, 10); ax.set_ylim(-0.5, 18.5); ax.set_aspect("equal")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Posizioni del libero e dell'attaccante - campo 9x18 m")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig("figures/01_mappa_campo.png", dpi=150)

# ------------------------------------------------------------------ 4. distribuzioni
fig2, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for m in (0, 1, 2):
    s = df[(df.muro == m) & (df.tipo_colpo != "emergenza")]
    axs[0].hist(s.x_dif, bins=24, alpha=0.55, color=MURO_COL[m], label=MURO_LABEL[m])
    axs[1].hist(s.y_dif, bins=24, alpha=0.55, color=MURO_COL[m])
axs[0].set_title("Distribuzione di x_dif per muro"); axs[0].set_xlabel("x_dif (m)")
axs[1].set_title("Distribuzione di y_dif per muro"); axs[1].set_xlabel("y_dif (m)")
axs[0].legend(fontsize=7)
fig2.tight_layout()
fig2.savefig("figures/02_distribuzioni.png", dpi=150)
print("\nFigure salvate in figures/")
