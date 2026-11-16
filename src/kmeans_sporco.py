"""
kmeans_sporco.py  (Fase 3a)
K-Means "sporco": applicato a TUTTI i punti, compensazioni comprese, un esperimento per tipo di muro.
Serve da confronto con la versione "pulita" (DBSCAN per togliere il rumore, poi K-Means).

Eseguire dalla cartella principale del progetto:  python src/kmeans_sporco.py
Output: figures/03_scelta_k.png, figures/04_kmeans_sporco.png, data/centroidi_kmeans_sporco.csv

NOTA: tipo_colpo == "emergenza" NON entra nel clustering; e' usato solo a valle,
per misurare quanto le compensazioni deformano i centroidi.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEED = 42
K_SCELTO = 3                       # tre zone difensive (sinistra, centro, destra) come da traccia
K_RANGE = range(2, 9)
FEATURES = ["x_dif", "y_dif"]
X_BASE, Y_BASE = 1.8, 3.6          # posizione di partenza del libero (posto 5)

MURO_LABEL = {0: "Muro 0 - parallela chiusa", 1: "Muro 1 - diagonale chiusa", 2: "Muro 2 - disorganizzato"}
MURO_COL = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}

os.makedirs("figures", exist_ok=True)
df = pd.read_csv("data/dataset_volley.csv")
df["emergenza"] = df.tipo_colpo == "emergenza"          # solo valutazione a valle


def campo(ax):
    """Meta' campo difensiva 9x9 m, rete in alto (y = 9)."""
    ax.add_patch(Rectangle((0, 0), 9, 9, fill=False, lw=2, color="black"))
    ax.plot([0, 9], [9, 9], color="black", lw=4)
    ax.plot([0, 9], [6, 6], color="gray", lw=1, ls="--")          # linea dei 3 m
    ax.set_xlim(-0.4, 9.4); ax.set_ylim(-0.4, 9.6); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")


# ------------------------------------------------------------------ 1. scelta di k
print("=== SCELTA DI k (inerzia e silhouette) ===")
risultati = []
for m, g in df.groupby("muro"):
    X = g[FEATURES].values
    for k in K_RANGE:
        km = KMeans(k, n_init=10, random_state=SEED).fit(X)
        risultati.append({"muro": m, "k": k, "inerzia": km.inertia_,
                          "silhouette": silhouette_score(X, km.labels_)})
ris = pd.DataFrame(risultati)
print(ris.pivot(index="k", columns="muro", values="silhouette").round(3).to_string())
print("\nk con silhouette massima per muro:", ris.loc[ris.groupby("muro").silhouette.idxmax()][["muro", "k"]].values.tolist())

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for m in (0, 1, 2):
    r = ris[ris.muro == m]
    axs[0].plot(r.k, r.inerzia, marker="o", color=MURO_COL[m], label=MURO_LABEL[m])
    axs[1].plot(r.k, r.silhouette, marker="o", color=MURO_COL[m])
axs[0].set_title("Metodo del gomito (inerzia)"); axs[0].set_xlabel("k"); axs[0].set_ylabel("inerzia")
axs[1].set_title("Silhouette media"); axs[1].set_xlabel("k"); axs[1].set_ylabel("silhouette")
for a in axs:
    a.axvline(K_SCELTO, color="gray", ls=":", lw=1)
axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/03_scelta_k.png", dpi=150)

# ------------------------------------------------------------------ 2. K-Means sporco (k = 3)
print(f"\n=== K-MEANS SPORCO, k = {K_SCELTO}, tutti i punti ===")
righe = []
fig, axs = plt.subplots(2, 2, figsize=(11, 11))
for ax, m in zip(axs.ravel()[:3], (0, 1, 2)):
    g = df[df.muro == m].copy()
    km = KMeans(K_SCELTO, n_init=10, random_state=SEED).fit(g[FEATURES].values)
    # rinomina i cluster da sinistra a destra, per leggerli sempre nello stesso ordine
    ordine = np.argsort(km.cluster_centers_[:, 0])
    rinomina = {old: new for new, old in enumerate(ordine)}
    g["cluster"] = [rinomina[c] for c in km.labels_]
    centri = km.cluster_centers_[ordine]

    campo(ax)
    for c in range(K_SCELTO):
        s = g[g.cluster == c]
        ax.scatter(s.x_dif, s.y_dif, s=18, alpha=0.65, label=f"cluster {c} (n={len(s)})")
    e = g[g.emergenza]
    ax.scatter(e.x_dif, e.y_dif, marker="x", s=36, color="red", label=f"compensazioni (n={len(e)})")
    ax.scatter(centri[:, 0], centri[:, 1], marker="*", s=380, color="black", edgecolor="white",
               lw=1.2, zorder=5, label="centroidi")
    ax.set_title(MURO_LABEL[m]); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)

    print(f"\n{MURO_LABEL[m]}  (n = {len(g)})")
    print("  cluster    n      x     y   compensazioni   %comp")
    for c in range(K_SCELTO):
        s = g[g.cluster == c]
        righe.append({"muro": m, "cluster": c, "x": round(centri[c, 0], 2), "y": round(centri[c, 1], 2),
                      "n": len(s), "n_compensazioni": int(s.emergenza.sum())})
        print(f"  {c:>7} {len(s):>4} {centri[c,0]:>6.2f} {centri[c,1]:>5.2f} {int(s.emergenza.sum()):>12} "
              f"{100*s.emergenza.mean():>8.1f}")

# pannello 4: confronto dei centroidi tra i tre esperimenti, partendo dalla base in posto 5
ax = axs.ravel()[3]
campo(ax)
cen = pd.DataFrame(righe)
ax.scatter([X_BASE], [Y_BASE], marker="s", s=140, color="black", label="partenza (posto 5)", zorder=5)
for m in (0, 1, 2):
    c = cen[cen.muro == m]
    ax.scatter(c.x, c.y, marker="*", s=300, color=MURO_COL[m], edgecolor="black", lw=0.8,
               label=MURO_LABEL[m], zorder=4)
ax.set_title("Centroidi a confronto (K-Means sporco)"); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)

fig.suptitle("K-Means sporco: punti = azioni, stella = centroide, x rosse = compensazioni", y=0.995)
fig.tight_layout(); fig.savefig("figures/04_kmeans_sporco.png", dpi=150)

pd.DataFrame(righe).to_csv("data/centroidi_kmeans_sporco.csv", index=False)
print("\nSalvati: figures/03_scelta_k.png, figures/04_kmeans_sporco.png, data/centroidi_kmeans_sporco.csv")
