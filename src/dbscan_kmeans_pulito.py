"""
dbscan_kmeans_pulito.py  (Fase 3b)
1) DBSCAN su tutti i punti: il rumore (etichetta -1) sono le compensazioni candidate.
2) K-Means "pulito": un esperimento per tipo di muro sui soli punti NON marcati come rumore.
3) Confronto dei centroidi pulito vs sporco (data/centroidi_kmeans_sporco.csv, da kmeans_sporco.py).

Eseguire dalla cartella principale del progetto, dopo kmeans_sporco.py:
    python src/dbscan_kmeans_pulito.py
Output: figures/05_dbscan.png, figures/06_kmeans_pulito.png,
        data/dataset_dbscan.csv, data/centroidi_kmeans_pulito.csv

NOTA: tipo_colpo == "emergenza" (verita' di riferimento della simulazione) NON entra ne' nel
DBSCAN ne' nel K-Means: serve solo a valutare a posteriori quanto il DBSCAN azzecca.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

SEED = 42
FEATURES = ["x_dif", "y_dif"]
MIN_SAMPLES = 8
K_SCELTO = 3
K_RANGE = range(2, 9)
X_BASE, Y_BASE = 1.8, 3.6

MURO_LABEL = {0: "Muro 0 - parallela chiusa", 1: "Muro 1 - diagonale chiusa", 2: "Muro 2 - disorganizzato"}
MURO_COL = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}

os.makedirs("figures", exist_ok=True)
df = pd.read_csv("data/dataset_volley.csv")
df["emergenza"] = df.tipo_colpo == "emergenza"          # solo valutazione a valle
X = df[FEATURES].values


def campo(ax):
    ax.add_patch(Rectangle((0, 0), 9, 9, fill=False, lw=2, color="black"))
    ax.plot([0, 9], [9, 9], color="black", lw=4)
    ax.plot([0, 9], [6, 6], color="gray", lw=1, ls="--")
    ax.set_xlim(-0.4, 9.4); ax.set_ylim(-0.4, 9.6); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")


# ------------------------------------------------------------------ 1. scelta di eps (k-distance graph)
nn = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X)
kdist = np.sort(nn.kneighbors(X)[0][:, -1])
# ginocchio = punto piu' lontano dalla retta che unisce primo e ultimo punto della curva
p = np.column_stack([np.arange(len(kdist)), kdist])
d = p - p[0]; v = (p[-1] - p[0]) / np.linalg.norm(p[-1] - p[0])
dist = np.abs(d[:, 0] * v[1] - d[:, 1] * v[0])
eps_knee = float(kdist[np.argmax(dist)])
print(f"=== SCELTA DI eps (k-distance con k = {MIN_SAMPLES}) ===")
print(f"ginocchio della curva: eps ~ {eps_knee:.2f} m")

print("\n--- griglia di controllo (verita' simulata: tipo_colpo == emergenza) ---")
print(" eps  min_s  cluster  rumore  compens.catturate  falsi_positivi")
for eps in (0.4, 0.5, 0.6, 0.7):
    for ms in (6, 8, 10):
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
        rum = lab == -1
        print(f"{eps:>4} {ms:>6} {len(set(lab)) - (1 if rum.any() else 0):>8} {rum.sum():>7} "
              f"{(rum & df.emergenza.values).sum():>10}/{df.emergenza.sum()} {(rum & ~df.emergenza.values).sum():>14}")

# ------------------------------------------------------------------ 2. DBSCAN finale
EPS = round(eps_knee, 2)
df["dbscan"] = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(X)
df["rumore"] = df.dbscan == -1
tp = int((df.rumore & df.emergenza).sum()); fp = int((df.rumore & ~df.emergenza).sum())
fn = int((~df.rumore & df.emergenza).sum())
print(f"\n=== DBSCAN finale: eps = {EPS}, min_samples = {MIN_SAMPLES} ===")
print(f"cluster trovati: {df.loc[~df.rumore, 'dbscan'].nunique()} | punti di rumore: {df.rumore.sum()} su {len(df)}")
print(f"compensazioni catturate: {tp}/{df.emergenza.sum()} | falsi positivi: {fp} | mancate: {fn}")
print("rumore per muro:", df.groupby("muro").rumore.sum().to_dict(), "| falsi positivi per muro:",
      df[df.rumore & ~df.emergenza].groupby("muro").size().to_dict())

fig, axs = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1, 1.1]})
axs[0].plot(kdist, color="tab:blue")
axs[0].axhline(EPS, color="red", ls="--", lw=1, label=f"eps scelto = {EPS}")
axs[0].set_xlabel("punti ordinati per distanza"); axs[0].set_ylabel(f"distanza dall'{MIN_SAMPLES}o vicino (m)")
axs[0].set_title("k-distance graph"); axs[0].legend(fontsize=8)
campo(axs[1])
for m in (0, 1, 2):
    s = df[(df.muro == m) & ~df.rumore]
    axs[1].scatter(s.x_dif, s.y_dif, s=16, alpha=0.7, color=MURO_COL[m], label=f"{MURO_LABEL[m]} (n={len(s)})")
n = df[df.rumore]
axs[1].scatter(n.x_dif, n.y_dif, marker="x", s=42, color="black", label=f"rumore DBSCAN (n={len(n)})")
e = df[df.emergenza]
axs[1].scatter(e.x_dif, e.y_dif, s=110, facecolor="none", edgecolor="red", lw=1.2,
               label=f"compensazioni simulate (n={len(e)})")
axs[1].set_title("DBSCAN: rumore vs compensazioni simulate")
axs[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
fig.tight_layout(); fig.savefig("figures/05_dbscan.png", dpi=150)

# ------------------------------------------------------------------ 3. K-Means pulito
pul = df[~df.rumore].copy()
print("\n=== K-MEANS PULITO: silhouette per k ===")
rows = []
for m, g in pul.groupby("muro"):
    for k in K_RANGE:
        km = KMeans(k, n_init=10, random_state=SEED).fit(g[FEATURES].values)
        rows.append({"muro": m, "k": k, "silhouette": silhouette_score(g[FEATURES].values, km.labels_)})
print(pd.DataFrame(rows).pivot(index="k", columns="muro", values="silhouette").round(3).to_string())

sporco = pd.read_csv("data/centroidi_kmeans_sporco.csv")
print(f"\n=== K-MEANS PULITO, k = {K_SCELTO}, punti senza rumore ===")
righe = []
fig, axs = plt.subplots(2, 2, figsize=(11, 11))
for ax, m in zip(axs.ravel()[:3], (0, 1, 2)):
    g = pul[pul.muro == m].copy()
    km = KMeans(K_SCELTO, n_init=10, random_state=SEED).fit(g[FEATURES].values)
    ordine = np.argsort(km.cluster_centers_[:, 0])
    g["cluster"] = [{o: i for i, o in enumerate(ordine)}[c] for c in km.labels_]
    centri = km.cluster_centers_[ordine]
    campo(ax)
    for c in range(K_SCELTO):
        s = g[g.cluster == c]
        ax.scatter(s.x_dif, s.y_dif, s=18, alpha=0.65, label=f"cluster {c} (n={len(s)})")
    r = df[(df.muro == m) & df.rumore]
    ax.scatter(r.x_dif, r.y_dif, marker="x", s=30, color="lightcoral", label=f"rumore escluso (n={len(r)})")
    sp = sporco[sporco.muro == m]
    ax.scatter(sp.x, sp.y, marker="D", s=70, facecolor="white", edgecolor="gray", lw=1.5, zorder=4,
               label="centroidi sporchi")
    ax.scatter(centri[:, 0], centri[:, 1], marker="*", s=380, color="black", edgecolor="white", lw=1.2,
               zorder=5, label="centroidi puliti")
    ax.set_title(MURO_LABEL[m])
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)

    # baricentro unico della difesa per quel muro: sporco vs pulito
    allm = df[df.muro == m]
    bs, bp = allm[FEATURES].mean(), g[FEATURES].mean()
    print(f"\n{MURO_LABEL[m]}  (n = {len(g)} su {len(allm)})")
    print(f"  baricentro unico  sporco = ({bs.x_dif:.2f}, {bs.y_dif:.2f})   pulito = ({bp.x_dif:.2f}, {bp.y_dif:.2f})")
    print("  cluster    n      x     y   spostamento dal centroide sporco piu' vicino")
    for c in range(K_SCELTO):
        dd = np.hypot(sp.x - centri[c, 0], sp.y - centri[c, 1])
        righe.append({"muro": m, "cluster": c, "x": round(centri[c, 0], 2), "y": round(centri[c, 1], 2),
                      "n": int((g.cluster == c).sum()),
                      "scost_da_sporco_m": round(float(dd.min()), 2)})
        print(f"  {c:>7} {(g.cluster == c).sum():>4} {centri[c,0]:>6.2f} {centri[c,1]:>5.2f}   {dd.min():>6.2f} m")

ax = axs.ravel()[3]
campo(ax)
ax.scatter([X_BASE], [Y_BASE], marker="s", s=140, color="black", label="partenza (posto 5)", zorder=5)
cp = pd.DataFrame(righe)
for m in (0, 1, 2):
    c = cp[cp.muro == m]
    ax.scatter(c.x, c.y, marker="*", s=300, color=MURO_COL[m], edgecolor="black", lw=0.8,
               label=MURO_LABEL[m], zorder=4)
ax.set_title("Centroidi puliti a confronto")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
fig.suptitle("K-Means pulito (dopo DBSCAN): stella = centroide pulito, rombo = centroide sporco", y=0.995)
fig.tight_layout(); fig.savefig("figures/06_kmeans_pulito.png", dpi=150)

df.drop(columns=["emergenza"]).to_csv("data/dataset_dbscan.csv", index=False)
cp.to_csv("data/centroidi_kmeans_pulito.csv", index=False)
print("\nSalvati: figures/05_dbscan.png, figures/06_kmeans_pulito.png, data/dataset_dbscan.csv, data/centroidi_kmeans_pulito.csv")
