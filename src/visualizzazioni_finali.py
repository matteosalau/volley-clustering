"""
visualizzazioni_finali.py  (Fase 4)
Figure finali del progetto "posizioni difensive del libero".
Eseguire dalla cartella principale:  python src/visualizzazioni_finali.py
Input : data/dataset_dbscan.csv, data/centroidi_kmeans_pulito.csv, data/centroidi_kmeans_sporco.csv
Output: figures/07_mappa_finale.png ... figures/11_outlier_per_zona.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)

X_BASE, Y_BASE = 1.8, 3.6
MURO_LABEL = {0: "Muro 0 - parallela chiusa", 1: "Muro 1 - diagonale chiusa", 2: "Muro 2 - disorganizzato"}
MURO_SHORT = {0: "Parallela chiusa", 1: "Diagonale chiusa", 2: "Disorganizzato"}
MURO_COL = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
ZONA_ORD = [1, 6, 2, 3, 4]   # da sinistra a destra nel sistema del libero (x medio crescente: 1.8,4.5,1.2,...)
ZONA_ORD = sorted([1, 2, 3, 4, 6], key=lambda z: {1: 1.8, 2: 1.2, 3: 4.5, 4: 7.8, 6: 4.5}[z])

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv(f"{DATA}/dataset_dbscan.csv")
cp = pd.read_csv(f"{DATA}/centroidi_kmeans_pulito.csv")
cs = pd.read_csv(f"{DATA}/centroidi_kmeans_sporco.csv")
std = df[~df.rumore]
out = df[df.rumore]


def campo(ax, label_rete=True):
    """Meta' campo difensiva in scala 1:1 (9 x 9 m), rete evidenziata in alto."""
    ax.add_patch(Rectangle((0, 0), 9, 9, facecolor="#f3ead8", edgecolor="black", lw=1.8, zorder=0))
    ax.add_patch(Rectangle((0, 6), 9, 3, facecolor="#e8d9b8", edgecolor="none", zorder=0))  # zona d'attacco (3 m)
    ax.plot([0, 9], [6, 6], color="gray", lw=1, ls="--", zorder=1)
    ax.plot([0, 9], [9, 9], color="#c0392b", lw=6, solid_capstyle="butt", zorder=2)         # rete
    if label_rete:
        ax.text(4.5, 9.45, "RETE", ha="center", va="bottom", color="#c0392b", fontsize=9, fontweight="bold")
    ax.text(8.9, 6.1, "linea dei 3 m", ha="right", va="bottom", fontsize=7, color="gray")
    ax.set_xlim(-0.5, 9.5); ax.set_ylim(-0.5, 10.1); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)  -  distanza dal fondo campo")


# ---------------------------------------------------------------- 07 mappa finale (tutti i muri)
fig, ax = plt.subplots(figsize=(8.6, 9))
campo(ax)
for m in (0, 1, 2):
    s = std[std.muro == m]
    ax.scatter(s.x_dif, s.y_dif, s=20, alpha=0.55, color=MURO_COL[m], edgecolor="none", zorder=3)
    ax.scatter(cs[cs.muro == m].x, cs[cs.muro == m].y, marker="D", s=60, facecolor="white",
               edgecolor=MURO_COL[m], lw=1.4, zorder=5)
    c = cp[cp.muro == m]
    ax.scatter(c.x, c.y, marker="*", s=420, color=MURO_COL[m], edgecolor="black", lw=1.3, zorder=6)
ax.scatter(out.x_dif, out.y_dif, marker="X", s=55, color="black", edgecolor="white", lw=0.6, zorder=7)
ax.scatter([X_BASE], [Y_BASE], marker="s", s=120, color="black", zorder=7)
h = [Line2D([], [], marker="o", ls="", color=MURO_COL[m], alpha=0.8, label=f"{MURO_LABEL[m]} (n={len(std[std.muro == m])})") for m in (0, 1, 2)]
h += [Line2D([], [], marker="*", ls="", color="lightgray", mec="black", ms=15, label="Centroide K-Means pulito (k=3 per muro)"),
      Line2D([], [], marker="D", ls="", mfc="white", mec="gray", ms=7, label="Centroide K-Means sporco (confronto)"),
      Line2D([], [], marker="X", ls="", color="black", mec="white", ms=9, label=f"Outlier DBSCAN (n={len(out)})"),
      Line2D([], [], marker="s", ls="", color="black", ms=8, label="Posizione di partenza")]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8, frameon=False)
ax.set_title("Posizioni difensive del libero - campo in scala (9 x 9 m)", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/07_mappa_finale.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 08 small multiples per muro
fig, axs = plt.subplots(1, 3, figsize=(15, 5.8), sharey=True)
for ax, m in zip(axs, (0, 1, 2)):
    campo(ax, label_rete=True)
    s = std[std.muro == m]
    ax.scatter(s.x_dif, s.y_dif, s=22, alpha=0.6, color=MURO_COL[m], edgecolor="none", zorder=3)
    o = out[out.muro == m]
    ax.scatter(o.x_dif, o.y_dif, marker="X", s=60, color="black", edgecolor="white", lw=0.6, zorder=7)
    ax.scatter(cs[cs.muro == m].x, cs[cs.muro == m].y, marker="D", s=65, facecolor="white", edgecolor="gray", lw=1.5, zorder=5)
    c = cp[cp.muro == m]
    ax.scatter(c.x, c.y, marker="*", s=430, color=MURO_COL[m], edgecolor="black", lw=1.3, zorder=6)
    ax.scatter([X_BASE], [Y_BASE], marker="s", s=70, color="black", zorder=7)
    ax.set_title(f"{MURO_LABEL[m]}\n{len(s)} azioni, {len(o)} outlier", fontsize=10)
    if ax is not axs[0]:
        ax.set_ylabel("")
hh = h[3:6]; hh[2] = Line2D([], [], marker="X", ls="", color="black", mec="white", ms=9, label="Outlier DBSCAN"); hh.append(h[6])
fig.legend(handles=hh, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
fig.suptitle("Stesso campo, stessa scala: un pannello per tipo di muro", fontweight="bold")
fig.tight_layout(rect=(0, 0.11, 1, 0.96)); fig.savefig(f"{OUT}/08_mappa_per_muro.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- 09 frecce di spostamento zona x muro
fig, ax = plt.subplots(figsize=(9, 7.4))
campo(ax)
ax.scatter([X_BASE], [Y_BASE], marker="s", s=130, color="black", zorder=8)
ax.set_xlim(-0.2, 5.2); ax.set_ylim(1.3, 5.6)  # zoom (scala 1:1 mantenuta)
for t in list(ax.texts):
    if t.get_text() in ("RETE", "linea dei 3 m"):
        t.remove()
off = {0: -0.08, 1: 0.0, 2: 0.08}
for m in (0, 1, 2):
    for z in ZONA_ORD:
        g = std[(std.muro == m) & (std.zona_attacco == z)]
        if len(g) < 5:
            continue
        mx, my = g.x_dif.mean(), g.y_dif.mean()
        ax.annotate("", xy=(mx, my), xytext=(X_BASE, Y_BASE + off[m] * 3),
                    arrowprops=dict(arrowstyle="-|>", color=MURO_COL[m], lw=1 + g.shape[0] / 40, alpha=0.75), zorder=4)
        ax.text(mx, my + 0.08, f"z{z}", color=MURO_COL[m], fontsize=9, fontweight="bold", ha="center", zorder=9)
h2 = [Line2D([], [], color=MURO_COL[m], lw=3, label=MURO_LABEL[m]) for m in (0, 1, 2)]
ax.legend(handles=h2, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8, frameon=False)
ax.set_title("Spostamento medio dal posto 5, per zona d'attacco (z) e muro\n(dettaglio ingrandito; spessore ~ n. azioni; outlier esclusi)", fontweight="bold", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/09_spostamento_per_zona.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 10 x_dif vs x_att (regressione per muro)
fig, ax = plt.subplots(figsize=(8.4, 5.6))
for m in (0, 1, 2):
    g = std[std.muro == m]
    ax.scatter(g.x_att, g.x_dif, s=14, alpha=0.35, color=MURO_COL[m], edgecolor="none")
    a, b = np.polyfit(g.x_att, g.x_dif, 1)
    xs = np.linspace(g.x_att.min(), g.x_att.max(), 50)
    ax.plot(xs, a * xs + b, color=MURO_COL[m], lw=2.5, label=f"{MURO_SHORT[m]}: pendenza {a:+.2f}")
ax.axhline(X_BASE, color="black", ls=":", lw=1); ax.text(0.2, X_BASE + 0.08, "x di partenza (posto 5)", fontsize=8)
ax.set_xlabel("x dell'attaccante (m)"); ax.set_ylabel("x del libero (m)")
ax.set_title("Quanto la posizione laterale dell'attaccante sposta il libero", fontweight="bold")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/10_xdif_vs_xatt.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- 11 profondita' per tipo di colpo + outlier per zona
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axs[0]
pos = 0; ticks = []; labs = []
for m in (0, 1, 2):
    for t, shade in (("forte", 1.0), ("piazzato", 0.45)):
        g = std[(std.muro == m) & (std.tipo_colpo == t)].y_dif
        bp = ax.boxplot(g, positions=[pos], widths=0.7, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black"))
        bp["boxes"][0].set(facecolor=MURO_COL[m], alpha=shade, edgecolor="black")
        ax.scatter(np.random.default_rng(1).normal(pos, 0.1, len(g)), g, s=6, color="black", alpha=0.25, zorder=3)
        ticks.append(pos); labs.append(t); pos += 1
    pos += 0.8
ax.set_xticks(ticks); ax.set_xticklabels(labs)
for i, m in enumerate((0, 1, 2)):
    ax.text(i * 2.8 + 0.5, ax.get_ylim()[0] - 0.7, MURO_SHORT[m], ha="center", color=MURO_COL[m], fontweight="bold", fontsize=9)
ax.set_ylabel("y del libero (m)  -  piu' alto = piu' vicino alla rete")
ax.set_title("Profondità per muro e tipo di colpo", fontweight="bold")

ax = axs[1]
tab = out.groupby("zona_attacco").size().reindex(ZONA_ORD, fill_value=0)
tot = df.groupby("zona_attacco").size().reindex(ZONA_ORD)
ax.bar([str(z) for z in ZONA_ORD], 100 * tab / tot, color="black", alpha=0.8)
for i, z in enumerate(ZONA_ORD):
    ax.text(i, 100 * tab[z] / tot[z] + 0.3, f"{tab[z]}/{tot[z]}", ha="center", fontsize=8)
ax.set_xlabel("zona d'attacco"); ax.set_ylabel("% azioni marcate outlier da DBSCAN")
ax.set_title("Dove cadono gli outlier", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/11_profondita_e_outlier.png", dpi=160); plt.close(fig)
print("Salvate:", sorted(f for f in os.listdir(OUT) if f[:2] in ("07", "08", "09", "10", "11")))
