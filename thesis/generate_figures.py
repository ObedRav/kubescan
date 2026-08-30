#!/usr/bin/env python3
"""
generate_figures.py
Genera las 7 figuras del TFE kubescan con datos reales del proyecto.
Ejecutar desde: /Users/obedrayo/Documents/UNIR/TFE/
    python3 thesis/generate_figures.py
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, RegularPolygon, Circle, Rectangle, Polygon
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

CHECKPOINTS = os.path.join(
    os.path.dirname(__file__), "..", "research", "models", "checkpoints"
)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 150,
})

# ── FIG 4.1  Architecture ────────────────────────────────────────────────────
_A_BLUE, _A_GREEN, _A_PURPLE, _A_RED, _A_ORANGE, _A_GRAY, _A_DARK = (
    '#2980b9', '#27ae60', '#8e44ad', '#e74c3c', '#e67e22', '#7f8c8d', '#2c3e50',
)

fig, ax = plt.subplots(figsize=(11, 5.6))
ax.set_xlim(0, 11); ax.set_ylim(-0.3, 5.6); ax.axis('off')


def arrow(ax, x1, y, x2, color=_A_DARK, lw=1.8):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=13))


def varrow(ax, x, y1, y2, color=_A_DARK, lw=1.8):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=13))


def box(ax, x, y, w, h, label, sublabel, color, fs=9.5):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                           facecolor=color, edgecolor=_A_DARK, linewidth=1.6, zorder=3)
    ax.add_patch(rect)
    if label:
        ax.text(x + w/2, y + 0.34, label, ha='center', va='center',
                fontsize=fs, fontweight='bold', color='white', zorder=4)
    if sublabel:
        ax.text(x + w/2, y + 0.10, sublabel, ha='center', va='center',
                fontsize=9, color='#ecf0f1', zorder=4)


def file_stack(ax, x, y, color, n=3, scale=0.55):
    for i in range(n):
        dx = dy = i * 0.10 * scale / 0.55
        rect = FancyBboxPatch((x + dx, y + dy), 0.7*scale/0.55, 0.9*scale/0.55,
                               boxstyle="round,pad=0.02", facecolor='white',
                               edgecolor=color, linewidth=1.2, zorder=3 + i)
        ax.add_patch(rect)
    for i in range(3):
        yy = y + (n-1)*0.10 + 0.62 - i*0.15
        ax.plot([x + (n-1)*0.10 + 0.12, x + (n-1)*0.10 + 0.55], [yy, yy],
                color=color, lw=0.9, alpha=0.6, zorder=4+n)


def tree_glyph(ax, cx, cy, color, s=0.34):
    """Simplified decision-tree icon: three small trees (Random Forest)."""
    for ox in (-0.42, 0, 0.42):
        x0, y0 = cx + ox, cy
        ax.plot([x0, x0-0.14*s/0.34], [y0, y0-0.26*s/0.34], color='white', lw=1.3, zorder=6)
        ax.plot([x0, x0+0.14*s/0.34], [y0, y0-0.26*s/0.34], color='white', lw=1.3, zorder=6)
        ax.plot([x0-0.14*s/0.34, x0-0.24*s/0.34], [y0-0.26*s/0.34, y0-0.46*s/0.34], color='white', lw=1.1, zorder=6)
        ax.plot([x0-0.14*s/0.34, x0-0.04*s/0.34], [y0-0.26*s/0.34, y0-0.46*s/0.34], color='white', lw=1.1, zorder=6)
        ax.plot([x0+0.14*s/0.34, x0+0.04*s/0.34], [y0-0.26*s/0.34, y0-0.46*s/0.34], color='white', lw=1.1, zorder=6)
        ax.plot([x0+0.14*s/0.34, x0+0.24*s/0.34], [y0-0.26*s/0.34, y0-0.46*s/0.34], color='white', lw=1.1, zorder=6)
        ax.add_patch(Circle((x0, y0), 0.045, color='white', zorder=7))


def mini_graph(ax, cx, cy, color, attention=False):
    """Tiny 4-node graph glyph; if attention=True, one edge is bold (attention weight)."""
    pts = {
        'a': (cx - 0.32, cy + 0.20), 'b': (cx + 0.30, cy + 0.22),
        'c': (cx - 0.18, cy - 0.24), 'd': (cx + 0.28, cy - 0.20),
    }
    edges = [('a', 'b', 1.0), ('a', 'c', 1.0), ('b', 'd', 2.6 if attention else 1.0),
             ('c', 'd', 1.0)]
    for n1, n2, lw in edges:
        x1, y1 = pts[n1]; x2, y2 = pts[n2]
        ax.plot([x1, x2], [y1, y2], color='white', lw=lw, alpha=0.95, zorder=6,
                solid_capstyle='round')
    for x, y in pts.values():
        ax.add_patch(Circle((x, y), 0.06, color='white', zorder=7))


def hexagon(ax, cx, cy, r, label, sublabel, color):
    hexpatch = RegularPolygon((cx, cy), numVertices=6, radius=r, orientation=np.pi/6,
                               facecolor=color, edgecolor=_A_DARK, linewidth=1.8, zorder=5)
    ax.add_patch(hexpatch)
    ax.text(cx, cy + 0.16, label, ha='center', va='center', fontsize=10.5,
            fontweight='bold', color='white', zorder=6)
    ax.text(cx, cy - 0.20, sublabel, ha='center', va='center', fontsize=9,
            color='#ecf0f1', zorder=6)


def _range_bar_cmap(hexcolor):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('c', ['#ffffff', hexcolor])


def range_bar(ax, x, y, w, h, color, label):
    grad = np.linspace(0.15, 1.0, 256).reshape(1, -1)
    ax.imshow(grad, extent=(x, x+w, y, y+h), aspect='auto', cmap=_range_bar_cmap(color), zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, facecolor='none', edgecolor=_A_DARK, linewidth=1.1, zorder=5))
    ax.text(x, y - 0.16, '0', ha='center', va='top', fontsize=8.5, color='white', zorder=6)
    ax.text(x + w, y - 0.16, '1', ha='center', va='top', fontsize=8.5, color='white', zorder=6)
    ax.text(x + w/2, y + h + 0.16, label, ha='center', va='bottom', fontsize=9,
            color='white', style='italic', fontweight='bold', zorder=6)


def pill(ax, x, y, w, h, text, color):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                           facecolor=color, edgecolor=_A_DARK, linewidth=1.1, zorder=4)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9,
            fontweight='bold', color='white', zorder=5)


# 1. input
file_stack(ax, 0.35, 2.55, _A_GRAY, n=3, scale=0.7)
ax.text(1.05, 2.15, 'YAML\nManifiestos', ha='center', va='top', fontsize=9.5,
        fontweight='bold', color=_A_GRAY)
ax.text(1.05, 1.75, 'Directorio local\n/ kubectl', ha='center', va='top', fontsize=9, color=_A_GRAY)
arrow(ax, 1.75, 2.9, 2.2)

# 2. Capa 1 (RF)
box(ax, 2.2, 1.15, 1.75, 3.5, None, None, _A_BLUE)
ax.text(3.075, 4.35, 'CAPA 1', ha='center', fontsize=9.5, fontweight='bold', color='white', zorder=4)
tree_glyph(ax, 3.075, 3.55, _A_BLUE)
ax.text(3.075, 2.55, 'Random Forest\n500 árboles\n25 features', ha='center', va='center',
        fontsize=9, color='#ecf0f1', zorder=4)
range_bar(ax, 2.55, 1.35, 1.05, 0.16, _A_BLUE, 'risk_score')
arrow(ax, 3.95, 2.9, 4.4)

# 3. Grafo de Cluster
box(ax, 4.4, 2.05, 1.7, 1.7, None, None, _A_PURPLE)
ax.text(5.25, 3.45, 'Grafo de\nClúster', ha='center', fontsize=8.6, fontweight='bold',
        color='white', zorder=4)
mini_graph(ax, 5.25, 2.65, _A_PURPLE)
ax.text(5.25, 1.78, 'G = (V, E)  ·  5 tipos de arista', ha='center', fontsize=8.5,
        color=_A_PURPLE, style='italic')
arrow(ax, 6.1, 2.9, 6.55)

# 4. Capa 2 (GAT)
box(ax, 6.55, 1.15, 1.75, 3.5, None, None, _A_GREEN)
ax.text(7.425, 4.35, 'CAPA 2', ha='center', fontsize=9.5, fontweight='bold', color='white', zorder=4)
mini_graph(ax, 7.425, 3.55, _A_GREEN, attention=True)
ax.text(7.425, 2.55, 'GAT\n3 capas, 4 heads\npooling mean+max', ha='center', va='center',
        fontsize=9, color='#ecf0f1', zorder=4)
range_bar(ax, 6.9, 1.35, 1.05, 0.16, _A_GREEN, 'p_chain')
arrow(ax, 8.3, 2.9, 8.75)

# 5. Capa 3 (Ensemble, fusion hexagon)
hexagon(ax, 9.65, 2.9, 1.0, 'CAPA 3', 'Ensemble · GA', _A_RED)
varrow(ax, 9.65, 1.9, 1.55, color=_A_RED)
pill(ax, 8.95, 1.05, 1.4, 0.32, 'CLEAN', _A_GREEN)
pill(ax, 8.95, 0.66, 1.4, 0.32, 'ISOLATED', _A_ORANGE)
pill(ax, 8.95, 0.27, 1.4, 0.32, 'ATTACK_CHAIN', _A_RED)

# edge-type legend (matches Figura 4.2 exactly), clear of all boxes
_edge_types = [('0', 'dir_prox', _A_GRAY), ('1', 'priv_reach', '#c0392b'),
               ('2', 'sa_lateral', _A_ORANGE), ('3', 'co_ns', _A_BLUE), ('4', 'RBAC', _A_PURPLE)]
_lo, _hi = 2.2, 8.3
_slot = (_hi - _lo) / len(_edge_types)
_ly = 0.55
for i, (code, name, c) in enumerate(_edge_types):
    _cx0 = _lo + _slot * i + 0.08
    ax.plot([_cx0, _cx0 + 0.18], [_ly, _ly], color=c, lw=2.2, zorder=4)
    ax.text(_cx0 + 0.24, _ly, f'({code}) {name}', ha='left', va='center', fontsize=8,
            color='#444', zorder=4)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_arquitectura.pdf'), bbox_inches='tight')
plt.close()
print("fig_arquitectura.pdf OK")

# ── FIG 4.2  Cluster graph ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.6, 5.0))
ax.set_xlim(-0.6, 3.7); ax.set_ylim(-0.7, 3.4); ax.axis('off')

_g_boundary = FancyBboxPatch((-0.35, -0.05), 3.75, 3.15, boxstyle="round,pad=0.02,rounding_size=0.12",
                              facecolor='#f7f9fa', edgecolor='#95a5a6', linewidth=1.4,
                              linestyle=(0, (5, 3)), zorder=1)
ax.add_patch(_g_boundary)
ax.text(-0.28, 2.98, 'Clúster Kubernetes', ha='left', va='top', fontsize=9, color='#7f8c8d',
        style='italic', zorder=2)

nodes = {
    'pod_A': (0.5, 2.5, 'pod-A', '(ESC)', '#c0392b'),
    'pod_B': (2.5, 2.5, 'pod-B', '(lateral)', '#e67e22'),
    'pod_C': (0.5, 0.8, 'pod-C', '(clean)', '#27ae60'),
    'pod_D': (2.5, 0.8, 'pod-D', '(clean)', '#27ae60'),
}


def pod_icon(ax, cx, cy, color, label, sublabel, r=0.36):
    rect = FancyBboxPatch((cx - r, cy - r), 2 * r, 2 * r, boxstyle="round,pad=0.02,rounding_size=0.12",
                           facecolor=color, edgecolor='#2c3e50', linewidth=1.4, zorder=3)
    ax.add_patch(rect)
    bar_w, bar_h = 0.09, 0.22
    for dx in (-0.15, 0.0, 0.15):
        ax.add_patch(Rectangle((cx + dx - bar_w / 2, cy + 0.06), bar_w, bar_h,
                                facecolor='white', edgecolor='none', alpha=0.9, zorder=4))
    ax.text(cx, cy - 0.17, label, ha='center', va='center', fontsize=9,
            fontweight='bold', color='white', zorder=5)
    ax.text(cx, cy - 0.32, sublabel, ha='center', va='center', fontsize=8,
            color='white', zorder=5)


def shield_icon(ax, cx, cy, color, label, sublabel, s=0.38):
    pts = np.array([
        (cx - s, cy + s * 0.75), (cx, cy + s * 1.05), (cx + s, cy + s * 0.75),
        (cx + s, cy - s * 0.5), (cx, cy - s * 1.05), (cx - s, cy - s * 0.5),
    ])
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor='#2c3e50',
                          linewidth=1.4, zorder=3))
    ax.add_patch(Rectangle((cx - 0.09, cy + 0.20), 0.18, 0.14, facecolor='white',
                            edgecolor='none', zorder=4))
    ax.add_patch(mpatches.Arc((cx, cy + 0.34), 0.14, 0.18, theta1=0, theta2=180,
                               color='white', lw=1.7, zorder=4))
    ax.text(cx, cy - 0.03, label, ha='center', va='center', fontsize=9,
            fontweight='bold', color='white', zorder=5)
    ax.text(cx, cy - 0.18, sublabel, ha='center', va='center', fontsize=8,
            color='white', zorder=5)


for name, (x, y, lbl, sub, col) in nodes.items():
    pod_icon(ax, x, y, col, lbl, sub)
role_xy = (1.5, 0.95)
shield_icon(ax, *role_xy, '#8e44ad', 'Role', '(admin)')


def edge(ax, p1, p2, color, style, label, rad=0.1, lx=None, ly=None):
    x1, y1 = p1; x2, y2 = p2
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                                linestyle=style, connectionstyle=f'arc3,rad={rad}',
                                shrinkA=16, shrinkB=16))
    mx = lx if lx is not None else (x1 + x2) / 2
    my = ly if ly is not None else (y1 + y2) / 2 + 0.15
    ax.text(mx, my, label, ha='center', va='center', fontsize=9, color=color,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor=color,
                      linewidth=0.7, alpha=0.92), zorder=6)


edge(ax, (0.5, 2.5), (2.5, 2.5), '#c0392b', 'solid', '(1) priv', rad=0.18)
edge(ax, (0.5, 2.5), (0.5, 0.8), '#c0392b', 'solid', '(1) priv', rad=-0.18, lx=0.18, ly=1.65)
edge(ax, (2.5, 2.5), (2.5, 0.8), '#e67e22', 'dashed', '(2) lateral', rad=0.32, lx=3.05, ly=1.65)
# co-namespace: routed as a low arc well clear of the Role shield
edge(ax, (0.5, 0.8), (2.5, 0.8), '#2980b9', 'dotted', '', rad=0.62)
ax.text(1.5, 0.02, '(3) co-ns', ha='center', va='center', fontsize=9, color='#2980b9',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='#2980b9',
                  linewidth=0.7, alpha=0.92), zorder=6)
edge(ax, role_xy, (2.5, 2.5), '#8e44ad', 'dashed', '(4) RBAC', rad=-0.32, lx=1.55, ly=1.7)

legend_items = [
    mpatches.Patch(color='#c0392b', label='Nodo de escape (ESC)'),
    mpatches.Patch(color='#e67e22', label='Nodo lateral (LAT)'),
    mpatches.Patch(color='#27ae60', label='Nodo limpio'),
    mpatches.Patch(color='#8e44ad', label='Rol privilegiado'),
]
ax.legend(handles=legend_items, fontsize=9, loc='lower left',
          bbox_to_anchor=(0.0, -0.02), framealpha=0.95, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_grafo_cluster.pdf'), bbox_inches='tight')
plt.close()
print("fig_grafo_cluster.pdf OK")

# ── FIG 5.1  RF Feature importance ──────────────────────────────────────────
# Datos reales cargados directamente de rf_results.json (fuente única de verdad)
with open(os.path.join(CHECKPOINTS, "rf_results.json")) as f:
    _rf_results = json.load(f)
_importances = _rf_results["binary"]["feature_importances"]
_top10 = sorted(_importances.items(), key=lambda kv: kv[1], reverse=True)[:10]
features = [name for name, _ in _top10]
importances = [val * 100 for _, val in _top10]
colors = ['#c0392b' if i < 2 else '#2980b9' for i in range(len(features))]

fig, ax = plt.subplots(figsize=(7, 3.6))
bars = ax.barh(features[::-1], importances[::-1], color=colors[::-1],
               edgecolor='white', height=0.65)
ax.set_xlabel('Importancia (%)')
for bar, val in zip(bars, importances[::-1]):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
            f'{val:.2f}%', va='center', fontsize=8.5)
ax.set_xlim(0, 38)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=8.5)
ax.legend(handles=[mpatches.Patch(color='#c0392b', label='Top-2 discriminadores'),
                   mpatches.Patch(color='#2980b9', label='Restantes features')],
          fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_rf_importance.pdf'), bbox_inches='tight')
plt.close()
print("fig_rf_importance.pdf OK")

# ── FIG 5.2  GNN evolution ───────────────────────────────────────────────────
# Datos reales: Tabla 5.2 del capitulo de resultados
phases   = ['Linea\nbase', 'Tras\naugmentacion', 'Correccion\nHOSTPATH', 'Dataset\nextendido']
f1_macro = [0.829, 0.915, 0.915, 0.917]
p_at_5   = [0.400, 0.520, 0.600, 0.880]
f1_err   = [0.098, 0.059, 0.050, 0.065]
p5_err   = [0.179, 0.098, 0.000, 0.098]

x = np.arange(len(phases))
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.errorbar(x, f1_macro, yerr=f1_err, marker='o', linewidth=2,
            color='#2980b9', capsize=4, label='F1 macro (+/-sigma)', markersize=6)
ax.errorbar(x, p_at_5, yerr=p5_err, marker='s', linewidth=2,
            color='#e74c3c', capsize=4, label='Precision@5 (+/-sigma)',
            markersize=6, linestyle='--')
ax.axhline(y=0.70, color='#e74c3c', linestyle=':', linewidth=1.2, alpha=0.7,
           label='Objetivo P@5 = 0.70')
ax.axhline(y=0.85, color='#2980b9', linestyle=':', linewidth=1.2, alpha=0.7,
           label='Objetivo F1 = 0.85')
ax.set_xticks(x); ax.set_xticklabels(phases, fontsize=9)
ax.set_ylim(0.25, 1.02); ax.set_ylabel('Valor de la metrica')
ax.legend(fontsize=8.5, loc='lower right')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_gnn_evolution.pdf'), bbox_inches='tight')
plt.close()
print("fig_gnn_evolution.pdf OK")

# ── FIG 5.3  Confusion matrix ────────────────────────────────────────────────
# Leída del artefacto desplegado (rf_results.json): [[316, 3], [0, 247]]
import json as _json
with open(os.path.join(os.path.dirname(__file__), '..', 'research', 'models',
                       'checkpoints', 'rf_results.json')) as _f:
    _rf = _json.load(_f)
cm = np.array(_rf['binary']['test_metrics']['confusion_matrix'])
labels = ['Seguro', 'Misconfigured']
fig, ax = plt.subplots(figsize=(4.5, 3.8))
im = ax.imshow(cm, cmap='Blues', vmin=0)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(labels, fontsize=9); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Prediccion', fontsize=9); ax.set_ylabel('Etiqueta real', fontsize=9)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14,
                color='white' if cm[i,j]>200 else 'black', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_rf_confusion.pdf'), bbox_inches='tight')
plt.close()
print("fig_rf_confusion.pdf OK")

# ── FIG 5.4  Ensemble P@k ────────────────────────────────────────────────────
# Test desplegado 2026-07-14 (86 grafos, 5 cadenas, restricción estructural):
# P@1=1.00, P@3=1.00, P@5=0.80 (test_results.json ranking_metrics)
fig, ax = plt.subplots(figsize=(4.5, 3.2))
bars = ax.bar(['P@1', 'P@3', 'P@5'], [1.00, 1.00, 0.80],
              color=['#27ae60','#f39c12','#2980b9'], edgecolor='white', width=0.5)
ax.axhline(y=0.70, color='#e74c3c', linestyle='--', linewidth=1.5, label='Objetivo P@5 = 0.70')
ax.set_ylim(0, 1.15); ax.set_ylabel('Precision@k')
for bar, val in zip(bars, [1.00, 1.00, 0.80]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_ensemble_pat_k.pdf'), bbox_inches='tight')
plt.close()
print("fig_ensemble_pat_k.pdf OK")

# ── FIG C.1  CLI pipeline (fusion tree) ─────────────────────────────────────
# Traza el mismo ejemplo real ejecutado en el Apendice C (kubescan scan sobre
# produccion-web/, ver appendix/guia_uso.tex): dos manifiestos, veredicto
# ATTACK_CHAIN, ensemble_score = 0.9037.
BLUE, GREEN, PURPLE, RED, ORANGE, GRAY, DARK = (
    '#2980b9', '#27ae60', '#8e44ad', '#e74c3c', '#e67e22', '#7f8c8d', '#2c3e50',
)


def _varrow(ax, x, y1, y2, color=DARK, lw=1.8):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=14))


def _diag_arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.6, label=None, label_dx=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=12))
    if label:
        mx, my = (x1 + x2) / 2 + label_dx, (y1 + y2) / 2
        ax.text(mx, my, label, ha='center', va='center', fontsize=9, color=color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          edgecolor=color, linewidth=0.8, alpha=0.95), zorder=6)


def _box(ax, x, y, w, h, label, sublabel, color, fs=9):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                           facecolor=color, edgecolor=DARK, linewidth=1.4, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2 + 0.11, label, ha='center', va='center',
            fontsize=fs, fontweight='bold', color='white', zorder=4)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.16, sublabel, ha='center', va='center',
                fontsize=8.5, color='#ecf0f1', zorder=4)


def _terminal_bar(ax, x, y, w, h, text):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                           facecolor=DARK, edgecolor='#111a22', linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    for i, c in enumerate(['#e74c3c', '#f1c40f', '#2ecc71']):
        ax.add_patch(Circle((x + 0.28 + i * 0.26, y + h - 0.26), 0.07, color=c, zorder=4))
    ax.text(x + w / 2, y + h / 2 - 0.13, text, ha='center', va='center', fontsize=9,
            color='#2ecc71', family='monospace', fontweight='bold', zorder=4)


def _file_stack(ax, x, y, color, n=3, label=None):
    for i in range(n):
        dx = dy = i * 0.11
        rect = FancyBboxPatch((x + dx, y + dy), 0.8, 1.0, boxstyle="round,pad=0.02",
                               facecolor='white', edgecolor=color, linewidth=1.3,
                               zorder=3 + i)
        ax.add_patch(rect)
    fold_x, fold_y = x + (n - 1) * 0.11 + 0.8, y + (n - 1) * 0.11 + 1.0
    ax.plot([fold_x - 0.2, fold_x, fold_x], [fold_y, fold_y, fold_y - 0.2],
            color=color, lw=1.3, zorder=4 + n)
    for i in range(3):
        yy = y + (n - 1) * 0.11 + 0.75 - i * 0.19
        ax.plot([x + (n - 1) * 0.11 + 0.13, x + (n - 1) * 0.11 + 0.67], [yy, yy],
                color=color, lw=1.0, alpha=0.6, zorder=4 + n)
    if label:
        cy = y + (n - 1) * 0.055 + 0.5
        ax.text(x + (n - 1) * 0.11 + 1.0, cy, label, ha='left', va='center',
                fontsize=9, color=color, fontweight='bold')


def _hexagon(ax, cx, cy, r, label, sublabel, color):
    hexpatch = RegularPolygon((cx, cy), numVertices=6, radius=r, orientation=np.pi / 6,
                               facecolor=color, edgecolor=DARK, linewidth=1.8, zorder=5)
    ax.add_patch(hexpatch)
    ax.text(cx, cy + 0.13, label, ha='center', va='center', fontsize=9.5,
            fontweight='bold', color='white', zorder=6)
    ax.text(cx, cy - 0.19, sublabel, ha='center', va='center', fontsize=9,
            color='#ecf0f1', zorder=6)


def _gauge(ax, x, y, w, h, score, t1=0.30, t2=0.60):
    for a, b, c in [(0, t1, GREEN), (t1, t2, ORANGE), (t2, 1.0, RED)]:
        ax.add_patch(Rectangle((x + a * w, y), (b - a) * w, h, facecolor=c,
                                edgecolor='none', alpha=0.85, zorder=3))
    ax.add_patch(Rectangle((x, y), w, h, facecolor='none', edgecolor=DARK,
                            linewidth=1.3, zorder=5))
    for t, lbl in [(0, '0'), (t1, '0,30'), (t2, '0,60'), (1.0, '1')]:
        ax.plot([x + t * w, x + t * w], [y, y - 0.08], color=DARK, lw=1.0, zorder=5)
        ax.text(x + t * w, y - 0.22, lbl, ha='center', va='top', fontsize=8.5, color=DARK)
    mx = x + score * w
    ax.plot([mx, mx], [y - 0.05, y + h + 0.05], color='white', lw=2.6, zorder=6)
    ax.plot([mx, mx], [y - 0.05, y + h + 0.05], color=DARK, lw=1.1, zorder=7)
    ax.plot(mx, y + h + 0.22, marker='v', color=DARK, markersize=8, zorder=7)
    score_str = f'{score:.3f}'.replace('.', ',')
    ax.text(mx, y + h + 0.42, f'ensemble_score = {score_str}', ha='center', va='bottom',
            fontsize=9, color=DARK, fontweight='bold')


fig, ax = plt.subplots(figsize=(7.6, 9.3))
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 12.3)
ax.axis('off')

_terminal_bar(ax, 2.4, 11.15, 5.2, 0.85, '$ kubescan scan produccion-web/')
_varrow(ax, 5.0, 11.15, 8.85)

_file_stack(ax, 6.1, 9.55, GRAY, n=3, label='N manifiestos\nYAML')
ax.plot([5.0, 6.1], [9.95, 9.95], color=GRAY, lw=1.2, linestyle=(0, (2, 1.5)), zorder=2)

_box(ax, 3.4, 7.85, 3.2, 0.9, 'extract_cluster_features()',
     'yaml_parser.py  ·  26 features/nodo', GRAY, fs=9.5)
_varrow(ax, 5.0, 7.85, 6.95)

_box(ax, 3.4, 6.05, 3.2, 0.9, 'build_cluster_graph()',
     'graph_builder.py  ·  G = (V, E)', PURPLE, fs=9.5)

y_branch = 4.35
_diag_arrow(ax, 4.2, 6.05, 1.6, y_branch + 0.9, color=BLUE)
_diag_arrow(ax, 5.0, 6.05, 5.0, y_branch + 0.9, color=GREEN)
_diag_arrow(ax, 5.8, 6.05, 8.4, y_branch + 0.9, color=RED)

_box(ax, 0.5, y_branch, 2.2, 0.9, 'Capa 1 · RF', 'predict_risk_scores()', BLUE, fs=9.5)
_box(ax, 3.9, y_branch, 2.2, 0.9, 'Capa 2 · GAT', 'run_gnn_ensemble()', GREEN, fs=9.5)
_box(ax, 7.3, y_branch, 2.2, 0.9, 'Señal de escape', 'compute_escape_signal()', RED, fs=9.5)

ax.text(1.6, y_branch - 0.26, 'mean_rf_risk = 0,996', ha='center', fontsize=9,
        color=BLUE, style='italic')
ax.text(5.0, y_branch - 0.26, 'chain_probability = 0,710', ha='center', fontsize=9,
        color=GREEN, style='italic')
ax.text(8.4, y_branch - 0.26, 'escape_signal = 1,0', ha='center', fontsize=9,
        color=RED, style='italic')

fx, fy = 5.0, 1.75
_diag_arrow(ax, 1.6, y_branch - 0.46, fx - 0.62, fy + 0.5, color=BLUE, label='w_rf', label_dx=-0.35)
_diag_arrow(ax, 5.0, y_branch - 0.46, fx, fy + 0.85, color=GREEN, label='w_gnn', label_dx=0.42)
_diag_arrow(ax, 8.4, y_branch - 0.46, fx + 0.62, fy + 0.5, color=RED, label='w_esc', label_dx=0.36)

_hexagon(ax, fx, fy, 0.95, 'EnsembleScorer', '.score()', DARK)
_varrow(ax, fx, fy - 0.95, 0.75)

_gauge(ax, 2.2, -0.05, 5.6, 0.5, score=0.947)
_varrow(ax, fx, -0.15, -0.95)

_box(ax, 3.3, -1.45, 3.4, 0.9, 'ATTACK_CHAIN', 'informe text / JSON', RED, fs=9.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_cli_pipeline.pdf'), bbox_inches='tight')
plt.close()
print("fig_cli_pipeline.pdf OK")

# ── FIG 5.x  Loss x selection ablation ───────────────────────────────────────
# Datos: Tabla tab:loss_ablation (corpus completo, particionado por familias,
# semilla 42). Solo el brazo desplegado ce_p5 (0.72) esta en
# checkpoints/cv_results.json; los otros tres brazos proceden de la ablacion
# factorial 2x2 (no desplegados).
_la_labels = ['Focal (gamma=2)', 'Entropia cruzada\nponderada']
_la_p5_f1  = [0.480, 0.600]   # seleccion de checkpoint por F1 macro
_la_p5_p5  = [0.680, 0.720]   # seleccion de checkpoint por P@5
_la_err_f1 = [0.271, 0.283]
_la_err_p5 = [0.204, 0.160]
x = np.arange(len(_la_labels)); w = 0.36
fig, ax = plt.subplots(figsize=(6.2, 3.7))
b1 = ax.bar(x - w/2, _la_p5_f1, w, yerr=_la_err_f1, capsize=4, color='#95a5a6',
            edgecolor='white', label='Seleccion por F1 macro')
b2 = ax.bar(x + w/2, _la_p5_p5, w, yerr=_la_err_p5, capsize=4, color='#2980b9',
            edgecolor='white', label='Seleccion por P@5')
ax.axhline(0.70, color='#e74c3c', linestyle='--', linewidth=1.3,
           label='Objetivo P@5 = 0.70')
for bars, vals in ((b1, _la_p5_f1), (b2, _la_p5_p5)):
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f'{v:.2f}',
                ha='center', va='bottom', fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(_la_labels, fontsize=9)
ax.set_ylabel('Precision@5 (validacion cruzada)'); ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc='upper left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_loss_ablation.pdf'), bbox_inches='tight')
plt.close()
print("fig_loss_ablation.pdf OK")

# ── FIG 5.x  Structural feasibility gate: before / after (test set) ───────────
_t = json.load(open(os.path.join(CHECKPOINTS, 'test_results.json')))
_g = _t['ranking_metrics']; _u = _t['ranking_metrics_ungated']
_metrics = ['P@1', 'P@3', 'P@5']
_ungated = [_u['precision_at_1'], _u['precision_at_3'], _u['precision_at_5']]
_gated   = [_g['precision_at_1'], _g['precision_at_3'], _g['precision_at_5']]
x = np.arange(len(_metrics)); w = 0.36
fig, ax = plt.subplots(figsize=(6.2, 3.7))
b1 = ax.bar(x - w/2, _ungated, w, color='#95a5a6', edgecolor='white',
            label='Sin restriccion estructural')
b2 = ax.bar(x + w/2, _gated, w, color='#27ae60', edgecolor='white',
            label='Con restriccion estructural')
ax.axhline(0.70, color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.7,
           label='Objetivo P@5 = 0.70')
for bars, vals in ((b1, _ungated), (b2, _gated)):
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f'{v:.2f}',
                ha='center', va='bottom', fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(_metrics); ax.set_ylim(0, 1.12)
ax.set_ylabel('Precision (conjunto de test)')
ax.legend(fontsize=8, loc='upper left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_structural_gate.pdf'), bbox_inches='tight')
plt.close()
print("fig_structural_gate.pdf OK")

# ── FIG 5.x  Multi-seed per-fold P@5 (robustness) ────────────────────────────
def _fold_p5(*parts):
    return json.load(open(os.path.join(CHECKPOINTS, *parts)))['fold_p5s']
_seed_folds = {
    42:  _fold_p5('cv_results.json'),
    7:   _fold_p5('..', 'checkpoints_seeds', 'seed_7', 'cv_results.json'),
    123: _fold_p5('..', 'checkpoints_seeds', 'seed_123', 'cv_results.json'),
}
_folds = np.arange(5)
_mk = {42: 'o', 7: 's', 123: '^'}
_col = {42: '#2980b9', 7: '#e74c3c', 123: '#f39c12'}
fig, ax = plt.subplots(figsize=(6.6, 3.7))
for s in (42, 7, 123):
    ax.plot(_folds, _seed_folds[s], marker=_mk[s], color=_col[s], linewidth=1.6,
            markersize=7, alpha=0.85, label=f'Semilla {s}')
ax.axhline(0.70, color='#7f8c8d', linestyle=':', linewidth=1.0,
           label='Objetivo P@5 = 0.70')
ax.annotate('unico pliegue con varianza\nentre semillas',
            xy=(1, 0.4), xytext=(1.35, 0.17), fontsize=8, color='#555',
            arrowprops=dict(arrowstyle='->', color='#888', lw=1))
ax.set_xticks(_folds)
ax.set_xticklabels([f'Pliegue {i}' for i in range(5)], fontsize=8.5)
ax.set_ylabel('Precision@5'); ax.set_ylim(0, 1.1)
ax.legend(fontsize=8, ncol=2, loc='lower right')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_multiseed_folds.pdf'), bbox_inches='tight')
plt.close()
print("fig_multiseed_folds.pdf OK")

# ── FIG 5.x  Ensemble test confusion matrix (replaces stale orphan) ──────────
_cm = np.array(_t['classification_metrics']['confusion_matrix'])
_cls = ['Limpio', 'Aislado', 'Cadena']
fig, ax = plt.subplots(figsize=(4.7, 4.0))
im = ax.imshow(_cm, cmap='Blues', vmin=0)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(_cls, fontsize=9); ax.set_yticklabels(_cls, fontsize=9)
ax.set_xlabel('Prediccion', fontsize=9); ax.set_ylabel('Etiqueta real', fontsize=9)
_thr = _cm.max() / 2.0
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(_cm[i, j]), ha='center', va='center', fontsize=13,
                color='white' if _cm[i, j] > _thr else 'black', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_gnn_confusion.pdf'), bbox_inches='tight')
plt.close()
print("fig_gnn_confusion.pdf OK")

print(f"\nTodas las figuras generadas en: {OUT}")
