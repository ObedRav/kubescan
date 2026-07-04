#!/usr/bin/env python3
"""
generate_figures.py
Genera las 6 figuras del TFE kubescan con datos reales del proyecto.
Ejecutar desde: /Users/obedrayo/Documents/UNIR/TFE/
    python3 thesis/generate_figures.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 150,
})

# ── FIG 4.1  Architecture ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.5); ax.axis('off')

def box(ax, x, y, w, h, label, sublabel, color, fs=8.5):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor='#2c3e50', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2+0.12, label, ha='center', va='center',
            fontsize=fs, fontweight='bold', color='white', zorder=4)
    ax.text(x+w/2, y+h/2-0.18, sublabel, ha='center', va='center',
            fontsize=7.5, color='#ecf0f1', zorder=4)

def arrow(ax, x1, y, x2):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.8))

box(ax, 0.1, 1.5, 1.6, 1.5, 'YAML\nManifiestos', 'Directorio\nlocal / kubectl', '#7f8c8d')
arrow(ax, 1.7, 2.25, 2.15)
box(ax, 2.15, 0.7, 1.8, 3.1, 'CAPA 1', 'Random Forest\n500 arboles\n25 features', '#2980b9')
ax.text(3.05, 0.4, 'risk_score in [0,1]', ha='center', fontsize=7.5, color='#2980b9', style='italic')
arrow(ax, 3.95, 2.25, 4.4)
box(ax, 4.4, 1.5, 1.6, 1.5, 'Grafo de\nCluster', '5 tipos de\naristas', '#8e44ad')
ax.text(5.2, 1.15, 'G = (V, E)', ha='center', fontsize=7.5, color='#8e44ad', style='italic')
arrow(ax, 6.0, 2.25, 6.45)
box(ax, 6.45, 0.7, 1.8, 3.1, 'CAPA 2', 'GAT\n3 capas, 4 heads\npooling mean+max', '#27ae60')
ax.text(7.35, 0.4, 'p_chain in [0,1]', ha='center', fontsize=7.5, color='#27ae60', style='italic')
arrow(ax, 8.25, 2.25, 8.7)
box(ax, 8.7, 1.5, 1.2, 1.5, 'CAPA 3', 'Ensemble\nGA', '#e74c3c')
ax.text(9.3, 1.0, 'CLEAN / ISOLATED\n/ ATTACK_CHAIN', ha='center', fontsize=7.5,
        color='#c0392b', fontweight='bold')
ax.text(0.1, 0.3,
        'Tipos de arista: (0) dir_proximity  (1) privilege_reach  (2) sa_lateral  (3) co_namespace  (4) RBAC_priv',
        fontsize=7, color='#555', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_arquitectura.pdf'), bbox_inches='tight')
plt.close()
print("fig_arquitectura.pdf OK")

# ── FIG 4.2  Cluster graph ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.2); ax.axis('off')
nodes = {
    'pod_A': (0.5, 2.5, 'pod-A\n(ESC)', '#c0392b'),
    'pod_B': (2.5, 2.5, 'pod-B\n(lateral)', '#e67e22'),
    'pod_C': (0.5, 0.8, 'pod-C\n(clean)', '#27ae60'),
    'role':  (1.5, 0.8, 'Role\n(admin)', '#8e44ad'),
    'pod_D': (2.5, 0.8, 'pod-D\n(clean)', '#27ae60'),
}
for name, (x, y, lbl, col) in nodes.items():
    ax.add_patch(plt.Circle((x, y), 0.35, color=col, zorder=3, alpha=0.9))
    ax.text(x, y, lbl, ha='center', va='center', fontsize=7.5,
            color='white', fontweight='bold', zorder=4)

def edge(ax, n1, n2, color, style, label, rad=0.1):
    x1, y1 = nodes[n1][:2]; x2, y2 = nodes[n2][:2]
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                                linestyle=style, connectionstyle=f'arc3,rad={rad}'))
    ax.text(mx, my+0.15, label, ha='center', fontsize=7, color=color,
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))

edge(ax, 'pod_A', 'pod_B', '#c0392b', 'solid', '(1) priv', 0.15)
edge(ax, 'pod_A', 'pod_C', '#c0392b', 'solid', '(1) priv', -0.1)
edge(ax, 'pod_B', 'pod_D', '#e67e22', 'dashed', '(2) lateral', 0.1)
ax.plot([0.5, 2.5], [0.8, 0.8], color='#2980b9', lw=1.5, linestyle='dotted', zorder=2)
ax.text(1.5, 0.55, '(3) co-ns', ha='center', fontsize=7, color='#2980b9')
edge(ax, 'role', 'pod_B', '#8e44ad', 'dashed', '(4) RBAC', 0.2)

legend_items = [
    mpatches.Patch(color='#c0392b', label='Nodo de escape (ESC)'),
    mpatches.Patch(color='#e67e22', label='Nodo lateral (LAT)'),
    mpatches.Patch(color='#27ae60', label='Nodo limpio'),
    mpatches.Patch(color='#8e44ad', label='Rol privilegiado'),
]
ax.legend(handles=legend_items, fontsize=7.5, loc='lower left', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_grafo_cluster.pdf'), bbox_inches='tight')
plt.close()
print("fig_grafo_cluster.pdf OK")

# ── FIG 5.1  RF Feature importance ──────────────────────────────────────────
# Datos reales: narrative doc total_misconfigs=33.25%, INSECURE_HTTP=24.84%
features = ['total_misconfigs','INSECURE_HTTP','all_secrets','cap_misuse',
            'SA_AUTOMOUNT','CAP_SYS_ADMIN','TRUE_HOST_PID','HOSTPATH_MOUNT',
            'IMAGE_LATEST','NO_RUN_AS_ROOT']
importances = [33.25, 24.84, 7.41, 5.83, 4.92, 3.17, 2.88, 2.61, 2.34, 1.97]
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
# Datos reales: 3 FP, 0 FN. 496 test muestras, 22.3% misconfig -> ~111 misc
# safe=385 -> TP_safe=382, FP=3 | misc=111 -> TP_misc=111, FN=0
cm = np.array([[382, 3], [0, 111]])
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
# Datos reales: P@1=1.00, P@3=0.67, P@5=0.80 (test set, 15 grafos, 4 cadenas)
fig, ax = plt.subplots(figsize=(4.5, 3.2))
bars = ax.bar(['P@1', 'P@3', 'P@5'], [1.00, 0.67, 0.80],
              color=['#27ae60','#f39c12','#2980b9'], edgecolor='white', width=0.5)
ax.axhline(y=0.70, color='#e74c3c', linestyle='--', linewidth=1.5, label='Objetivo P@5 = 0.70')
ax.set_ylim(0, 1.15); ax.set_ylabel('Precision@k')
for bar, val in zip(bars, [1.00, 0.67, 0.80]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_ensemble_pat_k.pdf'), bbox_inches='tight')
plt.close()
print("fig_ensemble_pat_k.pdf OK")

print(f"\nTodas las figuras generadas en: {OUT}")
