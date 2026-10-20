"""
Multi-Modal Fusion & Dynamic Gating Synergy
IMPROVED VERSION - larger fonts, cleaner bar charts, better annotation,
more padding in info boxes.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 1.4,
})

TITLE_FS = 22
SECT_FS  = 17
LABEL_FS = 15
TICK_FS  = 13
VAL_FS   = 13.5

# Data
molecules = [
    {
        'name': 'Exemplar 1: Fenarimol Analog',
        'smiles': 'OC(c1ccc(Cl)cc1)(c2cncnc2)c3ccccc3Cl',
        'type': 'Fused Aromatic: TriCyclic | MW 332, LogP 4.07',
        'w1': 0.0272, 'w2': 0.0510, 'w3': 0.9218,
        'y_true': -4.38,
        'y_v1': -3.16, 'y_v2': -3.54, 'y_v3': -5.23, 'y_fused': -4.46,
        'synergy': 'Synergistic Error Cancellation: View 1 (-3.16) and View 3 (-5.23)\nare opposing → Fusion corrects to -4.46 (AE=0.08)',
        'color': '#2980b9', 'bg': '#e8f4fd', 'box': '#d6eaf8',
    },
    {
        'name': 'Exemplar 2: Spiro-Hydantoin Complex',
        'smiles': 'O=C1NC(=O)C2(CCCCC2)N1',
        'type': 'Spiro: Non-planar Bicyclic | MW 182, LogP -0.79',
        'w1': 0.0317, 'w2': 0.0333, 'w3': 0.9350,
        'y_true': -2.35,
        'y_v1': -2.81, 'y_v2': -2.64, 'y_v3': -3.31, 'y_fused': -2.93,
        'synergy': 'Spiro-Geometry Dominance: Non-planar envelope → 93.5%\ngating to 3D EGNN (Spiro-Hydantoin)',
        'color': '#27ae60', 'bg': '#e8fdf0', 'box': '#d5f5e3',
    },
    {
        'name': 'Exemplar 3: Anthanthrene PAH',
        'smiles': 'c1cc2ccc3cccc4ccc5cccc6ccc1c1c2c3c4c5c61',
        'type': 'Rigid Planar PAH: 6 Aromatic Rings | MW 276, LogP 6.53',
        'w1': 0.1210, 'w2': 0.0350, 'w3': 0.8440,
        'y_true': -6.33,
        'y_v1': -5.17, 'y_v2': -5.84, 'y_v3': -5.94, 'y_fused': -6.71,
        'synergy': 'Aromatic Routing Boost: Planar PAH → 4× more attention\nto 1D grammar (w1=12.1%) and 2D topology (w2=3.5%)',
        'color': '#e67e22', 'bg': '#fdf5e8', 'box': '#fdebd0',
    },
]

BAR_COLORS = ['#3498db', '#27ae60', '#e67e22', '#2c3e50', '#7f8c8d']

# ─── Canvas ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(28, 34), facecolor='white')
fig.text(0.5, 0.997, 'Multi-Modal Fusion & Dynamic Gating Synergy in TriBioNode v2',
         ha='center', va='top', fontsize=TITLE_FS+2, fontweight='bold', color='#1a1a2e')
fig.text(0.5, 0.982, 'Instance-Adaptive Attention Allocation and Cross-View Orthogonal Error Cancellation',
         ha='center', va='top', fontsize=SECT_FS-1, color='#444')

outer = gridspec.GridSpec(3, 1, figure=fig,
                          top=0.975, bottom=0.02,
                          hspace=0.10, left=0.03, right=0.97)

for mi, mol in enumerate(molecules):
    row_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[mi],
        width_ratios=[1.05, 1.50, 1.45],
        wspace=0.30
    )

    # ── Col 0: Info card ──────────────────────────────────────────────
    ax_info = fig.add_subplot(row_gs[0])
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
    ax_info.axis('off')

    card = FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
                          boxstyle='round,pad=0.02',
                          facecolor=mol['bg'],
                          edgecolor=mol['color'], linewidth=2.5)
    ax_info.add_patch(card)
    # Header
    ax_info.add_patch(FancyBboxPatch((0.03, 0.84), 0.94, 0.13,
                                     boxstyle='round,pad=0.01',
                                     facecolor=mol['color'], alpha=0.88,
                                     edgecolor='none'))
    ax_info.text(0.50, 0.907, mol['name'], ha='center', va='center',
                 fontsize=SECT_FS-1, fontweight='bold', color='white',
                 transform=ax_info.transAxes)
    # SMILES (truncated)
    smiles = mol['smiles']
    if len(smiles) > 34:
        smiles = smiles[:34] + '…'
    ax_info.text(0.50, 0.800, smiles, ha='center', va='center',
                 fontsize=9.5, family='monospace', color='#333',
                 transform=ax_info.transAxes)
    ax_info.text(0.50, 0.760, mol['type'], ha='center', va='center',
                 fontsize=TICK_FS-1, color='#555', transform=ax_info.transAxes)

    # Experimental + Fused prediction box
    ax_info.add_patch(FancyBboxPatch((0.06, 0.48), 0.88, 0.26,
                                     boxstyle='round,pad=0.02',
                                     facecolor=mol['box'], linewidth=1.5,
                                     edgecolor=mol['color']))
    ax_info.text(0.50, 0.755, 'Experimental Benchmark Target:',
                 ha='center', va='center', fontsize=LABEL_FS-1,
                 fontweight='bold', color='#1a1a2e',
                 transform=ax_info.transAxes)
    ax_info.text(0.50, 0.712,
                 f'y_true = {mol["y_true"]:.2f} log(mol/L)',
                 ha='center', va='center', fontsize=LABEL_FS+1,
                 fontweight='bold', color='#1a1a2e',
                 transform=ax_info.transAxes)
    ax_info.add_patch(FancyBboxPatch((0.08, 0.495), 0.84, 0.06,
                                     boxstyle='round,pad=0.01',
                                     facecolor='white', linewidth=1.0,
                                     edgecolor=mol['color'], alpha=0.8))
    ae = abs(mol['y_fused'] - mol['y_true'])
    ax_info.text(0.50, 0.528,
                 f'→ TriBioNode Fused: {mol["y_fused"]:.2f}  (Error = {ae:.2f})',
                 ha='center', va='center', fontsize=LABEL_FS,
                 fontweight='bold', color='#27ae60',
                 transform=ax_info.transAxes)

    # Synergy text
    ax_info.add_patch(FancyBboxPatch((0.04, 0.06), 0.92, 0.39,
                                     boxstyle='round,pad=0.02',
                                     facecolor='#f8f9fa', linewidth=1.2,
                                     edgecolor='#aaa'))
    ax_info.text(0.50, 0.46, 'Multi-Modal Synergy Mechanism:', ha='center',
                 va='center', fontsize=LABEL_FS-1, fontstyle='italic',
                 color='#555', transform=ax_info.transAxes)
    for li, ln in enumerate(mol['synergy'].split('\n')[:3]):
        ax_info.text(0.50, 0.415 - li*0.055, ln.strip(),
                     ha='center', va='center', fontsize=TICK_FS,
                     color='#333', transform=ax_info.transAxes)

    # ── Col 1: Gating bar chart ───────────────────────────────────────
    ax_gate = fig.add_subplot(row_gs[1])
    ax_gate.set_facecolor('#fafafa')

    ws = [mol['w1'], mol['w2'], mol['w3']]
    labels = ['View 1\n(1D SMILES)', 'View 2\n(2D Graph)', 'View 3\n(3D EGNN)']
    colors_gate = ['#3498db', '#27ae60', '#e67e22']
    pct = [w*100 for w in ws]

    bars = ax_gate.barh(labels, ws, color=colors_gate, edgecolor='white',
                        linewidth=1.5, height=0.55, zorder=3)
    ax_gate.set_xlim(0, 1.08)
    ax_gate.set_xlabel('Gating Weight Allocation: $\\alpha_k \\in [0, 1]$',
                       fontsize=LABEL_FS)
    ax_gate.set_title(f'Dynamic Gating Attention Allocation $[w_1, w_2, w_3]$',
                      fontsize=SECT_FS-2, fontweight='bold', pad=10)
    ax_gate.tick_params(axis='y', labelsize=LABEL_FS)
    ax_gate.tick_params(axis='x', labelsize=TICK_FS)
    ax_gate.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_gate.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                             fontsize=TICK_FS)
    ax_gate.spines['top'].set_visible(False)
    ax_gate.spines['right'].set_visible(False)
    ax_gate.grid(axis='x', alpha=0.3, zorder=0)

    # Value annotations
    for bar, p, w in zip(bars, pct, ws):
        x_end = bar.get_width()
        ax_gate.text(min(x_end + 0.02, 1.05), bar.get_y() + bar.get_height()/2,
                     f'{p:.1f}%', va='center', ha='left',
                     fontsize=VAL_FS+1, fontweight='bold', color='#1a1a2e')

    # ── Col 2: Model output comparison bars ──────────────────────────
    ax_out = fig.add_subplot(row_gs[2])
    ax_out.set_facecolor('#fafafa')

    bar_labels = ['View 1\n(1D SMILES)', 'View 2\n(2D Graph)', 'View 3\n(3D EGNN)',
                  'TriBioNode\n(Fused)', 'Ground\nTruth']
    bar_vals   = [mol['y_v1'], mol['y_v2'], mol['y_v3'], mol['y_fused'], mol['y_true']]
    bar_colors = ['#3498db', '#27ae60', '#e67e22', '#2c3e50', '#7f8c8d']
    # highlight fused
    bar_ec     = ['none','none','none', mol['color'], 'none']
    bar_ew     = [0,0,0,2.5,0]

    x = np.arange(len(bar_labels))
    bars2 = ax_out.bar(x, bar_vals, color=bar_colors, edgecolor=bar_ec,
                       linewidth=bar_ew, width=0.62, zorder=3)

    ax_out.set_title('Model Output Comparison & Synergy Outcome',
                     fontsize=SECT_FS-2, fontweight='bold', pad=10)
    ax_out.set_ylabel('Solubility (log mol/L)', fontsize=LABEL_FS)
    ax_out.set_xticks(x)
    ax_out.set_xticklabels(bar_labels, fontsize=TICK_FS)
    ax_out.tick_params(axis='y', labelsize=TICK_FS)
    ax_out.spines['top'].set_visible(False)
    ax_out.spines['right'].set_visible(False)
    ax_out.grid(axis='y', alpha=0.3, zorder=0)

    # Value annotations on bars
    all_vals = bar_vals
    y_min, y_max = min(all_vals)-0.4, max(all_vals)+0.9
    ax_out.set_ylim(y_min, y_max)

    for i, (bar, v) in enumerate(zip(bars2, bar_vals)):
        offset = 0.08
        va = 'bottom' if v >= 0 else 'top'
        y_pos = v + offset if v >= 0 else v - offset
        fw = 'bold' if i == 3 else 'normal'
        fc = mol['color'] if i == 3 else '#1a1a2e'
        ax_out.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{v:.2f}', ha='center', va=va,
                    fontsize=VAL_FS, fontweight=fw, color=fc)

    # Draw arrow from ground truth to fused
    gt_y = mol['y_true']
    fused_y = mol['y_fused']
    ax_out.annotate('', xy=(3, fused_y), xytext=(4, gt_y),
                    arrowprops=dict(arrowstyle='<->', color=mol['color'],
                                    lw=1.5))
    ae_val = abs(fused_y - gt_y)
    mid_y = (fused_y + gt_y) / 2
    ax_out.text(3.52, mid_y, f'|Δ|={ae_val:.2f}',
                ha='center', va='center', fontsize=TICK_FS,
                color=mol['color'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec=mol['color'], lw=1))

out = '/home/farhankabirsifat/Desktop/ADD_thesis/figures/view_fusion_dynamic_gating_synergy.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
plt.close(fig)
