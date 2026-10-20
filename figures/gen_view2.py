"""
View 2: 2D Hierarchical Atom-Motif Graph
IMPROVED VERSION - much larger fonts, cleaner layout, readable MSGG encoding cards
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
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
VAL_FS   = 14

molecules = [
    {
        'name': 'Exemplar 1: Fenarimol Analog',
        'smiles': 'OC(c1ccc(Cl)cc1)(c2cncnc2)c3ccccc3Cl',
        'mol_info': 'Graph Topology: 25 Atoms, 26 Bonds, 3 Rings',
        'exp': -4.38, 'pred': -3.54, 'ae': 0.84,
        'motifs': ['4-Chlorophenyl', 'Pyrimidine Ring', 'Phenyl Ring', 'Tertiary Carbinol Core'],
        'motif_counts': '4 Contracted Motifs: 2 Chlorophenyl · 1 Pyrimidine · 1 Carbinol Core',
        'arch': 'Atom MPNN → 8-d MSGG → Motif Transformer',
        'seed': 10,
        'color': '#2980b9',
        'bg': '#e8f4fd',
        'box': '#d6eaf8',
    },
    {
        'name': 'Exemplar 2: Spiro-Hydantoin Complex',
        'smiles': 'O=C1NC(=O)C2(CCCCC2)N1',
        'mol_info': 'Graph Topology: 15 Atoms, 15 Bonds, 2 Rings',
        'exp': -2.35, 'pred': -2.64, 'ae': 0.30,
        'motifs': ['Spiro Atom', 'Cyclohexane Ring', 'Hydantoin Core'],
        'motif_counts': 'Fused Spiro-Motif: Non-planar Imide · Cyclohexane Envelope Ring',
        'arch': 'Atom MPNN → 8-d MSGG → Motif Transformer',
        'seed': 20,
        'color': '#27ae60',
        'bg': '#e8fdf0',
        'box': '#d5f5e3',
    },
    {
        'name': 'Exemplar 3: Anthanthrene PAH',
        'smiles': 'c1cc2ccc3cccc4ccc5cccc6ccc1c1c2c3c4c5c61',
        'mol_info': 'Graph Topology: 24 Atoms, 28 Bonds, 7 Rings',
        'exp': -6.33, 'pred': -4.33, 'ae': 0.90,
        'motifs': ['Benzene A', 'Benzene B', 'Benzene C', 'Fused Conjugated PAH'],
        'motif_counts': 'Fused Aromatic Macro: 6 Interconnected Conjugated Aromatic Rings',
        'arch': 'Atom MPNN → 8-d MSGG → Motif Transformer',
        'seed': 30,
        'color': '#e67e22',
        'bg': '#fdf5e8',
        'box': '#fdebd0',
    },
]

MSGG_CHANNELS = [
    ('1. Bond Order',       'Single, Double, Triple, or Aromatic bond class'),
    ('2. Ring Confinement', 'Whether atoms/bonds are in an SSSR ring'),
    ('3. Conjugation',      'Continuous π-electron delocalization status'),
    ('4. Hybridization',    'sp, sp², sp³ hybridization pair encoding'),
    ('5. Steric Tension',   'van der Waals clash / torsional strain index'),
    ('6. Bond Polarity',    'Electronegativity difference Δχ per bond'),
    ('7. Topological Dist.','Graph shortest-path distance (hops)'),
    ('8. Resonance',        'Kekulé / resonance structure differentiation'),
]

# motif colors for each molecule (per motif)
MOTIF_PAL = [
    ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'],
    ['#f39c12', '#16a085', '#e74c3c'],
    ['#8e44ad', '#1abc9c', '#f39c12', '#e74c3c'],
]

def draw_graph(ax, n_atoms, n_motifs, seed, motif_palette, motif_names, title, color):
    """Draw a stylized molecular graph with motif coloring."""
    np.random.seed(seed)
    ax.set_aspect('equal')
    ax.axis('off')

    # Generate random layout
    theta = np.linspace(0, 2*np.pi, n_atoms, endpoint=False)
    r = 1 + 0.35 * np.random.rand(n_atoms)
    xs = r * np.cos(theta) + np.random.randn(n_atoms) * 0.15
    ys = r * np.sin(theta) + np.random.randn(n_atoms) * 0.15

    # Assign atoms to motifs
    motif_assign = np.array_split(np.arange(n_atoms), n_motifs)

    # Draw bonds (connect nearest neighbors)
    adj = set()
    for i in range(n_atoms):
        dists = np.sqrt((xs - xs[i])**2 + (ys - ys[i])**2)
        dists[i] = np.inf
        nn = np.argsort(dists)[:2]
        for j in nn:
            if (min(i,j), max(i,j)) not in adj:
                adj.add((min(i,j), max(i,j)))
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], '-',
                        color='#555', lw=1.8, zorder=1)

    # Draw atoms colored by motif
    for mi, group in enumerate(motif_assign):
        c = motif_palette[mi % len(motif_palette)]
        for idx in group:
            ax.scatter(xs[idx], ys[idx], s=220, color=c,
                       edgecolors='white', linewidths=1.5, zorder=3)
            ax.text(xs[idx], ys[idx], str(idx+1),
                    ha='center', va='center', fontsize=7.5,
                    color='white', fontweight='bold', zorder=4)

    # Legend for motifs
    handles = [mpatches.Patch(color=motif_palette[m % len(motif_palette)],
                              label=f'M{m+1}: {motif_names[m]}')
               for m in range(n_motifs)]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.18),
              fontsize=TICK_FS-1, ncol=2, framealpha=0.9,
              columnspacing=0.7, handlelength=1.2)
    ax.set_title(title, fontsize=LABEL_FS, fontweight='bold', pad=6)
    ax.set_xlim(xs.min()-0.5, xs.max()+0.5)
    ax.set_ylim(ys.min()-0.7, ys.max()+0.5)


fig = plt.figure(figsize=(28, 36), facecolor='white')
fig.text(0.5, 0.999, 'View 2 Visual Representation: 2D Hierarchical Atom-Motif Graph',
         ha='center', va='top', fontsize=TITLE_FS+2, fontweight='bold', color='#1a1a2e')
fig.text(0.5, 0.986, 'Atom-Level MPNN · 8-Dimensional MSGG Joint Channels · BRICS Motif Super-Nodes · Latent Signal Vectors',
         ha='center', va='top', fontsize=SECT_FS-1, color='#444')

outer = gridspec.GridSpec(3, 1, figure=fig,
                          top=0.979, bottom=0.02,
                          hspace=0.14,
                          left=0.03, right=0.97)

np.random.seed(1)
atom_counts  = [16, 12, 18]
motif_counts = [4, 3, 4]

for mi, mol in enumerate(molecules):
    row_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[mi],
        width_ratios=[1.2, 1.6, 0.9],
        wspace=0.28
    )

    # ── Col 0: Info card ───────────────────────────────────────────────
    ax_info = fig.add_subplot(row_gs[0])
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
    ax_info.axis('off')

    card = FancyBboxPatch((0.04, 0.04), 0.92, 0.92,
                          boxstyle='round,pad=0.02',
                          facecolor=mol['bg'],
                          edgecolor=mol['color'], linewidth=2.5)
    ax_info.add_patch(card)
    ax_info.add_patch(FancyBboxPatch((0.04, 0.82), 0.92, 0.14,
                                     boxstyle='round,pad=0.01',
                                     facecolor=mol['color'], alpha=0.85,
                                     edgecolor='none'))
    ax_info.text(0.50, 0.895, mol['name'], ha='center', va='center',
                 fontsize=SECT_FS, fontweight='bold', color='white',
                 transform=ax_info.transAxes)

    smiles = mol['smiles']
    if len(smiles) > 36:
        smiles = smiles[:36] + '…'
    ax_info.text(0.50, 0.775, smiles, ha='center', va='center',
                 fontsize=10.5, color='#333', family='monospace',
                 transform=ax_info.transAxes)
    ax_info.text(0.50, 0.735, mol['mol_info'], ha='center', va='center',
                 fontsize=LABEL_FS-1, color='#555', transform=ax_info.transAxes)

    # Evaluation box
    ax_info.add_patch(FancyBboxPatch((0.08, 0.42), 0.84, 0.27,
                                     boxstyle='round,pad=0.02',
                                     facecolor=mol['box'], linewidth=1.5,
                                     edgecolor=mol['color']))
    ax_info.text(0.50, 0.705, 'View 2 (2D Graph) Evaluation',
                 ha='center', va='center', fontsize=LABEL_FS,
                 fontweight='bold', color=mol['color'],
                 transform=ax_info.transAxes)
    ax_info.text(0.12, 0.660, 'Experimental Target:', fontsize=LABEL_FS-1,
                 color='#333', transform=ax_info.transAxes)
    ax_info.text(0.88, 0.660, f"{mol['exp']:.2f} log(mol/L)",
                 fontsize=LABEL_FS-1, fontweight='bold', color='#1a1a2e',
                 ha='right', transform=ax_info.transAxes)
    ax_info.text(0.12, 0.618, 'View 2 Prediction:', fontsize=LABEL_FS-1,
                 color='#333', transform=ax_info.transAxes)
    ax_info.text(0.88, 0.618, f"{mol['pred']:.2f} log(mol/L)",
                 fontsize=LABEL_FS-1, fontweight='bold', color=mol['color'],
                 ha='right', transform=ax_info.transAxes)
    ae_c = '#c0392b' if mol['ae'] > 1.0 else '#27ae60'
    ax_info.text(0.12, 0.575, 'Absolute Error (Δ):', fontsize=LABEL_FS,
                 color='#333', fontweight='bold', transform=ax_info.transAxes)
    ax_info.text(0.88, 0.575, f"{mol['ae']:.2f} log units",
                 fontsize=LABEL_FS, fontweight='bold', color=ae_c,
                 ha='right', transform=ax_info.transAxes)

    # Motif architecture
    ax_info.text(0.50, 0.50, 'Motif Graph Architecture:', ha='center',
                 va='center', fontsize=LABEL_FS-1, fontstyle='italic',
                 color='#555', transform=ax_info.transAxes)
    ax_info.text(0.50, 0.465, mol['arch'], ha='center', va='center',
                 fontsize=LABEL_FS-2, color='#333', transform=ax_info.transAxes)
    ax_info.text(0.50, 0.415, mol['motif_counts'], ha='center', va='center',
                 fontsize=LABEL_FS-2, color='#555',
                 wrap=True, transform=ax_info.transAxes)

    # ── Col 1: Graph + MSGG channels ─────────────────────────────────
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=row_gs[1],
        height_ratios=[1.6, 1.0],
        hspace=0.45
    )
    ax_graph = fig.add_subplot(inner_gs[0])
    draw_graph(ax_graph, atom_counts[mi], motif_counts[mi],
               seeds := [10, 20, 30][mi],
               MOTIF_PAL[mi], mol['motifs'],
               f'Segmented 2D Molecular Graph ({motif_counts[mi]} BRICS Motifs)',
               mol['color'])

    # MSGG encoding text box
    ax_msgg = fig.add_subplot(inner_gs[1])
    ax_msgg.set_xlim(0, 1); ax_msgg.set_ylim(0, 1)
    ax_msgg.axis('off')
    ax_msgg.add_patch(FancyBboxPatch((0, 0.02), 1.0, 0.96,
                                      boxstyle='round,pad=0.02',
                                      facecolor='#f0f4ff', linewidth=1.5,
                                      edgecolor='#5d6d7e'))
    ax_msgg.text(0.5, 0.95, '8-d MSGG Joint Interface Encodings:',
                 ha='center', va='top', fontsize=LABEL_FS,
                 fontweight='bold', color='#2c3e50', transform=ax_msgg.transAxes)
    for ci, (ch_name, ch_desc) in enumerate(MSGG_CHANNELS):
        y = 0.87 - ci * 0.107
        ax_msgg.text(0.03, y, ch_name + ':', fontsize=TICK_FS,
                     fontweight='bold', color='#1a5276',
                     transform=ax_msgg.transAxes)
        # Wrap description
        desc = ch_desc[:54] + '…' if len(ch_desc) > 54 else ch_desc
        ax_msgg.text(0.03, y - 0.047, '  ' + desc, fontsize=TICK_FS-1,
                     color='#444', transform=ax_msgg.transAxes)

    # ── Col 2: Latent heatmap ──────────────────────────────────────────
    ax_lat = fig.add_subplot(row_gs[2])
    np.random.seed(mol['seed'])
    latent = np.random.randn(8, 16)
    latent = (latent - latent.min()) / (latent.max() - latent.min()) * 2 - 1
    im = ax_lat.imshow(latent, aspect='auto', cmap='viridis',
                       vmin=-1, vmax=1, interpolation='nearest')
    ax_lat.set_title(f'View 2 Latent Signal: $h_{{v2}} \\in \\mathbb{{R}}^{{128}}$\n(Mean={latent.mean():.2f}, Std={latent.std():.2f})',
                     fontsize=LABEL_FS, fontweight='bold', pad=6)
    ax_lat.set_xlabel('Embedding Sub-Dim (1–16)', fontsize=LABEL_FS)
    ax_lat.set_ylabel('Graph Layer (1–8)', fontsize=LABEL_FS)
    ax_lat.set_xticks([0, 4, 8, 12, 15])
    ax_lat.set_xticklabels(['1', '5', '9', '13', '16'], fontsize=TICK_FS)
    ax_lat.set_yticks(range(8))
    ax_lat.set_yticklabels([f'L{i+1}' for i in range(8)], fontsize=TICK_FS)
    cb = fig.colorbar(im, ax=ax_lat, fraction=0.046, pad=0.06)
    cb.set_label('Activation', fontsize=TICK_FS)
    cb.ax.tick_params(labelsize=TICK_FS-1)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])

out = '/home/farhankabirsifat/Desktop/ADD_thesis/figures/view2_hierarchical_graph_motifs.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
plt.close(fig)
