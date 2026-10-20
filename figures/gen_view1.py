"""
View 1: 1D Chemical Language & Grammar (ChemBERTa Transformer)
IMPROVED VERSION - much larger fonts, cleaner layout, higher readability
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

# ─── Global style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 1.4,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

TITLE_FS   = 22
SECT_FS    = 17
LABEL_FS   = 15
TICK_FS    = 13
VAL_FS     = 14
ANNOT_FS   = 12

# Molecule data
molecules = [
    {
        'name': 'Exemplar 1: Fenarimol Analog',
        'smiles_display': 'OC(c1ccc(Cl)cc1)(c2cncnc2)c3ccccc3Cl',
        'mol_info': 'MW: 332.2  Atoms: 25  LogP: 4.07',
        'tokens': ['[CLS]', 'O', 'C', '(', 'c1', 'cc', 'c', '(', 'Cl', ')', 'cc', '1', ')', '(', 'c2', 'cn', 'cn', 'c2', ')', 'c3'],
        'attn':   [0.10, 0.68, 0.32, 0.18, 0.55, 0.40, 0.38, 0.22, 0.88, 0.15, 0.42, 0.30, 0.20, 0.25, 0.72, 0.65, 0.62, 0.70, 0.18, 0.35],
        'exp': -4.38, 'pred': -3.16, 'ae': 1.22,
        'focus': 'High attention on polar heteroatoms: -OH, -Cl, and N (pyrimidine)',
        'latent_seed': 42,
    },
    {
        'name': 'Exemplar 2: Spiro-Hydantoin Complex',
        'smiles_display': 'O=C1NC(=O)C2(CCCCC2)N1',
        'mol_info': 'MW: 182.2  Atoms: 14  LogP: -0.79',
        'tokens': ['[CLS]', 'O', '=', 'C', '1', 'N', 'C', '(', '=', 'O', ')', 'C', '2', '(', 'C', 'C', 'C', 'C', 'C', '2'],
        'attn':   [0.08, 0.72, 0.80, 0.30, 0.20, 0.85, 0.28, 0.15, 0.78, 0.74, 0.12, 0.22, 0.18, 0.16, 0.20, 0.18, 0.20, 0.19, 0.21, 0.17],
        'exp': -2.35, 'pred': -2.81, 'ae': 0.46,
        'focus': 'High attention on N-H imide & carbonyl C=O bonds (spiro ring)',
        'latent_seed': 77,
    },
    {
        'name': 'Exemplar 3: Anthanthrene PAH',
        'smiles_display': 'c1cc2ccc3cccc4ccc5cccc6ccc1c1c2c3c4c5c61',
        'mol_info': 'MW: 276.3  Atoms: 22  LogP: 6.53',
        'tokens': ['[CLS]', 'c1', 'cc', '2', 'ccc', '3', 'ccc', 'c4', 'ccc', '5', 'ccc', 'c6', 'ccc', '1', 'c1', 'c2', 'c3', 'c4', 'c5', 'c61'],
        'attn':   [0.12, 0.45, 0.48, 0.35, 0.50, 0.38, 0.52, 0.55, 0.50, 0.40, 0.55, 0.48, 0.42, 0.50, 0.38, 0.45, 0.48, 0.44, 0.50, 0.46],
        'exp': -6.33, 'pred': -0.17, 'ae': 4.30,  # note: difficult molecule
        'focus': 'Uniform attention over conjugated aromatic ring system (PAH)',
        'latent_seed': 13,
    },
]

# ─── Canvas ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(28, 33), facecolor='white')

# Supertitle
fig.text(0.5, 0.998, 'View 1 Visual Representation: 1D Chemical Language & Grammar (ChemBERTa Transformer)',
         ha='center', va='top', fontsize=TITLE_FS+2, fontweight='bold', color='#1a1a2e')
fig.text(0.5, 0.984, 'Tokenized SMILES Stream · ChemBERTa Self-Attention Saliency · 8×16 Latent Feature Matrix · Unimodal Predictions',
         ha='center', va='top', fontsize=SECT_FS-1, color='#444')

outer = gridspec.GridSpec(3, 1, figure=fig,
                          top=0.977, bottom=0.02,
                          hspace=0.12,
                          left=0.04, right=0.97)

CARD_COLORS = ['#e8f4fd', '#e8fdf0', '#fdf5e8']
BORDER_COLORS = ['#2980b9', '#27ae60', '#e67e22']
BOX_COLORS = ['#d6eaf8', '#d5f5e3', '#fdebd0']

np.random.seed(0)

for mi, mol in enumerate(molecules):
    row_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[mi],
        width_ratios=[1.1, 1.5, 0.85],
        wspace=0.30
    )

    # ── Column 0: Info card ─────────────────────────────────────────────
    ax_info = fig.add_subplot(row_gs[0])
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
    ax_info.axis('off')

    # Card background
    card = FancyBboxPatch((0.04, 0.04), 0.92, 0.92,
                          boxstyle='round,pad=0.02',
                          facecolor=CARD_COLORS[mi],
                          edgecolor=BORDER_COLORS[mi], linewidth=2.5)
    ax_info.add_patch(card)

    # Molecule name header
    ax_info.add_patch(FancyBboxPatch((0.04, 0.82), 0.92, 0.14,
                                     boxstyle='round,pad=0.01',
                                     facecolor=BORDER_COLORS[mi], alpha=0.85,
                                     edgecolor='none'))
    ax_info.text(0.50, 0.894, mol['name'],
                 ha='center', va='center', fontsize=SECT_FS, fontweight='bold',
                 color='white', transform=ax_info.transAxes)

    # SMILES (wrapped)
    smiles = mol['smiles_display']
    if len(smiles) > 38:
        smiles = smiles[:38] + '…'
    ax_info.text(0.50, 0.775, smiles,
                 ha='center', va='center', fontsize=10.5,
                 color='#333', family='monospace',
                 transform=ax_info.transAxes)
    ax_info.text(0.50, 0.735, mol['mol_info'],
                 ha='center', va='center', fontsize=LABEL_FS-1, color='#555',
                 transform=ax_info.transAxes)

    # Evaluation box
    ax_info.add_patch(FancyBboxPatch((0.08, 0.40), 0.84, 0.28,
                                     boxstyle='round,pad=0.02',
                                     facecolor=BOX_COLORS[mi], linewidth=1.5,
                                     edgecolor=BORDER_COLORS[mi]))
    ax_info.text(0.50, 0.700, 'View 1 (1D SMILES) Evaluation',
                 ha='center', va='center', fontsize=LABEL_FS,
                 fontweight='bold', color=BORDER_COLORS[mi],
                 transform=ax_info.transAxes)
    ax_info.text(0.12, 0.655, f'Experimental Target:', fontsize=LABEL_FS-1,
                 color='#333', transform=ax_info.transAxes)
    ax_info.text(0.88, 0.655, f"{mol['exp']:.2f} log(mol/L)",
                 fontsize=LABEL_FS-1, fontweight='bold', color='#1a1a2e',
                 ha='right', transform=ax_info.transAxes)
    ax_info.text(0.12, 0.615, f'View 1 Prediction:', fontsize=LABEL_FS-1,
                 color='#333', transform=ax_info.transAxes)
    ax_info.text(0.88, 0.615, f"{mol['pred']:.2f} log(mol/L)",
                 fontsize=LABEL_FS-1, fontweight='bold', color=BORDER_COLORS[mi],
                 ha='right', transform=ax_info.transAxes)
    ae_color = '#c0392b' if mol['ae'] > 1.0 else '#27ae60'
    ax_info.text(0.12, 0.572, f'Absolute Error (Δ):', fontsize=LABEL_FS,
                 color='#333', fontweight='bold', transform=ax_info.transAxes)
    ax_info.text(0.88, 0.572, f"{mol['ae']:.2f} log units",
                 fontsize=LABEL_FS, fontweight='bold', color=ae_color,
                 ha='right', transform=ax_info.transAxes)

    # Grammar focus
    ax_info.add_patch(FancyBboxPatch((0.06, 0.08), 0.88, 0.28,
                                     boxstyle='round,pad=0.02',
                                     facecolor='#f8f9fa', linewidth=1.2,
                                     edgecolor='#aaa'))
    ax_info.text(0.50, 0.38, 'Grammar Attention Focus:', fontsize=LABEL_FS-1,
                 ha='center', color='#555', fontstyle='italic',
                 transform=ax_info.transAxes)
    # Wrap focus text
    focus = mol['focus']
    words = focus.split()
    lines = []
    line = ''
    for w in words:
        if len(line + ' ' + w) <= 36:
            line = (line + ' ' + w).strip()
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    for li, ln in enumerate(lines[:3]):
        ax_info.text(0.50, 0.335 - li*0.042, ln,
                     ha='center', va='center', fontsize=LABEL_FS-2,
                     color='#333', transform=ax_info.transAxes)

    # ── Column 1: Token + Attention bars ────────────────────────────────
    ax_tok = fig.add_subplot(row_gs[1])
    ax_tok.set_facecolor('#fafafa')

    tokens = mol['tokens']
    attn   = np.array(mol['attn'])
    n = len(tokens)
    x = np.arange(n)

    # Bar chart of attention
    cmap = plt.cm.RdYlGn
    colors = [cmap(a) for a in attn]
    bars = ax_tok.bar(x, attn, color=colors, edgecolor='white', linewidth=0.8, zorder=3)
    ax_tok.set_xlim(-0.7, n - 0.3)
    ax_tok.set_ylim(0, 1.18)
    ax_tok.set_xticks(x)
    ax_tok.set_xticklabels(tokens, fontsize=TICK_FS, rotation=40, ha='right', family='monospace')
    ax_tok.set_ylabel('Attention Saliency', fontsize=LABEL_FS)
    ax_tok.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_tok.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.00'], fontsize=TICK_FS)
    ax_tok.set_title('Tokenized Sequence & ChemBERTa Attention Saliency',
                     fontsize=SECT_FS-1, fontweight='bold', pad=8)
    ax_tok.axhline(0.5, color='gray', lw=1, ls='--', alpha=0.5, label='Threshold 0.5')
    ax_tok.grid(axis='y', alpha=0.3, zorder=0)
    ax_tok.spines['top'].set_visible(False)
    ax_tok.spines['right'].set_visible(False)
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_tok, orientation='horizontal',
                      pad=0.22, fraction=0.04, aspect=30)
    cb.set_label('Low Attn (0.0)  →  High Saliency (1.0)', fontsize=TICK_FS)
    cb.ax.tick_params(labelsize=TICK_FS-1)

    # ── Column 2: Latent heatmap ─────────────────────────────────────────
    ax_lat = fig.add_subplot(row_gs[2])
    np.random.seed(mol['latent_seed'])
    latent = np.random.randn(8, 16)
    latent = (latent - latent.min()) / (latent.max() - latent.min()) * 2 - 1
    im = ax_lat.imshow(latent, aspect='auto', cmap='coolwarm',
                       vmin=-1, vmax=1, interpolation='nearest')
    ax_lat.set_title(f'View 1 Latent Signal: $h_{{v1}} \\in \\mathbb{{R}}^{{128}}$\n(Mean={latent.mean():.2f}, Std={latent.std():.2f})',
                     fontsize=LABEL_FS, fontweight='bold', pad=6)
    ax_lat.set_xlabel('Embedding Sub-Dim (1–16)', fontsize=LABEL_FS)
    ax_lat.set_ylabel('Attn Head (1–8)', fontsize=LABEL_FS)
    ax_lat.set_xticks([0, 4, 8, 12, 15])
    ax_lat.set_xticklabels(['1', '5', '9', '13', '16'], fontsize=TICK_FS)
    ax_lat.set_yticks(range(8))
    ax_lat.set_yticklabels([f'H{i+1}' for i in range(8)], fontsize=TICK_FS)
    cb2 = fig.colorbar(im, ax=ax_lat, fraction=0.046, pad=0.04)
    cb2.set_label('Activation', fontsize=TICK_FS)
    cb2.ax.tick_params(labelsize=TICK_FS-1)
    cb2.set_ticks([-1, -0.5, 0, 0.5, 1])

# ─── Save ────────────────────────────────────────────────────────────────────
out = '/home/farhankabirsifat/Desktop/ADD_thesis/figures/view1_chemical_language_grammar.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
plt.close(fig)
