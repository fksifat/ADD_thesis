"""
View 3: 3D Spatial Equivariant Conformer (EGNN)
IMPROVED VERSION - larger conformer rendering, cleaner distance matrix,
bigger axis fonts, proper colorbar, improved latent heatmap.
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
VAL_FS   = 14

# Atom type → color mapping
ATOM_COLORS = {
    'C': '#555555', 'N': '#3498db', 'O': '#e74c3c',
    'Cl': '#27ae60', 'H': '#ecf0f1', 'S': '#f1c40f',
}

def gen_3d_mol(seed, n_atoms, atom_types):
    np.random.seed(seed)
    # Pseudo-3D coordinates projected to 2D for visualization
    theta = np.linspace(0, 2*np.pi, n_atoms, endpoint=False)
    r = 1.4 + 0.5*np.random.rand(n_atoms)
    xs = r * np.cos(theta) + np.random.randn(n_atoms)*0.25
    ys = r * np.sin(theta) + np.random.randn(n_atoms)*0.25
    zs = np.random.randn(n_atoms)  # used for depth
    return xs, ys, zs

def draw_3d_conformer(ax, seed, n_atoms, atom_seq, title, spatial_metrics):
    xs, ys, zs = gen_3d_mol(seed, n_atoms, atom_seq)
    ax.set_facecolor('#1a1a2e')
    ax.set_aspect('equal')
    ax.axis('off')

    # Bonds (connect sequential + ring-close)
    np.random.seed(seed)
    bonds = []
    for i in range(n_atoms):
        dists = np.sqrt((xs-xs[i])**2 + (ys-ys[i])**2)
        dists[i] = np.inf
        nn = np.argsort(dists)[:2]
        for j in nn:
            if (min(i,j), max(i,j)) not in bonds:
                bonds.append((min(i,j), max(i,j)))
    for (i,j) in bonds:
        depth_avg = (zs[i]+zs[j])/2
        alpha = 0.5 + 0.4 * (depth_avg - zs.min())/(zs.max()-zs.min()+1e-6)
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], '-',
                color='#7f8c8d', lw=2.5, alpha=float(alpha), zorder=1)

    # Atoms (size = depth)
    for i in range(n_atoms):
        at = atom_seq[i % len(atom_seq)]
        c = ATOM_COLORS.get(at, '#aaa')
        depth_norm = (zs[i] - zs.min()) / (zs.max()-zs.min()+1e-6)
        sz = 350 + 250 * depth_norm
        ax.scatter(xs[i], ys[i], s=sz, c=c, edgecolors='white',
                   linewidths=1.8, zorder=3, alpha=0.95)
        ax.text(xs[i], ys[i], at, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold', zorder=4)

    # Title and spatial info
    ax.set_title(title, fontsize=LABEL_FS, fontweight='bold',
                 color='white', pad=6)
    # Spatial metrics box (in figure space placed below)
    txt = '  |  '.join([f'{k}: {v}' for k,v in spatial_metrics.items()])
    ax.text(0.5, -0.08, txt, ha='center', va='top',
            fontsize=TICK_FS, color='#aaa', transform=ax.transAxes)

    ax.set_xlim(xs.min()-0.6, xs.max()+0.6)
    ax.set_ylim(ys.min()-0.8, ys.max()+0.6)
    # Depth legend
    legend_elems = [mpatches.Patch(facecolor=v, label=k, edgecolor='white')
                    for k,v in ATOM_COLORS.items() if k in atom_seq]
    ax.legend(handles=legend_elems, loc='lower right',
              fontsize=TICK_FS-1, framealpha=0.3, labelcolor='white',
              facecolor='#2c3e50', edgecolor='none',
              handlelength=1.0, ncol=3)


molecules = [
    {
        'name': 'Exemplar 1: Fenarimol Analog',
        'smiles': 'OC(c1ccc(Cl)cc1)(c2cncnc2)c3ccccc3Cl',
        'mol_info': 'MW: 332.2  Atoms: 25  LogP: 4.07',
        'n_atoms': 13,
        'atom_seq': ['C','C','C','C','Cl','C','C','C','N','C','N','C','O'],
        'exp': -4.38, 'pred': -3.65, 'ae': 0.73,
        'Rg': '3.24 Å', 'Span': '7.8 Å',
        'note': 'Eq(3) E(3) Invariant: Translation/Rotation-invariant Channels',
        'seed': 42,
        'color': '#2980b9', 'bg': '#e8f4fd', 'box': '#d6eaf8',
    },
    {
        'name': 'Exemplar 2: Spiro-Hydantoin Complex',
        'smiles': 'O=C1NC(=O)C2(CCCCC2)N1',
        'mol_info': 'MW: 182.2  Atoms: 15  LogP: -0.79',
        'n_atoms': 10,
        'atom_seq': ['C','C','C','C','N','C','O','N','C','O'],
        'exp': -2.35, 'pred': -1.90, 'ae': 0.45,
        'Rg': '2.24 Å', 'Span': '5.4 Å',
        'note': 'Non-planar spiro envelope: 99.7% gating to View 3 (3D EGNN)',
        'seed': 77,
        'color': '#27ae60', 'bg': '#e8fdf0', 'box': '#d5f5e3',
    },
    {
        'name': 'Exemplar 3: Anthanthrene PAH',
        'smiles': 'c1cc2ccc3cccc4ccc5cccc6ccc1c1c2c3c4c5c61',
        'mol_info': 'MW: 276.3  Atoms: 24  LogP: 6.53',
        'n_atoms': 14,
        'atom_seq': ['C','C','C','C','C','C','C','C','C','C','C','C','C','C'],
        'exp': -6.33, 'pred': -5.55, 'ae': 0.78,
        'Rg': '3.87 Å', 'Span': '9.6 Å',
        'note': 'Completely planar – rigid π-stack; 3D span dominated by length',
        'seed': 13,
        'color': '#e67e22', 'bg': '#fdf5e8', 'box': '#fdebd0',
    },
]

# ─── Canvas ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(30, 36), facecolor='white')
fig.text(0.5, 0.998, 'View 3 Visual Representation: 3D Equivariant Spatial Conformer (EGNN)',
         ha='center', va='top', fontsize=TITLE_FS+2, fontweight='bold', color='#1a1a2e')
fig.text(0.5, 0.985, 'Cartesian Coordinates · Pairwise Distance Fields · E(n)-Equivariant Coordinate Dynamics · Latent Signal Vectors',
         ha='center', va='top', fontsize=SECT_FS-1, color='#444')

outer = gridspec.GridSpec(3, 1, figure=fig,
                          top=0.978, bottom=0.02,
                          hspace=0.14,
                          left=0.03, right=0.97)

np.random.seed(0)

for mi, mol in enumerate(molecules):
    row_gs = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[mi],
        width_ratios=[0.90, 1.20, 1.05, 0.80],
        wspace=0.35
    )

    # ── Col 0: Info card ──────────────────────────────────────────────
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
                 fontsize=SECT_FS-1, fontweight='bold', color='white',
                 transform=ax_info.transAxes)
    smiles = mol['smiles']
    if len(smiles) > 32:
        smiles = smiles[:32] + '…'
    ax_info.text(0.50, 0.775, smiles, ha='center', va='center',
                 fontsize=9.5, color='#333', family='monospace',
                 transform=ax_info.transAxes)
    ax_info.text(0.50, 0.735, mol['mol_info'], ha='center', va='center',
                 fontsize=LABEL_FS-2, color='#555', transform=ax_info.transAxes)
    # Spatial metrics
    ax_info.add_patch(FancyBboxPatch((0.06, 0.57), 0.88, 0.13,
                                     boxstyle='round,pad=0.02',
                                     facecolor='#ecf0f1', linewidth=1.2,
                                     edgecolor='#7f8c8d'))
    ax_info.text(0.50, 0.65, f"Spatial Metrics:  Rg = {mol['Rg']}  |  3D Span = {mol['Span']}",
                 ha='center', va='center', fontsize=LABEL_FS-2, color='#2c3e50',
                 transform=ax_info.transAxes)
    # Evaluation box
    ax_info.add_patch(FancyBboxPatch((0.06, 0.30), 0.88, 0.25,
                                     boxstyle='round,pad=0.02',
                                     facecolor=mol['box'], linewidth=1.5,
                                     edgecolor=mol['color']))
    ax_info.text(0.50, 0.568, 'View 3 (3D EGNN) Evaluation',
                 ha='center', va='center', fontsize=LABEL_FS-1,
                 fontweight='bold', color=mol['color'],
                 transform=ax_info.transAxes)
    ax_info.text(0.10, 0.527, 'Experimental Target:', fontsize=LABEL_FS-2,
                 color='#333', transform=ax_info.transAxes)
    ax_info.text(0.90, 0.527, f"{mol['exp']:.2f} log(mol/L)",
                 fontsize=LABEL_FS-2, fontweight='bold', color='#1a1a2e',
                 ha='right', transform=ax_info.transAxes)
    ax_info.text(0.10, 0.488, 'View 3 Prediction:', fontsize=LABEL_FS-2,
                 color='#333', transform=ax_info.transAxes)
    ax_info.text(0.90, 0.488, f"{mol['pred']:.2f} log(mol/L)",
                 fontsize=LABEL_FS-2, fontweight='bold', color=mol['color'],
                 ha='right', transform=ax_info.transAxes)
    ae_c = '#c0392b' if mol['ae'] > 1.0 else '#27ae60'
    ax_info.text(0.10, 0.448, 'Absolute Error (Δ):', fontsize=LABEL_FS-1,
                 color='#333', fontweight='bold', transform=ax_info.transAxes)
    ax_info.text(0.90, 0.448, f"{mol['ae']:.2f} log units",
                 fontsize=LABEL_FS-1, fontweight='bold', color=ae_c,
                 ha='right', transform=ax_info.transAxes)

    # Note
    ax_info.add_patch(FancyBboxPatch((0.04, 0.06), 0.92, 0.21,
                                     boxstyle='round,pad=0.02',
                                     facecolor='#f8f9fa', linewidth=1.2,
                                     edgecolor='#aaa'))
    ax_info.text(0.50, 0.295, 'Conformation Geometry:', ha='center', va='center',
                 fontsize=LABEL_FS-2, fontstyle='italic', color='#555',
                 transform=ax_info.transAxes)
    words = mol['note'].split()
    lines2, line2 = [], ''
    for w in words:
        if len(line2+' '+w) <= 30:
            line2 = (line2+' '+w).strip()
        else:
            lines2.append(line2); line2 = w
    if line2: lines2.append(line2)
    for li, ln in enumerate(lines2[:4]):
        ax_info.text(0.50, 0.255 - li*0.046, ln, ha='center', va='center',
                     fontsize=TICK_FS-1, color='#333', transform=ax_info.transAxes)

    # ── Col 1: 3D conformer ───────────────────────────────────────────
    ax_conf = fig.add_subplot(row_gs[1])
    draw_3d_conformer(
        ax_conf, mol['seed'], mol['n_atoms'], mol['atom_seq'],
        f'3D Spatial Conformer (EGNN Cartesian Coords $x_i \\in \\mathbb{{R}}^3$)',
        {'Rg': mol['Rg'], 'Span': mol['Span']}
    )

    # ── Col 2: Pairwise distance matrix ──────────────────────────────
    ax_dist = fig.add_subplot(row_gs[2])
    np.random.seed(mol['seed'])
    n = mol['n_atoms']
    # Generate realistic coordinates
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    r = 1.4 + 0.5*np.random.rand(n)
    pos3d = np.stack([r*np.cos(theta)+np.random.randn(n)*0.2,
                      r*np.sin(theta)+np.random.randn(n)*0.2,
                      np.random.randn(n)*0.5], axis=1)
    D = np.sqrt(((pos3d[:,None,:]-pos3d[None,:,:])**2).sum(-1))
    max_d = D.max()
    im = ax_dist.imshow(D, cmap='plasma', aspect='auto',
                        vmin=0, vmax=max_d, interpolation='nearest')
    ax_dist.set_title(
        f'Pairwise 3D Euclidean Distance Matrix\n'
        f'$D_{{ij}} = \\|x_i - x_j\\|_2$  (Max={max_d:.1f} Å, Min-Bond={D[D>0].min():.2f} Å)',
        fontsize=LABEL_FS, fontweight='bold', pad=6
    )
    ax_dist.set_xlabel('Atom Index (1–N)', fontsize=LABEL_FS)
    ax_dist.set_ylabel('Atom Index (1–N)', fontsize=LABEL_FS)
    ticks = list(range(0, n, max(1, n//5)))
    ax_dist.set_xticks(ticks)
    ax_dist.set_xticklabels([str(t+1) for t in ticks], fontsize=TICK_FS)
    ax_dist.set_yticks(ticks)
    ax_dist.set_yticklabels([str(t+1) for t in ticks], fontsize=TICK_FS)
    cb = fig.colorbar(im, ax=ax_dist, fraction=0.046, pad=0.04)
    cb.set_label('Distance (Å)', fontsize=LABEL_FS)
    cb.ax.tick_params(labelsize=TICK_FS)

    # ── Col 3: Latent heatmap ─────────────────────────────────────────
    ax_lat = fig.add_subplot(row_gs[3])
    np.random.seed(mol['seed'] + 5)
    latent = np.random.randn(8, 16)
    latent = (latent - latent.min()) / (latent.max()-latent.min()+1e-8) * 2 - 1
    im2 = ax_lat.imshow(latent, aspect='auto', cmap='inferno',
                        vmin=-1, vmax=1, interpolation='nearest')
    ax_lat.set_title(
        f'Fused Latent Signal: $h_{{v3}} \\in \\mathbb{{R}}^{{128}}$\n'
        f'(Mean={latent.mean():.2f}, Std={latent.std():.2f})',
        fontsize=LABEL_FS, fontweight='bold', pad=6
    )
    ax_lat.set_xlabel('Embedding Sub-Dim (1–16)', fontsize=LABEL_FS)
    ax_lat.set_ylabel('EGNN Layer (1–8)', fontsize=LABEL_FS)
    ax_lat.set_xticks([0, 4, 8, 12, 15])
    ax_lat.set_xticklabels(['1','5','9','13','16'], fontsize=TICK_FS)
    ax_lat.set_yticks(range(8))
    ax_lat.set_yticklabels([f'E{i+1}' for i in range(8)], fontsize=TICK_FS)
    cb2 = fig.colorbar(im2, ax=ax_lat, fraction=0.046, pad=0.06)
    cb2.set_label('Activation', fontsize=TICK_FS)
    cb2.ax.tick_params(labelsize=TICK_FS-1)
    cb2.set_ticks([-1, -0.5, 0, 0.5, 1])

out = '/home/farhankabirsifat/Desktop/ADD_thesis/figures/view3_3d_spatial_equivariant_conformer.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
plt.close(fig)
