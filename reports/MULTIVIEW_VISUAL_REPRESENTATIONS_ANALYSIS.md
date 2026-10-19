# Dedicated Visual Representations for View 1, View 2, View 3, and Multi-Modal Fusion

**Architecture**: TriBioNode v2  
**Reference Paper**: *Dual-View Learning Based on Images and Sequences for Molecular Property Prediction* (Zhang et al., IEEE JBHI 2024; [ISMol](https://github.com/Mrzhang1999/ISMol))  
**Source Outputs**: Delaney ESOL Scaffold Split (`outputs/view1_predictions_and_metrics.csv`, `outputs/view2_predictions_and_metrics.csv`, `outputs/view3_predictions_and_metrics.csv`, `outputs/tribionode_fused_predictions.csv`, `outputs/multiview_embeddings.npz`)

To address visual overlap and ensure every component has adequate space, each modality is rendered into its own dedicated, high-resolution visualization image:

---

## 1. View 1: 1D Chemical Language & Grammar (ChemBERTa Transformer)
📍 **File**: [`figures/view1_chemical_language_grammar.png`](file:///home/farhankabirsifat/Desktop/ADD_thesis/figures/view1_chemical_language_grammar.png)

![View 1 Chemical Language Grammar](file:///home/farhankabirsifat/Desktop/ADD_thesis/figures/view1_chemical_language_grammar.png)

### Key Content Displayed:
- **Molecule Info & View 1 Metric Badge**: Chemical name, IUPAC class, formula, molecular weight, calculated $\log P$, experimental true solubility ($y_{\text{true}}$), View 1 predicted solubility ($\hat{y}_1$), and absolute error ($|y - \hat{y}|$).
- **Tokenized Sequence Stream & Self-Attention Saliency**:
  - Full canonical SMILES string parsed into discrete chemical tokens (atoms, double bonds `=O`, branching parentheses `()`, ring closures `1-7`).
  - Saliency intensity color mapping: Heteroatoms ($\text{-OH}$, $\text{-Cl}$, $\text{N}$) and key branching points display strong self-attention ($>0.80$).
  - Self-attention intensity colorbar.
- **View 1 Latent Signal Vector ($\mathbf{h}_{v1} \in \mathbb{R}^{128}$)**:
  - Formatted as an $8 \times 16$ attention-head feature matrix, illustrating the 128-dimensional embedding vector extracted by the 1D language encoder.

---

## 2. View 2: 2D Hierarchical Atom-Motif Graph (Atom MPNN + MSGG + BRICS Transformer)
📍 **File**: [`figures/view2_hierarchical_graph_motifs.png`](file:///home/farhankabirsifat/Desktop/ADD_thesis/figures/view2_hierarchical_graph_motifs.png)

![View 2 Hierarchical Graph Motifs](file:///home/farhankabirsifat/Desktop/ADD_thesis/figures/view2_hierarchical_graph_motifs.png)

### Key Content Displayed:
- **Molecule Info & View 2 Metric Badge**: Graph topology parameters (heavy atom count, total bond count, ring count), View 2 predicted logS ($\hat{y}_2$), and error.
- **Segmented 2D Graph Structure**:
  - Large, crystal-clear 2D molecular layout rendered with RDKit.
  - Distinct pastel-colored highlighting for each decomposed BRICS functional motif (e.g. 4-chlorophenyl, pyrimidine ring, 2-chlorophenyl, carbinol core).
  - Clear identification of motif super-nodes contractable into the higher-level motif graph.
- **8-Dimensional MSGG Joint Channel Specifications**:
  - Comprehensive breakdown of the 8 multi-scale joint features encoded across motif interfaces:
    1. Joint Bond Order (Single, Double, Triple, Aromatic)
    2. Ring Confinement (Ring vs Exocyclic)
    3. Conjugation Status ($\pi$-electron delocalization)
    4. Hybridization Pair ($sp^2-sp^2$, $sp^2-sp^3$, $sp^3-sp^3$)
    5. Rotational Flexibility (Rotatable degrees of freedom)
    6. Steric Congestion (Local crowding around joint)
    7. Graph Topological Distance (Shortest path between motif centers)
    8. Heteroatom Polarity Differential
- **View 2 Latent Signal Vector ($\mathbf{h}_{v2} \in \mathbb{R}^{128}$)**:
  - Formatted as an $8 \times 16$ motif-head feature matrix, capturing the pooled hierarchical topological representation.

---

## 3. View 3: 3D Spatial Equivariant Conformer (EGNN + RBF Distance Field)
📍 **File**: [`figures/view3_3d_spatial_equivariant_conformer.png`](file:///home/farhankabirsifat/Desktop/ADD_thesis/figures/view3_3d_spatial_equivariant_conformer.png)

![View 3 3D Spatial Equivariant Conformer](file:///home/farhankabirsifat/Desktop/ADD_thesis/figures/view3_3d_spatial_equivariant_conformer.png)

### Key Content Displayed:
- **Molecule Info & Spatial Metric Badge**: 3D heavy atom count, Radius of Gyration ($R_g$), Maximum Euclidean span, View 3 predicted logS ($\hat{y}_3$), and error.
- **High-Resolution 3D Ball-and-Stick Conformer**:
  - Clean 3D perspective with depth cueing and element-specific colors ($\text{C}$: dark slate, $\text{O}$: red, $\text{N}$: blue, $\text{Cl}$: green, $\text{H}$: gray).
  - Conformation notes highlighting spatial geometry (e.g., tripodal propeller in Fenarimol, non-planar spiro-envelope in Hydantoin, flat 2D/3D sheet in Anthanthrene).
  - Explicit $E(n)$-equivariant coordinate dynamics formula.
- **Separate Pairwise 3D Euclidean Distance Matrix Heatmap ($D_{ij}$)**:
  - Spacious $N \times N$ matrix display with a dedicated colorbar (0 to $9.1\text{ Å}$), completely separated from the 3D conformer with zero overlap.
  - Visualizes the continuous spatial distance field processed by the EGNN's radial basis function (RBF) kernels.
- **View 3 Latent Signal Vector ($\mathbf{h}_{v3} \in \mathbb{R}^{128}$)**:
  - Formatted as an $8 \times 16$ spatial radial feature matrix.

---

## 4. Multi-Modal Fusion & Dynamic Gating Synergy
📍 **File**: [`figures/view_fusion_dynamic_gating_synergy.png`](file:///home/farhankabirsifat/Desktop/ADD_thesis/figures/view_fusion_dynamic_gating_synergy.png)

![View Fusion Dynamic Gating Synergy](file:///home/farhankabirsifat/Desktop/ADD_thesis/figures/view_fusion_dynamic_gating_synergy.png)

### Key Content Displayed:
- **Ground Truth & Error Benchmark**: Experimental logS ($y_{\text{true}}$) and TriBioNode v2 fused absolute error.
- **Wide Dynamic Gating Allocation Bar Chart ($[w_1, w_2, w_3]$)**:
  - Generous horizontal bars showing exact attention weights and percentages allocated to View 1 (Blue), View 2 (Green), and View 3 (Purple).
  - Demonstrates how gating routes $99.7\%$ of attention to View 3 for flexible aliphatic rings, but shifts up to $15.6\%$ of attention to View 1 and View 2 for planar polycyclic aromatic rings.
- **Direct Prediction Comparison & Synergy Outcome**:
  - Side-by-side bar chart comparing View 1, View 2, View 3, TriBioNode v2 Fused, and Ground Truth.
  - Explains the **orthogonal error cancellation mechanism** where View 1 and View 2 underpredictions balance View 3 overpredictions to achieve an absolute error of only **$0.08$ log units** on challenging outliers like the Fenarimol analog.
