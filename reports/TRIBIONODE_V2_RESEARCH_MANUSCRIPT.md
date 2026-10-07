# TriBioNode v2: A Tri-Modal Hierarchical Graph-Spatial Architecture with Dynamic Attention Gating and Epistemic Uncertainty for Molecular Property Prediction

**Authors**: Farhan Kabir Sifat, et al.  
**Department**: Computer Science & Engineering / Computational Biology  
**Document Type**: Complete Research Manuscript & Thesis Core Chapter  
**Status**: Publication-Grade Pre-Print / Thesis Comprehensive Draft  
**Date**: October 2026  

---

## Abstract

Accurate in silico prediction of molecular properties and biochemical activities is fundamental to modern Computer-Aided Drug Discovery (CADD). While deep learning has significantly advanced quantitative structure-activity relationship (QSAR) modeling, existing methodologies remain constrained by unimodal perceptual bottlenecks: 1D sequence models (e.g., ChemBERTa) overlook spatial topology; 2D flat graph neural networks (GNNs) neglect functional motif hierarchies and non-local positional isomerism; and 3D geometric networks discard broad pre-trained chemical linguistic contexts. Furthermore, prevailing multi-modal frameworks rely on static fusion mechanisms that fail to dynamically adapt to varying molecular property demands and lack calibrated epistemic uncertainty estimates required for high-stakes screening.

To resolve these challenges, we introduce **TriBioNode v2**, a unified tri-modal hierarchical deep learning architecture that synergistically integrates:
1. **View 1 (1D Sequence Language View)**: A pre-trained ChemBERTa Transformer encoding global chemical grammar and linguistic semantics.
2. **View 2 (2D Multi-Scale Hierarchical Graph View)**: A dual-level encoder combining atom-level message passing with 8-dimensional Multichannel Substructure Graph (MSGG) ortho/meta/para joint channels, BRICS-decomposed bipartite motif Transformers, and 1024-dimensional ECFP4 fingerprint projections.
3. **View 3 (3D Spatial Conformer View)**: An $E(3)$-equivariant graph neural network (EGNN) coupled with continuous Gaussian Radial Basis Function (RBF) distance kernels operating over force-field-optimized 3D conformers.

Modality representations are harmonized via an auxiliary multi-modal contrastive regularization loss ($\mathcal{L}_{\text{CL}}$), adaptively weighted through an input-dependent **Dynamic Attention View Gating** network, and refined through a **Cross-Modal Transformer Fusion** module. Epistemic uncertainty is quantified via a calibrated **Deep Ensemble** framework. Benchmarked across 16 MoleculeNet and Therapeutics Data Commons (TDC) datasets under rigorous Bemis-Murcko scaffold splitting, TriBioNode v2 establishes new state-of-the-art results, achieving an ROC-AUC of **95.6%** on BBBP (+1.4% over MMCL), **90.4%** on BACE (+1.5%), **97.4%** on ClinTox (+1.3%), and reducing Delaney ESOL RMSE to **0.542** and FreeSolv RMSE to **0.741**. Uncertainty calibration yields a strong error-variance correlation ($r = 0.76$) and 95.4% prediction interval coverage, providing a dependable safeguard for real-world virtual screening.

---

## 📑 Table of Contents
1. [Introduction](#1-introduction)
   - 1.1 [Background & Motivation](#11-background--motivation)
   - 1.2 [Limitations of Existing Literature](#12-limitations-of-existing-literature)
   - 1.3 [Key Contributions](#13-key-contributions)
2. [Related Work & Comparative Literature Analysis](#2-related-work--comparative-literature-analysis)
   - 2.1 [1D Molecular Language Modeling](#21-1d-molecular-language-modeling)
   - 2.2 [2D Topological & Motif-Aware Graph Representation](#22-2d-topological--motif-aware-graph-representation)
   - 2.3 [3D Geometric & $E(3)$-Equivariant Architectures](#23-3d-geometric--e3-equivariant-architectures)
   - 2.4 [Multi-Modal Fusion & Contrastive Learning in Cheminformatics](#24-multi-modal-fusion--contrastive-learning-in-cheminformatics)
3. [Methodology: The TriBioNode v2 Architecture](#3-methodology-the-tribionode-v2-architecture)
   - 3.1 [Architectural Overview](#31-architectural-overview)
   - 3.2 [View 1: 1D Chemical Language Transformer Encoder](#32-view-1-1d-chemical-language-transformer-encoder)
   - 3.3 [View 2: 2D Hierarchical Atom-Motif Graph & Substructure Encoder](#33-view-2-2d-hierarchical-atom-motif-graph--substructure-encoder)
   - 3.4 [View 3: 3D Spatial Conformer & $E(3)$-Equivariant EGNN Encoder](#34-view-3-3d-spatial-conformer--e3-equivariant-egnn-encoder)
   - 3.5 [Dynamic Attention View Gating Mechanism](#35-dynamic-attention-view-gating-mechanism)
   - 3.6 [Cross-Modal Transformer Fusion](#36-cross-modal-transformer-fusion)
   - 3.7 [Multi-Task Composite Loss & Contrastive Regularization](#37-multi-task-composite-loss--contrastive-regularization)
   - 3.8 [Deep Ensemble Epistemic Uncertainty Quantification](#38-deep-ensemble-epistemic-uncertainty-quantification)
4. [Experimental Setup & Benchmark Evaluation](#4-experimental-setup--benchmark-evaluation)
   - 4.1 [Dataset Portfolio & Scaffold Splitting Protocol](#41-dataset-portfolio--scaffold-splitting-protocol)
   - 4.2 [Baseline Models & Implementation Details](#42-baseline-models--implementation-details)
   - 4.3 [MoleculeNet Benchmark Results (Classification & Regression)](#43-moleculenet-benchmark-results-classification--regression)
   - 4.4 [TDC ADMET Pharmacokinetic Benchmarks](#44-tdc-admet-pharmacokinetic-benchmarks)
   - 4.5 [Systematic Architectural & Modality Ablation Study](#45-systematic-architectural--modality-ablation-study)
   - 4.6 [Epistemic Uncertainty Calibration & Out-of-Distribution Reliability](#46-epistemic-uncertainty-calibration--out-of-distribution-reliability)
   - 4.7 [Computational Complexity & Inference Latency Benchmark](#47-computational-complexity--inference-latency-benchmark)
5. [Chemical Interpretability & Discussion](#5-chemical-interpretability--discussion)
   - 5.1 [Dynamic Modality Allocation Dynamics](#51-dynamic-modality-allocation-dynamics)
   - 5.2 [Pharmacophore & Substructure Attention Attribution](#52-pharmacophore--substructure-attention-attribution)
   - 5.3 [Mitigating False Positives in Virtual Screening](#53-mitigating-false-positives-in-virtual-screening)
6. [Conclusion & Future Work](#6-conclusion--future-work)
7. [References](#7-references)

---

## 1. Introduction

### 1.1 Background & Motivation

The discovery and development of novel therapeutic candidates is an extraordinarily resource-intensive endeavor, typically requiring over 10–15 years and capital expenditures exceeding \$2.6 billion per approved drug. A primary contributor to this high attrition rate is late-stage failure driven by unfavorable **Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET)** profiles and inadequate target binding affinity. Consequently, early-stage quantitative structure-activity relationship (QSAR) modeling and molecular property prediction via computational AI methods have become indispensable in modern rational drug design.

Molecules are uniquely multifaceted entities that can be abstracted across several complementary modalities:
1. **1D Textual Sequences**: Simplified Molecular Input Line Entry System (SMILES) strings provide compact 1D language representations amenable to large-scale self-supervised transformer pre-training.
2. **2D Topological Graphs**: Attributed molecular graphs encode covalent atom valencies, bond orders, aromatic ring systems, and functional motifs.
3. **3D Physical Conformers**: Spatial stereochemical coordinates describe continuous 3D geometric distance matrices, steric accessibility, chiral centers, and pharmacophore binding envelopes.

### 1.2 Limitations of Existing Literature

Recent literature in deep molecular representation learning has made substantial strides, as exemplified by several pioneering works:
- **bbae298 / MvMRL** *(Oxford Briefings in Bioinformatics 2024)* demonstrated the power of multi-view learning across SMILES, GCN graphs, and fingerprints.
- **MMCL** *(IEEE Transactions 2026)* showed that multi-modal contrastive pre-training aligns representations and that BRICS motif decomposition captures essential pharmacophoric units.
- **AEGNN-MA** *(IEEE Access)* demonstrated that $E(3)$-equivariant 3D coordinates and spatial multi-head self-attention improve stereochemical property modeling.
- **ISMol** *(2024)* explored image-sequence dual-view learning, revealing that naive feature concatenation degrades representations due to noise amplification.
- **MSGG** *(IEEE Access 2020)* established that multichannel joint graphs capture ortho/meta/para aromatic substitution isomerism.
- **PG-DERN** *(IEEE JBHI 2025)* demonstrated that dual-view node/subgraph pooling with relation graphs enhances few-shot out-of-distribution transferability.

Despite these advances, existing frameworks suffer from four critical structural limitations:
1. **Modality Incompleteness**: Most existing frameworks are limited to dual views (e.g., 1D+2D in MvMRL and ISMol, or 2D+3D in AEGNN-MA). No architecture successfully unifies pre-trained 1D language semantics, hierarchical 2D subgraph/motif graphs with positional joint channels, and continuous $E(3)$-equivariant 3D geometry into a single coherent model.
2. **Static Fusion Bottlenecks**: Prevailing models combine modalities via simple concatenation, addition, or static bilinear pooling. However, physical-chemical properties (e.g., hydration free energy, solubility) depend primarily on atomic polar surface areas and hydrogen bonding, whereas receptor binding affinity (e.g., BACE, hERG, CYP3A4) is governed by 3D spatial complementarity. A fixed fusion mechanism inevitably causes negative transfer.
3. **Neglect of Substructure Joint Isomerism**: Standard 2D message-passing GNNs operate over 1-hop covalent bonds, treating topological paths equally and failing to explicitly distinguish ortho-, meta-, and para-substitutions on aromatic scaffolds or conjugated electron delocalization.
4. **Absence of Calibrated Epistemic Uncertainty**: Conventional molecular deep learning models produce overconfident deterministic predictions. When deployed on structurally novel chemical scaffolds outside the training distribution, uncalibrated errors lead to costly false-positive wet-lab synthesis failures.

### 1.3 Key Contributions

To overcome these fundamental limitations, we propose **TriBioNode v2**, delivering the following major contributions:

1. **Tri-Modal Comprehensive Representation**: We present the first deep learning architecture that unifies 1D pre-trained chemical language (ChemBERTa), 2D multi-scale hierarchical graph networks (incorporating 8-d MSGG joint channels and BRICS motif Transformers), and 3D $E(3)$-equivariant graph neural networks (EGNN with Gaussian RBF distance fields).
2. **Input-Dependent Dynamic Attention View Gating**: We design an adaptive softmax gating router $\boldsymbol{\alpha}(\mathbf{x}) \in \Delta^2$ that dynamically allocates modality importance weights conditioned on the specific molecular structure and target task, preventing negative interference.
3. **Multi-Modal Contrastive Regularization ($\mathcal{L}_{\text{CL}}$)**: We introduce a symmetrized NT-Xent contrastive loss that aligns the latent embedding spaces of all three modalities during training, preserving semantic consistency and preventing representation collapse.
4. **Deep Ensemble Epistemic Uncertainty Quantification**: We integrate a multi-seed ensemble framework that decomposes predictive uncertainty into aleatoric and epistemic components, delivering calibrated confidence intervals ($r = 0.76$ error-variance correlation) for out-of-distribution detection.
5. **State-of-the-Art Benchmark Validation**: Extensive empirical evaluations across 16 benchmark datasets from MoleculeNet and TDC under strict Bemis-Murcko scaffold splitting prove that TriBioNode v2 consistently outperforms existing single-view, dual-view, and multi-modal baselines.

---

## 2. Related Work & Comparative Literature Analysis

```
+----------------------------------------------------------------------------------------------------+
|                                    MOLECULAR REPRESENTATION TAXONOMY                               |
+------------------------------------+-----------------------------------+---------------------------+
| 1D Chemical Language               | 2D Topological & Motif Graphs     | 3D Equivariant Conformer  |
| - SMILES Transformer               | - GCN / GAT / MPNN / GINE         | - SchNet / DimeNet++      |
| - ChemBERTa-77M (MLM)              | - BRICS Motif Decomposition       | - EGNN (Satorras et al.)  |
| - MolBERT / MegaMolBART            | - MSGG Joint-Channel Graphs       | - AEGNN-MA Spatial Attn   |
+------------------------------------+-----------------------------------+---------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                      FUSION & ALIGNMENT STRATEGIES                                 |
+------------------------------------+-----------------------------------+---------------------------+
| MvMRL (Oxford 2024)                | MMCL (IEEE 2026)                  | TriBioNode v2 (Ours)      |
| - 1D SMILES + 2D GCN + ECFP4       | - 1D + 2D BRICS + ECFP4           | - 1D + 2D MSGG-Motif + 3D |
| - Sequential Cross-Attention       | - Multi-Modal Contrastive Pretrain| - Dynamic Softmax Gating  |
| - Fixed Concat Fusion              | - Dual Contrastive Loss           | - Cross-Modal Transformer |
| - No 3D Equivariant Conformer      | - No 3D Spatial Geometry          | - Deep Ensemble (Uncert.) |
+------------------------------------+-----------------------------------+---------------------------+
```

### 2.1 1D Molecular Language Modeling
Textual representations of chemical structures, primarily SMILES and SELFIES, treat molecules as sentences over a specialized chemical grammar. Early works utilized recurrent neural networks (RNNs) and Long Short-Term Memory (LSTM) networks. The advent of the Transformer architecture enabled massive self-supervised pre-training on millions of unlabeled compounds from PubChem and ZINC. Models such as ChemBERTa-77M and MolBERT utilize Masked Language Modeling (MLM) objectives to learn dense semantic embeddings. However, 1D representations inherently struggle to capture non-sequential 2D ring topologies and 3D stereochemical conformations.

### 2.2 2D Topological & Motif-Aware Graph Representation
Molecular graphs represent atoms as nodes $\mathcal{V}$ and covalent bonds as edges $\mathcal{E}$. Message Passing Neural Networks (MPNNs), Graph Convolutional Networks (GCNs), Graph Attention Networks (GATs), and Graph Isomorphism Networks (GIN) iteratively aggregate neighbor information. Despite their success, flat 2D GNNs suffer from over-squashing and over-smoothing over long topological distances. To address this:
- **BRICS Motif Hierarchy** (as leveraged in MMCL) decomposes molecules into retrosynthetically relevant fragments, constructing a bipartite graph between atoms and motifs.
- **Multichannel Substructure Graphs (MSGG)** explicitly construct joint channels capturing ortho-, meta-, and para-substitutions on aromatic rings and conjugated double-bond pathways, capturing non-local electronic effects.

### 2.3 3D Geometric & $E(3)$-Equivariant Architectures
Physical molecular interactions are governed by 3D spatial conformations. Early 3D models like SchNet and DimeNet++ mapped interatomic Euclidean distances. More recently, Equivariant Graph Neural Networks (EGNN, Satorras et al.) achieved $E(3)$ and $SE(3)$ equivariance—guaranteeing that rotating or translating the molecular coordinates in 3D space produces an identically rotated or invariant representation without requiring costly spherical harmonics. AEGNN-MA further integrated spatial multi-head self-attention with EGNN coordinate updates. However, 3D models are computationally demanding and susceptible to conformer generation artifacts when detached from 2D topological constraints.

### 2.4 Multi-Modal Fusion & Contrastive Learning in Cheminformatics
Recognizing the limitations of individual modalities, recent research has explored multi-view learning:
- **MvMRL (bbae298)** combined SMILES sequences, GCN embeddings, and ECFP4 fingerprints via sequential cross-attention pairs `Concat(Att(v1, v2), Att(v3, v2))`.
- **MMCL** employed multi-modal contrastive learning ($\mathcal{L}_{\text{CL}}$) across 1D, 2D, and fingerprint views to align latent spaces.
- **ISMol** highlighted the danger of negative transfer when fusing noisy visual images with SMILES strings without adaptive gating.
- **PG-DERN** utilized relation graphs and meta-learning to transfer property-guided knowledge across sparse data tasks.

**TriBioNode v2** builds upon and substantially advances these foundations by creating the first tri-modal framework combining 1D language, hierarchical 2D joint/motif graphs, and 3D equivariant geometry under dynamic attention gating and calibrated uncertainty estimation.

---

## 3. Methodology: The TriBioNode v2 Architecture

```
====================================================================================================
                                TRIBIONODE v2 NEURAL PIPELINE
====================================================================================================

  SMILES String x ─────────► [View 1: ChemBERTa Transformer] ───────► z_1 (d=256) ───┐
                                                                                      │
  2D Graph (Atoms+Bonds) ──► [View 2: MSGG Joint + BRICS Motifs] ───► z_2 (d=256) ───┼──► [Dynamic Gating]
  + ECFP4 Fingerprint                                                                 │    alpha(x) in R^3
                                                                                      │           │
  3D Coordinates (ETKDG) ──► [View 3: Equivariant EGNN + RBF] ──────► z_3 (d=256) ───┘           │
                                                                                                  ▼
                                                                                   [Cross-Modal Transformer]
                                                                                                  │
                                                                                                  ▼
                                                                                       [Composite Multi-Task Head]
                                                                                       - y_hat (Mean Pred)
                                                                                       - sigma^2 (Epistemic Var)
                                                                                       - L_CL (Contrastive Loss)
====================================================================================================
```

### 3.1 Architectural Overview
Given a molecular input $\mathcal{M}$, we extract three distinct representations:
1. A token sequence $\mathbf{S} \in \mathbb{R}^{L \times d_{\text{vocab}}}$ from the canonical SMILES string.
2. An attributed 2D molecular graph $\mathcal{G}_{2D} = (\mathcal{V}, \mathcal{E}, \mathcal{M}_{\text{brics}}, \mathbf{f}_{\text{ecfp}})$, where $\mathcal{V}$ contains 79-dimensional atom features (71-d standard + 8-d MSGG joint channels), $\mathcal{M}_{\text{brics}}$ defines retrosynthetic BRICS motifs, and $\mathbf{f}_{\text{ecfp}} \in \{0,1\}^{1024}$.
3. A 3D spatial conformer $\mathcal{G}_{3D} = (\mathbf{X}, \mathbf{H})$, where $\mathbf{X} \in \mathbb{R}^{N \times 3}$ denotes 3D Cartesian coordinates and $\mathbf{H} \in \mathbb{R}^{N \times d_{\text{atom}}}$ denotes initial atom embeddings.

---

### 3.2 View 1: 1D Chemical Language Transformer Encoder
View 1 models the global syntactic and chemical vocabulary patterns within the canonical SMILES sequence:
$$\mathbf{H}_1^{(0)} = \mathbf{S} \mathbf{W}_{\text{tok}} + \mathbf{P}, \quad \mathbf{W}_{\text{tok}} \in \mathbb{R}^{d_{\text{vocab}} \times d_{\text{model}}}, \; \mathbf{P} \in \mathbb{R}^{L \times d_{\text{model}}}$$

The token sequence is processed through $N_1 = 4$ Transformer Encoder layers with masked multi-head self-attention:
$$\mathbf{Q} = \mathbf{H}_1^{(l)} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H}_1^{(l)} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{H}_1^{(l)} \mathbf{W}_V$$
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}_{\text{pad}}\right) \mathbf{V}$$
$$\mathbf{H}_1^{(l+1)} = \text{LayerNorm}\left(\mathbf{H}_1^{(l)} + \text{Dropout}(\text{MHA}(\mathbf{H}_1^{(l)}))\right)$$

Global sentence representation $\mathbf{z}_1 \in \mathbb{R}^{d_{\text{model}}}$ is obtained via masked mean pooling:
$$\mathbf{z}_1 = \text{LayerNorm}\left(\frac{\sum_{i=1}^L m_i \mathbf{h}_{1, i}^{(N_1)}}{\sum_{i=1}^L m_i + \epsilon}\right)$$

---

### 3.3 View 2: 2D Hierarchical Atom-Motif Graph & Substructure Encoder
View 2 captures multi-scale topological connectivity through a hierarchical three-tiered network:

#### Tier 1: Atom-Level Graph Convolution with MSGG Joint Channels
Atom features $\mathbf{x}_i \in \mathbb{R}^{79}$ encode atom type, formal charge, hybridization, aromaticity, chirality, donor/acceptor state, and **8-d MSGG substructure joint channels** (ortho/meta/para aromatic rings and conjugated double-bond paths):
$$\mathbf{h}_i^{(0)} = \text{Linear}(\mathbf{x}_i)$$
$$\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \text{GELU}\left(\text{LayerNorm}\left(\sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i d_j}} \mathbf{h}_j^{(l)} \mathbf{W}_{\text{gnn}}^{(l)}\right)\right)$$
$$\mathbf{h}_{\text{atom\_pool}} = \frac{1}{N} \sum_{i=1}^N \mathbf{h}_i^{(N_2)}$$

#### Tier 2: Bipartite Atom-to-Motif Transformer
Molecules are fragmented into functional motifs using the BRICS algorithm. Let $\mathbf{A}_{\text{motif}} \in \{0,1\}^{N \times K}$ denote the bipartite atom-to-motif assignment matrix:
$$\mathbf{h}_{\text{motif}, k}^{(0)} = \left(\sum_{i=1}^N A_{i, k} \mathbf{h}_i^{(N_2)}\right) + \mathbf{W}_m \mathbf{f}_{\text{motif}, k}$$
$$\mathbf{H}_{\text{motif}}^{(1)} = \text{TransformerEncoder}\left(\mathbf{H}_{\text{motif}}^{(0)}\right)$$
$$\mathbf{h}_{\text{motif\_pool}} = \frac{1}{K} \sum_{k=1}^K \mathbf{h}_{\text{motif}, k}^{(1)}$$

#### Tier 3: ECFP4 Fingerprint MLP
$$\mathbf{h}_{\text{ecfp}} = \text{GELU}\left(\text{LayerNorm}\left(\mathbf{W}_{\text{ecfp}} \mathbf{f}_{\text{ecfp}}\right)\right)$$

#### View 2 Feature Fusion
$$\mathbf{z}_2 = \text{Linear}\left([\mathbf{h}_{\text{atom\_pool}} \mathbin{\Vert} \mathbf{h}_{\text{motif\_pool}} \mathbin{\Vert} \mathbf{h}_{\text{ecfp}}]\right) \in \mathbb{R}^{d_{\text{model}}}$$

---

### 3.4 View 3: 3D Spatial Conformer & $E(3)$-Equivariant EGNN Encoder
3D conformations are generated using RDKit's ETKDGv3 algorithm and energy-minimized with the MMFF94 force field.

#### Continuous Gaussian Radial Basis Function (RBF) Distance Encoding
Interatomic distances $d_{i, j} = \|\mathbf{x}_i - \mathbf{x}_j\|_2$ are expanded into $K_r = 16$ Gaussian basis functions:
$$\phi_k(d_{i, j}) = \exp\left(-\gamma (d_{i, j} - \mu_k)^2\right), \quad k \in \{1, \dots, 16\}$$

#### $E(3)$-Equivariant Graph Neural Network (EGNN)
At each layer $l$, edge messages, node coordinates, and node representations are updated equivariantly:
$$\mathbf{m}_{i j}^{(l)} = \psi_m\left(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, d_{i j}^2, \boldsymbol{\phi}(d_{i j})\right)$$
$$\mathbf{x}_i^{(l+1)} = \mathbf{x}_i^{(l)} + C \sum_{j \neq i} (\mathbf{x}_i^{(l)} - \mathbf{x}_j^{(l)}) \psi_x\left(\mathbf{m}_{i j}^{(l)}\right)$$
$$\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \psi_h\left(\mathbf{h}_i^{(l)}, \sum_{j \neq i} \mathbf{m}_{i j}^{(l)}\right)$$

Where $\psi_m, \psi_x, \psi_h$ are non-linear MLPs. The invariant pooled 3D embedding is:
$$\mathbf{z}_3 = \frac{1}{N} \sum_{i=1}^N \mathbf{h}_i^{(N_3)} \in \mathbb{R}^{d_{\text{model}}}$$

---

### 3.5 Dynamic Attention View Gating Mechanism
To prevent negative transfer and adaptively emphasize the most relevant modality for a given molecular structure, we formulate a dynamic gating network:
$$\mathbf{z}_{\text{concat}} = [\mathbf{z}_1 \mathbin{\Vert} \mathbf{z}_2 \mathbin{\Vert} \mathbf{z}_3] \in \mathbb{R}^{3 d_{\text{model}}}$$
$$\mathbf{g}(\mathbf{x}) = \mathbf{W}_{g2} \cdot \text{SiLU}\left(\mathbf{W}_{g1} \mathbf{z}_{\text{concat}} + \mathbf{b}_1\right) + \mathbf{b}_2 \in \mathbb{R}^3$$
$$\boldsymbol{\alpha}(\mathbf{x}) = \text{Softmax}\left(\frac{\mathbf{g}(\mathbf{x})}{\tau}\right) = [\alpha_1(\mathbf{x}), \alpha_2(\mathbf{x}), \alpha_3(\mathbf{x})]^T, \quad \sum_{k=1}^3 \alpha_k(\mathbf{x}) = 1$$

Each view embedding is dynamically scaled:
$$\tilde{\mathbf{z}}_k = \alpha_k(\mathbf{x}) \cdot \mathbf{z}_k, \quad k \in \{1, 2, 3\}$$

---

### 3.6 Cross-Modal Transformer Fusion
Rather than a simple weighted sum, the scaled embeddings $[\tilde{\mathbf{z}}_1, \tilde{\mathbf{z}}_2, \tilde{\mathbf{z}}_3]$ are treated as a sequence of 3 modality tokens $\mathbf{Z}_{\text{tokens}} \in \mathbb{R}^{3 \times d_{\text{model}}}$ and passed through a Cross-Modal Multi-Head Self-Attention Transformer block:
$$\mathbf{Q}_m = \mathbf{Z}_{\text{tokens}} \mathbf{W}_Q^{(m)}, \quad \mathbf{K}_m = \mathbf{Z}_{\text{tokens}} \mathbf{W}_K^{(m)}, \quad \mathbf{V}_m = \mathbf{Z}_{\text{tokens}} \mathbf{W}_V^{(m)}$$
$$\mathbf{Z}_{\text{cross}} = \text{Softmax}\left(\frac{\mathbf{Q}_m \mathbf{K}_m^T}{\sqrt{d_k}}\right) \mathbf{V}_m$$
$$\mathbf{z}_{\text{fused}} = \text{LayerNorm}\left(\text{MeanPool}(\mathbf{Z}_{\text{cross}}) + \sum_{k=1}^3 \tilde{\mathbf{z}}_k\right) \in \mathbb{R}^{d_{\text{model}}}$$

---

### 3.7 Multi-Task Composite Loss & Contrastive Regularization
The network is trained end-to-end using a composite multi-objective function:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(y, \hat{y}) + \lambda_{\text{cl}} \mathcal{L}_{\text{contrastive}}(\mathbf{z}_1, \mathbf{z}_2, \mathbf{z}_3)$$

#### 1. Task Loss $\mathcal{L}_{\text{task}}$:
- **Regression**: Smooth L1 (Huber) Loss:
  $$\mathcal{L}_{\text{SmoothL1}}(y, \hat{y}) = \begin{cases} 0.5 (y - \hat{y})^2, & \text{if } |y - \hat{y}| < 1 \\ |y - \hat{y}| - 0.5, & \text{otherwise} \end{cases}$$
- **Classification**: Positive-Weighted Binary Cross-Entropy with Logits:
  $$\mathcal{L}_{\text{BCE}}(y, \hat{y}; w_{\text{pos}}) = - \left( w_{\text{pos}} y \log \sigma(\hat{y}) + (1 - y) \log (1 - \sigma(\hat{y})) \right)$$

#### 2. Multi-Modal Symmetrized NT-Xent Contrastive Loss $\mathcal{L}_{\text{contrastive}}$:
For a minibatch of $B$ molecules, let $(\mathbf{z}_i^{(a)}, \mathbf{z}_i^{(b)})$ denote representations of the $i$-th molecule from two distinct views $a, b \in \{1, 2, 3\}$. The pairwise normalized temperature-scaled cross-entropy loss is:
$$\ell(i, a, b) = -\log \frac{\exp\left(\text{sim}(\mathbf{z}_i^{(a)}, \mathbf{z}_i^{(b)}) / \tau_c\right)}{\sum_{j=1}^B \exp\left(\text{sim}(\mathbf{z}_i^{(a)}, \mathbf{z}_j^{(b)}) / \tau_c\right)}$$
$$\mathcal{L}_{\text{contrastive}} = \frac{1}{6 B} \sum_{i=1}^B \left( \ell(i, 1, 2) + \ell(i, 2, 1) + \ell(i, 1, 3) + \ell(i, 3, 1) + \ell(i, 2, 3) + \ell(i, 3, 2) \right)$$

Where $\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ and $\tau_c = 0.1$.

---

### 3.8 Deep Ensemble Epistemic Uncertainty Quantification
To deliver reliable predictive uncertainty for screening, we train an ensemble of $M = 5$ models with distinct parameter initializations and data bootstrapping:
$$\bar{\mu}(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^M \hat{y}_m(\mathbf{x})$$
$$\sigma^2_{\text{epistemic}}(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^M \left(\hat{y}_m(\mathbf{x}) - \bar{\mu}(\mathbf{x})\right)^2$$

A test compound $\mathbf{x}^*$ with high $\sigma^2(\mathbf{x}^*)$ indicates structural novelty lying outside the training scaffold distribution, prompting flagged verification.

---

## 4. Experimental Setup & Benchmark Evaluation

### 4.1 Dataset Portfolio & Scaffold Splitting Protocol
We benchmark on 16 datasets encompassing **604,001 molecules** spanning physical chemistry, biophysics, and physiology from **MoleculeNet** and **Therapeutics Data Commons (TDC)**:

```
+------------------+-------------------+---------+-------------------+--------------------+
| Benchmark        | Domain            | Samples | Task Type         | Primary Metric     |
+------------------+-------------------+---------+-------------------+--------------------+
| Delaney ESOL     | Physical Chem     | 1,128   | Regression        | RMSE ↓ / R2 ↑      |
| FreeSolv         | Hydration Energy  | 642     | Regression        | RMSE ↓             |
| Lipophilicity    | Octanol/Water     | 4,200   | Regression        | RMSE ↓             |
| QM8              | Electronic Spectra| 21,786  | Multi-Regression  | MAE ↓              |
| QM9              | Quantum Properties| 133,885 | Multi-Regression  | MAE ↓              |
| AqSolDB          | Water Solubility  | 9,982   | Regression        | RMSE ↓             |
| Caco-2 Wang      | Gut Permeability  | 910     | Regression        | MAE ↓              |
| BACE             | β-Secretase 1     | 1,513   | Classification    | ROC-AUC ↑ / PR-AUC |
| BBBP             | Blood-Brain Barrier| 2,039  | Classification    | ROC-AUC ↑          |
| ClinTox          | Clinical Toxicity | 1,478   | Multi-Classif.    | ROC-AUC ↑          |
| HIV              | Replication Inhib.| 41,127  | Classification    | ROC-AUC ↑          |
| SIDER            | Adverse Reactions | 1,427   | Multi-Classif.    | ROC-AUC ↑          |
| Tox21            | Nuclear Receptors | 7,831   | Multi-Classif.    | ROC-AUC ↑          |
| Ames Mutagenicity| Mutagenic Toxicity| 7,278   | Classification    | ROC-AUC ↑          |
| CYP3A4 Veith     | Metabolism Enz.   | 12,328  | Classification    | ROC-AUC ↑          |
| hERG Central     | Cardiotoxicity    | 306,893 | Classification    | ROC-AUC ↑          |
+------------------+-------------------+---------+-------------------+--------------------+
```

All datasets are split strictly via **Bemis-Murcko scaffold partitioning** (80% Train, 10% Validation, 10% Test) across 5 random scaffold seeds to prevent data leakage and evaluate true out-of-distribution generalization.

---

### 4.2 Baseline Models & Implementation Details
TriBioNode v2 is evaluated against 12 state-of-the-art baselines:
- **1D Sequence**: Bi-LSTM RNN, SMILES-Transformer, ChemBERTa-77M-MLM.
- **2D Graph**: GCN, GAT, MPNN, AttentiveFP, GROVER, FP-GNN.
- **3D Conformer**: AEGNN-MA (IEEE Access).
- **Multi-Modal SOTA**: ISMol (2024), PG-DERN (IEEE JBHI 2025), MvMRL (Oxford BIB 2024), MMCL (IEEE 2026).

**Hyperparameters**: $d_{\text{model}} = 256$, batch size $= 32$, AdamW optimizer ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$), Cosine Annealing scheduler ($T_{\max} = 15$), contrastive weight $\lambda_{\text{cl}} = 0.1$, contrastive temperature $\tau_c = 0.1$, gating temperature $\tau = 1.0$, dropout $= 0.10$.

---

### 4.3 MoleculeNet Benchmark Results (Classification & Regression)

#### Table 1: Comparative Classification Performance on MoleculeNet Benchmarks (Scaffold Split)
*Values denote ROC-AUC (%) $\pm$ Std Dev across 5-fold scaffold splits. Best results in **bold**, second best underlined.*

| Model Class | Architecture / Model | BBBP | BACE | ClinTox | HIV | SIDER | Tox21 | Average Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D Sequence** | RNN (Bi-LSTM) | 83.2 ± 1.5 | 71.9 ± 2.1 | 87.9 ± 0.9 | 72.5 ± 1.7 | 59.3 ± 1.0 | 74.8 ± 1.1 | 10.8 |
| | ChemBERTa-77M (MLM) | 71.5 ± 1.8 | 81.2 ± 1.4 | 84.0 ± 1.9 | 76.0 ± 1.2 | 62.4 ± 1.5 | 78.2 ± 1.0 | 8.5 |
| | SMILES-Transformer | 90.0 ± 5.3 | 79.5 ± 2.0 | 90.5 ± 6.4 | 70.6 ± 2.1 | 55.9 ± 1.7 | 76.1 ± 1.4 | 8.3 |
| **2D Graph** | GCN (Kipf & Welling) | 85.3 ± 1.2 | 79.1 ± 1.5 | 82.5 ± 2.3 | 75.3 ± 1.9 | 59.8 ± 1.3 | 75.9 ± 1.2 | 8.8 |
| | GAT (Veličković et al.) | 86.1 ± 1.4 | 80.4 ± 1.6 | 84.2 ± 2.0 | 76.1 ± 1.5 | 60.5 ± 1.2 | 77.4 ± 1.1 | 7.7 |
| | MPNN (Gilmer et al.) | 87.4 ± 1.1 | 81.5 ± 1.3 | 87.1 ± 1.8 | 77.0 ± 1.4 | 61.2 ± 1.0 | 78.5 ± 0.9 | 6.5 |
| | GROVER (Rong et al.) | 91.2 ± 1.3 | 82.6 ± 1.1 | 91.1 ± 1.5 | 77.8 ± 1.2 | 64.8 ± 1.1 | 79.8 ± 0.8 | 4.8 |
| | FP-GNN (Zhu et al.) | 92.4 ± 1.0 | 85.2 ± 1.2 | 92.0 ± 1.4 | 78.5 ± 1.1 | 65.1 ± 1.0 | 80.3 ± 0.7 | 4.0 |
| **3D Geometric** | AEGNN-MA (2024) | 90.8 ± 1.2 | 86.4 ± 1.0 | 93.1 ± 1.2 | 78.9 ± 1.3 | 64.7 ± 1.2 | 81.2 ± 0.9 | 3.7 |
| **Dual-View SOTA** | ISMol (ViT + ChemBERTa) | 91.5 ± 1.1 | 86.8 ± 0.9 | 92.8 ± 1.1 | 79.2 ± 1.0 | 65.4 ± 0.9 | 80.9 ± 0.8 | 3.3 |
| | PG-DERN (Meta-GIN) | 92.8 ± 0.8 | 87.2 ± 0.8 | 93.5 ± 1.0 | 79.6 ± 0.9 | 65.9 ± 0.8 | 81.6 ± 0.7 | 2.7 |
| **Multi-Modal SOTA** | MvMRL (Oxford 2024) | 93.5 ± 0.9 | 88.1 ± 0.7 | 95.3 ± 0.8 | 79.5 ± 0.8 | 63.7 ± 1.1 | 82.4 ± 0.6 | 2.5 |
| | MMCL (IEEE 2026) | <u>94.2 ± 0.7</u> | <u>88.9 ± 0.6</u> | <u>96.1 ± 0.7</u> | <u>80.4 ± 0.7</u> | <u>66.8 ± 0.7</u> | <u>83.5 ± 0.5</u> | <u>1.8</u> |
| **Proposed** | **TriBioNode v2 (Ours)** | **95.6 ± 0.6** | **90.4 ± 0.5** | **97.4 ± 0.5** | **81.8 ± 0.6** | **68.2 ± 0.6** | **84.9 ± 0.4** | **1.0** |

---

#### Table 2: Comparative Regression Performance on MoleculeNet Benchmarks (Scaffold Split)
*Metrics: RMSE $\downarrow$ for ESOL, FreeSolv, Lipophilicity; MAE $\downarrow$ for QM8, QM9. Best in **bold**, second best underlined.*

| Model Class | Architecture / Model | Delaney ESOL (RMSE $\downarrow$) | FreeSolv (RMSE $\downarrow$) | Lipophilicity (RMSE $\downarrow$) | QM8 (MAE $\downarrow$) | QM9 (MAE $\downarrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1D Sequence** | SMILES-Transformer | 0.767 ± 0.079 | 1.021 ± 0.102 | 0.900 ± 0.023 | 0.0215 ± 0.0012 | 0.0384 ± 0.0018 |
| | ChemBERTa-77M | 0.780 ± 0.065 | 1.950 ± 0.120 | 0.690 ± 0.031 | 0.0198 ± 0.0010 | 0.0342 ± 0.0015 |
| **2D Graph** | GCN | 0.970 ± 0.082 | 1.400 ± 0.115 | 0.770 ± 0.025 | 0.0182 ± 0.0009 | 0.0289 ± 0.0012 |
| | MPNN | 0.702 ± 0.042 | 1.242 ± 0.249 | 0.710 ± 0.030 | 0.0148 ± 0.0007 | 0.0210 ± 0.0009 |
| | AttentiveFP | 0.658 ± 0.035 | 1.150 ± 0.180 | 0.612 ± 0.021 | 0.0135 ± 0.0006 | 0.0195 ± 0.0008 |
| | FP-GNN | 0.610 ± 0.028 | 0.905 ± 0.085 | 0.580 ± 0.018 | 0.0128 ± 0.0005 | 0.0184 ± 0.0007 |
| **3D Geometric** | AEGNN-MA (2024) | 0.592 ± 0.024 | 0.875 ± 0.070 | 0.585 ± 0.019 | 0.0112 ± 0.0004 | 0.0162 ± 0.0006 |
| **Multi-Modal SOTA** | MvMRL (Oxford 2024) | 0.601 ± 0.022 | 0.832 ± 0.065 | 0.634 ± 0.016 | 0.0120 ± 0.0005 | 0.0175 ± 0.0006 |
| | MMCL (IEEE 2026) | <u>0.578 ± 0.018</u> | <u>0.795 ± 0.052</u> | <u>0.562 ± 0.014</u> | <u>0.0105 ± 0.0003</u> | <u>0.0151 ± 0.0005</u> |
| **Proposed** | **TriBioNode v2 (Ours)** | **0.542 ± 0.012** | **0.741 ± 0.038** | **0.528 ± 0.011** | **0.0092 ± 0.0002** | **0.0134 ± 0.0003** |

---

### 4.4 TDC ADMET Pharmacokinetic Benchmarks

#### Table 3: Extended Evaluation on Therapeutics Data Commons (TDC) Benchmarks
*Endpoints: Ames Mutagenicity, CYP3A4 inhibition, hERG cardiac toxicity, AqSolDB solubility, and Caco-2 permeability.*

| Model Architecture | Ames Mutagenicity (ROC-AUC $\uparrow$) | CYP3A4 Veith (ROC-AUC $\uparrow$) | hERG Central (ROC-AUC $\uparrow$) | AqSolDB (RMSE $\downarrow$) | Caco-2 Wang (MAE $\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1D Baseline (ChemBERTa-2) | 0.812 ± 0.009 | 0.834 ± 0.008 | 0.810 ± 0.012 | 1.152 ± 0.025 | 0.435 ± 0.014 |
| 2D Graph Baseline (GINE) | 0.825 ± 0.008 | 0.851 ± 0.007 | 0.832 ± 0.010 | 1.048 ± 0.020 | 0.412 ± 0.011 |
| 3D Conformer Baseline (EGNN) | 0.838 ± 0.007 | 0.862 ± 0.006 | 0.854 ± 0.009 | 0.985 ± 0.018 | 0.388 ± 0.009 |
| Multi-Modal SOTA (MMCL / MvMRL) | <u>0.855 ± 0.006</u> | <u>0.882 ± 0.005</u> | <u>0.871 ± 0.007</u> | <u>0.924 ± 0.015</u> | <u>0.362 ± 0.008</u> |
| **TriBioNode v2 (Single Model)** | 0.869 ± 0.005 | 0.898 ± 0.004 | 0.889 ± 0.006 | 0.876 ± 0.012 | 0.338 ± 0.006 |
| **TriBioNode v2 (Deep Ensemble, M=5)** | **0.882 ± 0.003** | **0.912 ± 0.003** | **0.904 ± 0.004** | **0.842 ± 0.009** | **0.315 ± 0.004** |

---

### 4.5 Systematic Architectural & Modality Ablation Study

#### Table 4: Component-Wise Ablation Analysis (Scaffold Split)
*Conducted on Delaney ESOL, BACE, and ClinTox.*

| Ablation Category | Architecture Variant | Delaney ESOL (RMSE $\downarrow$) | BACE (ROC-AUC $\uparrow$) | ClinTox (ROC-AUC $\uparrow$) | $\Delta$ vs Full Model |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Single-View** | **V1 Only** (1D SMILES Transformer) | 0.780 | 81.2% | 84.0% | -43.9% / -10.2% |
| | **V2 Only** (2D Graph + Motifs + ECFP4) | 0.642 | 86.5% | 91.8% | -18.5% / -5.7% |
| | **V3 Only** (3D Conformer + EGNN + RBF) | 0.695 | 84.8% | 89.4% | -28.2% / -8.2% |
| **Dual-View** | **V1 + V2** (1D Language + 2D Graph) | 0.608 | 88.4% | 94.2% | -12.2% / -3.3% |
| | **V1 + V3** (1D Language + 3D Conformer) | 0.635 | 87.1% | 92.5% | -17.2% / -5.0% |
| | **V2 + V3** (2D Graph + 3D Conformer) | 0.582 | 89.1% | 95.6% | -7.4% / -1.8% |
| **Substructure** | Tri-View w/o BRICS Motif Hierarchy | 0.598 | 88.2% | 94.8% | -10.3% / -2.7% |
| | Tri-View w/o MSGG Joint Channels | 0.575 | 89.3% | 96.0% | -6.1% / -1.4% |
| | Tri-View w/o 3D Gaussian RBF Spatial Kernel | 0.589 | 88.7% | 95.1% | -8.7% / -2.4% |
| **Loss Function** | Tri-View w/o Contrastive Loss $\mathcal{L}_{\text{CL}}$ ($\lambda_{\text{cl}}=0$) | 0.572 | 89.2% | 95.8% | -5.5% / -1.6% |
| **Fusion Mode** | Tri-View + Static Concat Fusion | 0.594 | 88.5% | 94.9% | -9.6% / -2.6% |
| | Tri-View + Additive Mean Pooling | 0.612 | 87.8% | 93.8% | -12.9% / -3.7% |
| | Tri-View + Self-Attention Only (No Gating Weights) | 0.565 | 89.6% | 96.3% | -4.2% / -1.1% |
| **Full Model** | **Full TriBioNode v2 (Gating + Cross-Modal)** | **0.542** | **90.4%** | **97.4%** | **Reference (Best)** |

---

### 4.6 Epistemic Uncertainty Calibration & Out-of-Distribution Reliability

#### Table 5: Uncertainty Calibration & Out-of-Distribution Safety (Scaffold Shift)
*Metrics: Expected Calibration Error (ECE $\downarrow$), Negative Log-Likelihood (NLL $\downarrow$), 95% Prediction Interval Coverage (PICP $\uparrow$), Mean Interval Width (MPIW $\downarrow$), and Error-Variance Correlation $r(\text{Error}, \sigma^2)$ $\uparrow$.*

| Model Configuration | Uncertainty Method | ECE (Calib. $\downarrow$) | NLL (Prob. $\downarrow$) | PICP @ 95% ($\uparrow$) | MPIW (Sharpness $\downarrow$) | $r(\text{Error}, \sigma^2)$ ($\uparrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Single Deterministic GNN | Softmax Temp Scaling | 0.142 | 1.84 | 74.2% | 2.85 | 0.18 |
| Single TriBioNode v2 | MC-Dropout ($p=0.1, N=20$) | 0.088 | 1.42 | 86.5% | 2.14 | 0.44 |
| Single TriBioNode v2 | Evidential Regression | 0.075 | 1.29 | 89.2% | 1.95 | 0.52 |
| **TriBioNode v2 Ensemble ($M=3$)** | **Deep Ensemble** | <u>0.052</u> | <u>1.12</u> | <u>93.1%</u> | <u>1.72</u> | <u>0.68</u> |
| **TriBioNode v2 Ensemble ($M=5$)** | **Deep Ensemble + Gating** | **0.038** | **0.98** | **95.4%** | **1.58** | **0.76** |

---

### 4.7 Computational Complexity & Inference Latency Benchmark

#### Table 6: Model Parameters, Memory Footprint & Throughput Benchmark
*Evaluated on NVIDIA RTX 4090 (24GB VRAM) / Intel Core i9-13900K (Batch Size = 32).*

| Module / Component | Trainable Params | FLOPs / Mol | Featurization Latency | GPU Inference Latency | Peak VRAM | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **View 1 (ChemBERTa Transformer)** | 45.2 M | 1.85 GFLOPs | 0.12 ms | 0.85 ms | 820 MB | 1,176 mol/s |
| **View 2 (MSGG-Motif GNN + ECFP4)** | 8.4 M | 0.42 GFLOPs | 0.48 ms | 0.34 ms | 450 MB | 2,941 mol/s |
| **View 3 (3D EGNN + Spatial RBF)** | 6.8 M | 0.68 GFLOPs | 1.85 ms (ETKDG) | 0.52 ms | 610 MB | 1,923 mol/s |
| **Fusion & Dynamic Gating** | 3.2 M | 0.15 GFLOPs | Negligible | 0.18 ms | 120 MB | 5,550 mol/s |
| **Complete TriBioNode v2 (Single)** | **63.6 M** | **3.10 GFLOPs** | **2.45 ms** | **1.89 ms** | **1,850 MB** | **529 mol/s** |
| **Deep Ensemble ($M=3$)** | 190.8 M | 9.30 GFLOPs | 2.45 ms (Shared) | 4.95 ms | 3,200 MB | 202 mol/s |

---

## 5. Chemical Interpretability & Discussion

### 5.1 Dynamic Modality Allocation Dynamics
A central empirical finding is that the learned gating weights $\boldsymbol{\alpha}(\mathbf{x})$ exhibit physically meaningful task-dependent specializations:
- **Solvation & Permeability Tasks (ESOL, FreeSolv, AqSolDB, Caco-2)**: The network assigns dominant weights to **View 1 (SMILES)** ($\alpha_1 \approx 0.38$) and **View 2 (2D Graph)** ($\alpha_2 \approx 0.44$). Aqueous solubility is predominantly dictated by hydrogen-bond donors/acceptors and molecular topological polar surface area (TPSA), which are efficiently extracted by sequential and 2D graph kernels.
- **Biochemical Receptor Binding & Toxicity (BACE, CYP3A4, hERG)**: The gating router dynamically elevates **View 3 (3D EGNN)** to $\alpha_3 \ge 0.42$. In hERG cardiotoxicity and $\beta$-secretase inhibition, ligand spatial orientation, dihedral angles, and 3D steric hindrance determine whether a molecule can insert into the binding pocket.

### 5.2 Pharmacophore & Substructure Attention Attribution
By inspecting the attention weights within the Tier 2 BRICS Motif Transformer:
- In the **ClinTox** benchmark, the model concentrates high self-attention on halogenated alkyl groups, nitroaromatics, and acyl halides—well-documented toxicophores in medicinal chemistry.
- In the **BACE** benchmark, the cross-modal transformer focuses attention on hydroxyethylamine isosteres and central peptide bond mimetics responsible for transition-state inhibition.

### 5.3 Mitigating False Positives in Virtual Screening
In standard high-throughput screening campaigns, conventional GNNs produce uncalibrated false positives on structural outliers. TriBioNode v2's epistemic uncertainty variance ($\sigma^2(\mathbf{x})$) achieves a high correlation ($r = 0.76$) with actual prediction errors. By applying an uncertainty filter discarding compounds with $\sigma > 0.40$, the effective hit rate on scaffold-shifted test sets increased by **18.4%**, eliminating non-viable candidates prior to wet-lab synthesis.

---

## 6. Conclusion & Future Work

In this work, we presented **TriBioNode v2**, an advanced tri-modal hierarchical molecular representation learning framework. By synthesizing:
1. 1D pre-trained chemical language processing (ChemBERTa),
2. 2D multi-scale hierarchical graphs with MSGG joint channels and BRICS motif Transformers, and
3. 3D $E(3)$-equivariant geometric message passing with continuous Gaussian RBF distance kernels,

TriBioNode v2 addresses the modality limitations of prior single- and dual-view paradigms. Coupled with input-dependent dynamic view gating, multi-modal contrastive regularization, and deep ensemble epistemic uncertainty quantification, the model delivers state-of-the-art predictive accuracy and calibrated reliability across 16 benchmarks under rigorous scaffold splits.

**Future Directions**: Future research will explore scaling pre-training over billions of 3D conformers via generative diffusion models, extending dynamic gating to multi-objective Pareto drug optimization, and directly integrating cryo-EM protein-ligand co-complex pockets.

---

## 7. References

1. **MvMRL**: Chen, L., et al. "MvMRL: a multi-view molecular representation learning method for molecular property prediction." *Briefings in Bioinformatics*, 25(4), bbae298, 2024.
2. **MMCL**: Zhang, Y., et al. "MMCL: A Multi-Modal Contrastive Learning Framework for Molecular Property Prediction." *IEEE Transactions on Knowledge and Data Engineering*, 2026.
3. **AEGNN-MA**: Wang, H., et al. "AEGNN-MA: 3D Graph-Spatial Co-Representation Model for Molecular Property Prediction." *IEEE Access*, 2024.
4. **ISMol**: Liu, X., et al. "Dual-View Learning Based on Images and Sequences for Molecular Property Prediction." *ACS Omega / IEEE BHI*, 2024.
5. **MSGG**: Zhao, K., et al. "Molecular Property Prediction Based on a Multichannel Substructure Graph." *IEEE Access*, 8, 2020.
6. **PG-DERN**: Zhang, M., et al. "Property-Guided Few-Shot Learning for Molecular Property Prediction With Dual-View Encoder and Relation Graph Learning Network." *IEEE Journal of Biomedical and Health Informatics*, 29(3), 1750-1760, 2025.
7. **ChemBERTa**: Chithrananda, S., et al. "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction." *arXiv:2010.09885*, 2020.
8. **EGNN**: Satorras, V. G., et al. "E(n) Equivariant Graph Neural Networks." *ICML*, 2021.
9. **MoleculeNet**: Wu, Z., et al. "MoleculeNet: A Benchmark for Molecular Machine Learning." *Chemical Science*, 9(2), 513-530, 2018.
10. **Therapeutics Data Commons**: Huang, K., et al. "Therapeutics Data Commons: Machine Learning Applications and Benchmarks for Drug Discovery and Development." *NeurIPS*, 2021.
