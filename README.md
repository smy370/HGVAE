# HGVAE
Official implementation of HGVAE for single-cell multi-omics integration

# A Hierarchical Graph Variational Autoencoder-Based Model for Multimodal Single-Cell Data Integration and Deciphering Cellular Heterogeneityn

# Overview
&nbsp;&nbsp;&nbsp;&nbsp;Single-cell multi-omics technologies enable the simultaneous profiling of cellular transcriptional states and chromatin accessibility, providing powerful approaches for characterizing cell types, gene regulatory mechanisms, and complex biological processes. However, RNA-seq and ATAC-seq data exhibit substantial modality heterogeneity, with distinct distribution patterns while containing complementary biological information. Effectively integrating multimodal features and learning biologically meaningful cellular representations remains a critical challenge.<br>
&nbsp;&nbsp;&nbsp;&nbsp;In this study, we propose HGVAE (Hierarchical Graph Variational Autoencoder), a hierarchical graph variational autoencoder model designed for the integration of single-cell RNA-seq and ATAC-seq multi-omics data. HGVAE constructs modality-specific graph structures and progressively integrates RNA and ATAC modalities through hierarchical graph representation learning, thereby modeling intercellular topological relationships and capturing complementary cross-modal features.HGVAE incorporates hierarchical graph learning, multi-head attention mechanisms, and dual graph reconstruction constraints to achieve adaptive cross-modal information integration and learn stable, biologically interpretable shared latent representations. Based on these representations, HGVAE supports downstream analyses including cell-type identification, cell clustering, and cellular heterogeneity characterization.Evaluation results demonstrate that HGVAE learns more accurate and discriminative cellular representations and outperforms multiple existing methods in cell-type identification tasks. Furthermore, HGVAE enhances biologically meaningful RNA–ATAC regulatory associations and enables the characterization of cellular state transitions and molecular heterogeneity in complex disease contexts.
&nbsp;&nbsp;&nbsp;&nbsp;This repository provides the implementation of HGVAE and related analysis workflows, aiming to offer a reproducible computational tool for single-cell multi-omics integration and bioinformatics research.

# Overview of the repository
```
main.py             Main execution file (entry script)
Preprocessing.py    Data preprocessing module
HGVAE.py            Model architecture definition and execution pipeline
```

# Installation
pip install torch==1.13.1+cu116 numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2
