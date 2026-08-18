# HGVAE
Official implementation of HGVAE for single-cell multi-omics integration
# HGVAE: Hierarchical Graph Variational Autoencoder for Single-Cell Multi-Omics Integration

## Overview
We propose HGVAE, a hierarchical graph variational autoencoder framework for single-cell multi-omics integration.
The method constructs cell-level graph structures and captures cross-modality latent relationships to improve cell-type clustering and multi-omics alignment.

## Overview of the repository
main.py             Main execution file (entry script)
Preprocessing.py    Data preprocessing module
HGVAE.py            Model architecture definition and execution pipeline

## Installation
pip install torch==1.13.1+cu116 numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2
