# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import pandas as pd
import os
import Preprocessing
import HGVAE

# Set environment variable
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # or ':16:8'

# Enable deterministic algorithms
torch.use_deterministic_algorithms(True)


# ==== Main Function =====
if __name__ == '__main__':

    # step 1: Load data starting from Row 1, Column 1 of RNA CSV file
    dfRNA = pd.read_csv('Human_PBMC3k/RNA_Counts.csv')

    # step 2: Preprocess RNA dataset
    RNACellBarcodes, RNAGeneNames, RNATensor = Preprocessing.scRNA_seq(dfRNA)

    # step 3: Load data starting from Row 1, Column 1 of ATAC CSV file
    dfATAC = pd.read_csv('Human_PBMC3k/Atac_Activity_Matrix.csv')

    # step 4: Preprocess ATAC dataset
    ATACCellBarcodes, ATACChromatinRegions, ATACTensor, RNACellBarcodes, RNATensor = Preprocessing.scATAC_seq(dfATAC, RNACellBarcodes, RNATensor)

    # step 5: Construct adjacency matrix for RNA modality
    RNAEdgeIndex, RNAEdgeWeight = HGVAE.BuildAdjacencyMatrix(Type='RNA', Tensor=RNATensor, k=20)

    # step 6: Construct adjacency matrix for ATAC modality
    ATACEdgeIndex, ATACEdgeWeight = HGVAE.BuildAdjacencyMatrix(Type='ATAC', Tensor=ATACTensor, k=30)

    # step 7: Concatenate RNA and ATAC features
    ConcatenatedTensor = HGVAE.Concatenate(RNATensor, ATACTensor)

    # step 8: Train the model
    HGVAE.ModelTrain(ConcatenatedTensor.T, RNACellBarcodes, RNAEdgeIndex, RNAEdgeWeight, ATACEdgeIndex, ATACEdgeWeight, Epochs=300)