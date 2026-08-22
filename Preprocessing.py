import pandas as pd
import numpy as np
import torch

# Automatically select GPU or CPU
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== 1. Preprocess scRNA-seq dataset =====
def scRNA_seq(dfRNA):
    print("Preprocessing RNA dataset:")

    # Preserve cell barcodes (column names) and gene names (row names)
    RNACellBarcodes = dfRNA.columns[1:].tolist()  # Cell barcodes
    RNAGeneNames = dfRNA.iloc[1:, 0]  # Gene names

    # Convert RNA dataset from Pandas DataFrame to PyTorch Tensor
    RNATensor = torch.tensor(dfRNA.iloc[1:, 1:].values.astype(np.float32), device=Device)

    # Remove genes detected in less than 1% of cells
    print("Removing genes expressed in less than 1% of the cells...")
    mask = (RNATensor > 0).float().mean(1) >= 0.01
    RNATensor = RNATensor[mask]  # Filter RNA dataset by genes
    RNAGeneNames = RNAGeneNames[mask.cpu().numpy()]  # Filter gene names synchronously
    print('After removing genes expressed in less than 1% of the cells, {} genes remaining'.format(RNATensor.shape[0]))

    # Remove cells with fewer than 1% of genes detected
    print("Removing cells in which fewer than 1% of genes were expressed...")
    mask = (RNATensor > 0).float().mean(0) >= 0.01  # Calculate proportion column-wise
    RNATensor = RNATensor[:, mask]  # Filter RNA dataset by cells
    RNACellBarcodes = [c for c, m in zip(RNACellBarcodes, mask.cpu().numpy()) if m]  # Filter cell barcodes synchronously
    print('After removing cells in which fewer than 1% of genes were expressed, {} cells remaining'.format(RNATensor.shape[1]))

    # Select the top 2000 genes with the highest variance across all cells (highly variable genes)
    print("Extracting 2000 highly variable genes...")
    VarGene = RNATensor.var(1)  # Compute variance per gene
    mask = torch.zeros_like(VarGene, dtype=torch.bool)
    mask[torch.topk(VarGene, 2000).indices] = True  # Select the 2000 genes with largest variance

    # Synchronously filter tensor and gene names
    RNATensor = RNATensor[mask]  # (2000, n_cell)
    RNAGeneNames = [g for g, m in zip(RNAGeneNames, mask.cpu().numpy()) if m]

    # Normalization (performed column-wise)
    print("normalizing...")
    total = RNATensor.sum(dim=0, keepdim=True)  # 1 × n_cells
    RNATensor = RNATensor * (1e4 / total)

    # Apply natural log transformation to the 2000 genes of each cell
    print("Performing natural log transformation for smoothing...")
    RNATensor = torch.log1p(RNATensor)

    # Feature scaling
    print("Feature Scaling...")
    CellMeans = RNATensor.mean(dim=0, keepdim=True)
    CellStds = RNATensor.std(dim=0, keepdim=True)
    RNATensor = (RNATensor - CellMeans) / (CellStds + 1e-6)  # Feature scaling

    print("RNA dataset preprocessing completed!\n")

    return RNACellBarcodes, RNAGeneNames, RNATensor


# =============== 2. Preprocess scATAC-seq dataset ================
def scATAC_seq(dfATAC, RNACellBarcodes, RNATensor):
    print("Preprocessing ATAC dataset:")

    # Preserve cell barcodes (column names) and chromatin region names (row names)
    ATACCellBarcodes = dfATAC.columns[1:].tolist()  # Cell barcodes
    ATACChromatinRegions = dfATAC.iloc[1:, 0]  # Chromatin region names

    # Convert dataset from Pandas DataFrame to PyTorch Tensor
    ATACTensor = torch.tensor(dfATAC.iloc[1:, 1:].values.astype(np.float32), device=Device)

    # Remove chromatin regions detected in less than 1% of cells
    print("Removing chromatin regions detected in less than 1% of the cells...")
    mask = (ATACTensor > 0).float().mean(1) >= 0.01
    ATACTensor = ATACTensor[mask]  # Filter ATAC dataset by chromatin regions
    ATACChromatinRegions = ATACChromatinRegions[mask.cpu().numpy()]  # Filter chromatin region names synchronously
    print('After removing chromatin regions detected in less than 1% of the cells, {} regions remaining'.format(ATACTensor.shape[0]))

    # Remove cells with fewer than 1% of chromatin regions detected
    print("Removing cells in which fewer than 1% of chromatin regions were detected...")
    mask = (ATACTensor > 0).float().mean(0) >= 0.01  # Calculate proportion column-wise
    ATACTensor = ATACTensor[:, mask]  # Filter ATAC dataset by cells
    ATACCellBarcodes = [c for c, m in zip(ATACCellBarcodes, mask.cpu().numpy()) if m]  # Filter cell barcodes synchronously
    print('After removing cells in which fewer than 1% of chromatin regions were detected, {} cells remaining'.format(ATACTensor.shape[1]))

    # Select the top 2000 chromatin regions with the highest variance across all cells (highly variable chromatin regions)
    print("Extracting 2000 highly variable chromatin regions...")
    VarChromatin = ATACTensor.var(1)  # Compute variance per chromatin region
    mask = torch.zeros_like(VarChromatin, dtype=torch.bool)
    mask[torch.topk(VarChromatin, 2000).indices] = True  # Select the 2000 chromatin regions with largest variance

    # Synchronously filter tensor and chromatin region names
    ATACTensor = ATACTensor[mask]  # (2000, n_cell)
    ATACChromatinRegions = [g for g, m in zip(ATACChromatinRegions, mask.cpu().numpy()) if m]

    # Normalization (performed column-wise)
    print("Normalizing...")
    total = ATACTensor.sum(dim=0, keepdim=True)  # 1 × n_cells
    ATACTensor = ATACTensor * (1e4 / total)

    # Apply natural log transformation to the 2000 chromatin regions of each cell
    print("Performing natural log transformation...")
    ATACTensor = torch.log1p(ATACTensor)

    # Feature scaling
    print("Feature Scaling...")
    CellMeans = ATACTensor.mean(dim=0, keepdim=True)
    CellStds = ATACTensor.std(dim=0, keepdim=True)
    ATACTensor = (ATACTensor - CellMeans) / (CellStds + 1e-6)  # Feature scaling

    # Reorder ATAC cells to match RNA cell order, and retain only common cells shared by RNA and ATAC data
    print("Re-order ATAC cells to match RNA, and select the common cells present in both RNA and ATAC data...")

    # 1. Identify shared cells while preserving the ordering from RNA data
    CommonCells = pd.Index(RNACellBarcodes).intersection(ATACCellBarcodes, sort=False)  # Maintain RNA cell order

    # 2. Reassign cell barcodes to the shared cell set
    RNACellBarcodes, ATACCellBarcodes = CommonCells.tolist(), CommonCells.tolist()

    # 3. Reorder and subset RNA and ATAC datasets
    RNATensor = RNATensor[:, pd.Index(RNACellBarcodes).get_indexer(CommonCells)]  # Subset following RNA cell order
    ATACTensor = ATACTensor[:, pd.Index(ATACCellBarcodes).get_indexer(CommonCells)]  # Reorder following RNA cell order

    print("ATAC dataset preprocessing completed!\n")

    return ATACCellBarcodes, ATACChromatinRegions, ATACTensor, RNACellBarcodes, RNATensor
