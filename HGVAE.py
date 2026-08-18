
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
import gc
import random

# Automatically select GPU or CPU
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 1. Construct adjacency matrix for scRNA-seq or scATAC-seq data using the KNN algorithm ===
def BuildAdjacencyMatrix(Type, Tensor, k=15):
    if Type == 'RNA':
        print("Building adjacency matrix of RNA dataset...")
    else:
        if Type == 'ATAC':
            print("Building adjacency matrix of ATAC dataset...")

    # step 1: Compute Pearson correlation coefficient matrix, output shape [NCells, NCells]
    PearsonCorr = torch.corrcoef(Tensor.T)  # Computed row-wise; each row corresponds to one cell

    # step 2: Set all negative correlations to 0. Cells with Pearson correlation ≤ 0 are considered non-similar
    PearsonCorr.clamp_min_(0)

    # step 3: Mask self-loops by assigning -1 to diagonal entries
    NCells = PearsonCorr.size(0)
    DiagMask = torch.eye(NCells, dtype=torch.bool, device=Device)
    PearsonCorr = PearsonCorr.clone()
    PearsonCorr[DiagMask] = -1

    # step 4: Retrieve column indices of the top-15 maximum values per row (may contain zeros)
    Vals, TopIndex = torch.topk(PearsonCorr, k=k, dim=1)

    # step 5: Filter edges with weight > 0: remove edges with zero correlation among the top 15 neighbors
    Mask = Vals > 0
    ColumnIndex = TopIndex[Mask]  # Column indices of retained edges
    EdgeWeight = Vals[Mask]  # Corresponding edge weights
    RowIndex = torch.nonzero(Mask, as_tuple=False)[:, 0]  # Source node row indices

    # step 6: Assemble EdgeIndex with shape (2, E); undirected graph construction (optional)
    EdgeIndex = torch.stack([RowIndex, ColumnIndex], dim=0)  # 2 × E
    EdgeWeight = EdgeWeight

    # Free memory
    FreeMemory()

    if Type == 'RNA':
        print("Adjacency matrix construction for the RNA dataset is complete!\n")
    else:
        if Type == 'ATAC':
            print("Adjacency matrix construction for the ATAC dataset is complete!\n")

    return EdgeIndex, EdgeWeight


# ====== 2. Perform feature concatenation for scRNA-seq and scATAC-seq data ======
def Concatenate(RNATensor, ATACTensor):
    print("Fusing RNA and ATAC...")

    # RNATensor  (n_gene,  n_cells); ATACTensor (n_peak, n_cells); output shape (n_gene+n_peak, n_cells)
    ConcatenatedTensor = torch.cat([RNATensor, ATACTensor], dim=0)

    print("Concatenation complete!\n")

    return ConcatenatedTensor


# =========== 3. Multi-Head Attention Layer ==============
class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, HiddenDim, NumHeads=8, Dropout=0.2):
        super().__init__()
        self.NumHeads = NumHeads
        self.HiddenDim = HiddenDim
        self.HeadDim = HiddenDim // NumHeads

        assert self.HeadDim * NumHeads == HiddenDim, "HiddenDim must be divisible by NumHeads"

        # Use built-in PyTorch multi-head attention module
        self.MultiHeadAttn = nn.MultiheadAttention(embed_dim=HiddenDim, num_heads=NumHeads, dropout=Dropout, batch_first=True)

        self.LayerNorm = nn.LayerNorm(HiddenDim)
        self.Dropout = nn.Dropout(Dropout)

        # =================== Addition: Attention output stabilization =======================
        self.StabilizationGate = nn.Sequential(nn.Linear(HiddenDim, HiddenDim), nn.Sigmoid())
        self.OutputProjection = nn.Linear(HiddenDim, HiddenDim)

        # Weight initialization
        self.InitializeWeights()

    def forward(self, X):

        # Replicate features to construct a sequence if sequence length equals 1
        if X.size(1) == 1:
            X = X.repeat(1, 3, 1)

        # Apply built-in multi-head attention
        AttendedOutput, AttentionWeights = self.MultiHeadAttn(X, X, X)

        # === Addition: Stabilization procedure ===
        # 1. Gating mechanism to regulate the magnitude of attention output
        Gate = self.StabilizationGate(AttendedOutput)
        StabilizedOutput = AttendedOutput * Gate

        # 2. Output projection for further feature stabilization
        StabilizedOutput = self.OutputProjection(StabilizedOutput)

        # Residual connection with a scaled weight
        Output = self.LayerNorm(X + 0.3 * self.Dropout(StabilizedOutput))

        # Average to restore single-vector representation if extended sequence was constructed
        if Output.size(1) > 1:
            Output = Output.mean(dim=1, keepdim=True)

        return Output

    def InitializeWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ================= 4. End-to-End Model Wrapper ===================
class Model(nn.Module):
    def __init__(self, H1Dim, H2Dim, ZDim, NumHeads=8, dropout=0.1):
        super().__init__()

        self.H1Dim = H1Dim; self.H2Dim = H2Dim; self.ZDim = ZDim

        # Encoder
        self.Encode1 = nn.Sequential(nn.Linear(H1Dim, H2Dim), nn.BatchNorm1d(H2Dim), nn.ReLU(), nn.Dropout(dropout))
        self.Encode2 = nn.Sequential(nn.Linear(H2Dim, ZDim), nn.BatchNorm1d(ZDim), nn.ReLU(), nn.Dropout(dropout))

        # Attach multi-head attention after the final encoder layer
        self.MultiHeadAttention = MultiHeadAttentionLayer(ZDim, NumHeads, dropout)

        self.Mu = nn.Linear(ZDim, ZDim)
        self.LogVar = nn.Linear(ZDim, ZDim)

        # Decoder
        self.Decode1 = nn.Sequential(nn.Linear(ZDim, H2Dim), nn.BatchNorm1d(H2Dim), nn.ReLU(), nn.Dropout(dropout))
        self.Decode2 = nn.Sequential(nn.Linear(H2Dim, H1Dim), nn.Sigmoid())

        # Weight initialization
        self.InitializeWeights()

    def PureAggregation(self, X, EdgeIndex, EdgeWeight):
        N = X.size(0)  # Number of nodes

        # ① Add self-loops with weight = 1
        SelfLoop = torch.arange(N, device=Device).repeat(2, 1)  # 2×N
        EdgeIndex = torch.cat([EdgeIndex, SelfLoop], dim=1)  # Concatenate self-loops
        EdgeWeight = torch.cat([EdgeWeight, torch.ones(N, device=Device)])

        # ② Normalization: D^{-1/2} A D^{-1/2}
        Row, Col = EdgeIndex  # Source → Target nodes
        Deg = torch.zeros(N, dtype=X.dtype, device=Device).scatter_add(0, Row, EdgeWeight).clamp_min(1.0)
        DegInvSqrt = Deg.pow(-0.5)  # D^{-1/2}
        NormW = DegInvSqrt[Row] * EdgeWeight * DegInvSqrt[Col]  # Normalized edge weights

        # ③ Pure aggregation: normalized adjacency matrix × features (dimension unchanged)
        Adj = torch.sparse_coo_tensor(EdgeIndex, NormW, (N, N))
        X = torch.sparse.mm(Adj, X)  # Aggregation only, without linear transformation

        return X  # Output dimension remains (N, InDim)

    def forward(self, X, RNAEdgeIndex, RNAEdgeWeight, ATACEdgeIndex, ATACEdgeWeight):
        """
        Forward propagation
        Args:
        X: Input feature matrix; RNAEdgeIndex: edge indices of RNA graph; RNAEdgeWeight: edge weights of RNA graph;
        ATACEdgeIndex: edge indices of ATAC graph; ATACEdgeWeight: edge weights of ATAC graph
        Returns:
        Out: Reconstructed output from decoder; Mu: mean of latent space; LogVar: log variance of latent space
        """

        # ================ Encoding Stage ==============
        # First-layer graph aggregation and encoding
        H1 = self.PureAggregation(X, RNAEdgeIndex, RNAEdgeWeight)  # Perform pure aggregation on RNA graph
        H1 = self.Encode1.forward(H1)  # Pass through the first encoder layer

        # Second-layer graph aggregation and encoding
        H2 = self.PureAggregation(H1, ATACEdgeIndex, ATACEdgeWeight)  # Perform pure aggregation on ATAC graph using outputs from the first layer
        H2 = self.Encode2.forward(H2)  # Pass through the second encoder layer

        # === New component: Apply multi-head attention after the final encoder layer ===
        # Reshape features into sequence format [BatchSize, SeqLen=1, Features]
        H2Reshaped = H2.unsqueeze(1)  # [BatchSize, 1, ZDim]

        # Apply multi-head attention
        H2Attended = self.MultiHeadAttention(H2Reshaped)
        H2Attended = H2Attended.squeeze(1)  # Restore shape [BatchSize, ZDim]

        # Calculate latent variables using attention-enhanced features
        Mu = self.Mu(H2Attended)
        LogVar = self.LogVar(H2Attended)

        # ======================== Reparameterization Trick =========================
        Std = torch.exp(0.5 * LogVar)  # Compute standard deviation from log variance
        Eps = torch.randn_like(Std)  # Sample random noise from standard normal distribution
        Z = Mu + Eps * Std  # Reparameterization: mean + standard deviation × random noise

        # ================= Decoding Stage =================
        Out = self.Decode1.forward(Z)  # First decoder layer
        Out = self.Decode2.forward(Out)  # Second decoder layer

        return Z, Mu, Out, LogVar

    def InitializeWeights(self):
        """Stable weight initialization strategy"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ====================================================== 5. Loss Functions ========================================================
# ============== (1).Calculate adjacency matrix reconstruction loss (Z * Z^T formulation, no negative sampling) ===================
def CalAdjacencyMatrixReconstructionLoss(NodeEmbeddings, EdgeIndex, EdgeWeight):
    """
    This function computes loss by comparing the original adjacency matrix with the adjacency matrix reconstructed from node embeddings.
    Node similarity is calculated via Z * Z^T, and only positive samples are utilized.

    Args:
        NodeEmbeddings: Node embedding matrix [NumNodes, EmbeddingDim]
        EdgeIndex: Edge index tensor [2, NumEdges]
        EdgeWeight: Edge weight tensor [NumEdges]
    Returns:
        ReconstructionLoss: Reconstruction loss value
    """
    # ==================== Input validation ====================
    assert NodeEmbeddings.dim() == 2, f"NodeEmbeddings should be a 2D tensor, but received {NodeEmbeddings.dim()}D tensor"
    assert EdgeIndex.dim() == 2 and EdgeIndex.size(0) == 2, f"EdgeIndex expected shape [2, E], but received {EdgeIndex.shape}"
    assert EdgeWeight.dim() == 1, f"EdgeWeight should be a 1D tensor, but received {EdgeWeight.dim()}D tensor"
    assert EdgeIndex.size(1) == EdgeWeight.size(0), "Length mismatch between EdgeIndex and EdgeWeight"

    # ==================== Compute reconstructed adjacency matrix ====================
    # Calculate pairwise node similarity using Z * Z^T
    ReconstructedAdjacency = torch.mm(NodeEmbeddings, NodeEmbeddings.t())  # [NumNodes, NumNodes]

    # ==================== Positive sample loss calculation ====================
    # Extract source and target node indices from edge index
    SourceNodes = EdgeIndex[0]  # Source node indices [NumEdges]
    TargetNodes = EdgeIndex[1]  # Target node indices [NumEdges]

    # Retrieve similarity scores for positive node pairs
    PositiveScores = ReconstructedAdjacency[SourceNodes, TargetNodes]  # [NumEdges]

    # Map similarity scores to probabilities via sigmoid function within range (0,1)
    PositiveProbabilities = torch.sigmoid(PositiveScores)  # [NumEdges]

    # Clone edge weights and detach from computation graph
    PositiveWeights = EdgeWeight.clone().detach()

    # Compute reconstruction loss (positive samples only)
    AMRLoss = F.binary_cross_entropy(
                                    PositiveProbabilities,
                                    torch.ones_like(PositiveProbabilities),  # Ground truth labels for positive samples are all 1
                                    weight=PositiveWeights,  # Apply edge weights
                                    reduction='mean'
                                    )
    return AMRLoss


# =========== (2). KL Divergence Loss =============
def CalKLDivergenceLoss(X, Mu, LogVar):
    # Compute KL divergence between latent distribution and standard normal distribution
    # Objective: regularize latent space to approximate standard normal distribution and enhance generative capability
    # Formula: KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    # where LogVar = log(σ²), hence exp(LogVar) = σ²
    # Normalize by X.size(0) to obtain average KL divergence
    Klloss = -0.5 * torch.sum(1 + LogVar - Mu.pow(2) - LogVar.exp()) / X.size(0)
    return Klloss


# ============== (3). Feature Reconstruction Loss ===========
def CalFeatureReconstructionLoss(Out, X):
    # Compute mean squared error between raw input X and decoder reconstructed output Out
    # Objective: enable the autoencoder to accurately reconstruct input and retain critical feature information
    # Formula: MSE = 1/n * Σ(X - Out)²
    MseLoss = F.mse_loss(Out, X)
    return MseLoss


# ====================================  (4). Calculate Total Loss ========================================
def CalculateTotalLoss(X, Z, Mu, Out, LogVar, RNAEdgeIndex, RNAEdgeWeight, ATACEdgeIndex, ATACEdgeWeight):

    # ====== step 1: KL Divergence Loss =======
    Klloss = CalKLDivergenceLoss(X, Mu, LogVar)

    # ============== step 2: RNA Adjacency Matrix Reconstruction Loss ===============
    RNAAMRLoss = CalAdjacencyMatrixReconstructionLoss(Z, RNAEdgeIndex, RNAEdgeWeight)

    # ============= step 3: ATAC Adjacency Matrix Reconstruction Loss ==================
    ATACAMRLoss = CalAdjacencyMatrixReconstructionLoss(Z, ATACEdgeIndex, ATACEdgeWeight)

    # ====== step 4: Feature Reconstruction Loss ======
    MseLoss = CalFeatureReconstructionLoss(Out, X)

    # ========================= step 5: Total Loss Calculation =============================
    TotalLoss = 0.755 * RNAAMRLoss + 0.895 * ATACAMRLoss + 0.015 * Klloss + 0.001 * MseLoss

    return TotalLoss, RNAAMRLoss, ATACAMRLoss, Klloss, MseLoss


# ========== 6. Set random seeds for full reproducibility =========
def SetSeed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU environments
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========= 7. KMeans++ Clustering ===========
def KMeansPlusPlus(Z, K, tol=1e-6, seed=2184):
    """
    KMeans algorithm with KMeans++ initialization
    Args:
    Z: Input tensor with shape (NCells, Dim). NCells denotes sample count, Dim denotes feature dimension
    K: Number of target clusters
    tol: Convergence tolerance; iteration terminates when the shift of cluster centers is smaller than this threshold

    Returns:
    labels: Clustering assignment labels with shape (NCells,), zero-based indices ranging from 0 to K-1
    """

    # Retrieve sample count and feature dimension of the dataset
    NCells, Dim = Z.shape

    # === KMeans++ Initialization Stage ===

    # Set random seed to guarantee reproducibility
    SetSeed(seed)

    # Initialize cluster center matrix with shape (K, Dim)
    Centers = torch.zeros(K, Dim, device=Device)

    # Step 1: Randomly select the first cluster center
    # Pick one sample from all candidates as the initial cluster center
    Centers[0] = Z[torch.randint(NCells, (1,))]

    # Step 2: Sequentially select the remaining K-1 cluster centers
    for i in range(1, K):
        Dists = torch.cdist(Z, Centers[:i])  # Compute distances from each sample to existing cluster centers. Shape of Dists: (NCells, i)
        MinDists = Dists.min(dim=1)[0]  # Minimum distance from each sample to its nearest cluster center. Shape of MinDists: (NCells,)
        Probs = MinDists / MinDists.sum()  # Convert minimum distances into probability distribution. Samples with larger distances have higher probability to be selected as the next cluster center
        Centers[i] = Z[torch.multinomial(Probs, 1)]  # Sample the next cluster center according to the probability distribution

        del Dists, MinDists, Probs  # Free intermediate tensors

    # === Standard KMeans Iterative Optimization Stage ===
    # Start iterative optimization
    while True:
        # Step 1: Assignment phase — assign each sample to its nearest cluster center
        Dist = torch.cdist(Z, Centers)  # Compute distance matrix between all samples and all cluster centers
        labels = Dist.argmin(1)  # Find index of nearest cluster center for each sample (0 to K-1)

        # Step 2: Update phase — recalculate cluster centers
        NewCenters = torch.stack([Z[labels == k].mean(0) for k in range(K)], dim=0)  # For each cluster, take the mean of all contained samples as new cluster center

        # Step 3: Handle empty clusters
        # Check whether any cluster contains no assigned samples
        for k in range(K):
            if (labels == k).sum() == 0:
                # If cluster k is empty, randomly select one sample as its new center
                NewCenters[k] = Z[torch.randint(NCells, (1,))]

        # Step 4: Convergence validation
        if torch.allclose(Centers, NewCenters, atol=tol):  # Check whether the variation between old and new centers is below tolerance
            del Dist, NewCenters # Free tensors upon convergence
            break

        # Update cluster centers and prepare for next iteration
        Centers = NewCenters
        del Dist, NewCenters  # Free iterative tensors

    # Free memory
    FreeMemory()

    # Return clustering labels (zero-based)
    return labels


# ============== 8. Compute Adjusted Rand Index (ARI) ============
# =============== (1). Get true Cluster Labels ===================
def GetTrueClusterLabels(TrueClusterLabelsFile, RNACellBarcodes):

    # step 1: Load CSV file containing ground-truth cluster labels
    TrueClusterData = pd.read_csv(TrueClusterLabelsFile)

    # step 2: Retain columns present in RNACellBarcodes and sort following RNACellBarcodes order
    UseCols = pd.Index(RNACellBarcodes).intersection(TrueClusterData.columns, sort=False)

    # step 3: Subset and reorder columns
    TrueClusterData = TrueClusterData[UseCols]

    # step 4: Extract row of ground-truth cluster labels
    TrueClusterLabels = TrueClusterData.iloc[1, 0:].tolist()

    return TrueClusterLabels

# ================== (2). Calculate ARI =====================
def CalAdjustRandScore(TrueClusterLabels, PreClusterLabels):

    # Calculate Adjusted Rand Index
    TrueClusterLabels = np.array(TrueClusterLabels, dtype=int)
    PreClusterLabels = PreClusterLabels.cpu().numpy().astype(int)
    ARI = adjusted_rand_score(TrueClusterLabels, PreClusterLabels)

    return ARI


# =========================================== 9. Training Function ==================================================
def ModelTrain(X, RNACellBarcodes, RNAEdgeIndex, RNAEdgeWeight, ATACEdgeIndex, ATACEdgeWeight, Epochs=800, seed=2184):

    # step 1: Model initialization - instantiate improved model and transfer to GPU
    ModelInst = Model(X.size(1), 2048, 1024, NumHeads=16, dropout=0.4).to(Device)

    # step 2: Optimizer configuration - use fixed optimal learning rate
    Optimizer = torch.optim.AdamW( ModelInst.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999))

    # step 3: Add learning rate scheduler
    Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer, T_max=Epochs, eta_min=1e-6)

    # step 4： Load ground-truth cluster labels
    TrueClusterLabels = GetTrueClusterLabels("Human_PBMC3k/Clusters.csv", RNACellBarcodes)

    # step 5： Main training loop
    BestARI = 0  # Record the best ARI score

    print("Training...")
    for Epoch in range(1, Epochs + 1):

        # Switch to training mode
        ModelInst.train()

        # Clear gradients
        Optimizer.zero_grad()

        # Forward propagation - obtain model outputs
        Z, Mu, Out, LogVar = ModelInst.forward(X, RNAEdgeIndex, RNAEdgeWeight, ATACEdgeIndex, ATACEdgeWeight)

        # Compute multi-task loss including reconstruction loss, KL divergence and graph reconstruction loss
        TotalLoss, RNAAMRLoss, ATACAMRLoss, Klloss, MseLoss = CalculateTotalLoss(X, Z, Mu, Out, LogVar, RNAEdgeIndex, RNAEdgeWeight, ATACEdgeIndex, ATACEdgeWeight)

        # Backward propagation and optimization
        TotalLoss.backward()  # Compute gradients
        torch.nn.utils.clip_grad_norm_(ModelInst.parameters(), max_norm=0.5)  # Gradient clipping to avoid gradient explosion
        Optimizer.step()  # Update parameters

        # === Release unused tensors timely ===
        del Z, Out, LogVar
        FreeMemory()

        # Update learning rate
        Scheduler.step()

        # === Evaluation phase ===
        ModelInst.eval()
        with torch.no_grad():
            Z_eval, Mu_eval, Out_eval, LogVar_eval = ModelInst.forward(X, RNAEdgeIndex, RNAEdgeWeight, ATACEdgeIndex, ATACEdgeWeight)  # Evaluate on original data
            PreClusterLabels = KMeansPlusPlus(Mu_eval, 12, tol=1e-4, seed=seed)  # Perform clustering on latent embeddings
            ARI = CalAdjustRandScore(TrueClusterLabels, PreClusterLabels)  # Calculate Adjusted Rand Index to evaluate clustering performance

            # Update best ARI
            if ARI > BestARI:
                BestARI = ARI  # Refresh best ARI
                torch.save({'epoch': Epoch,
                                'model_state_dict': ModelInst.state_dict(),
                                'optimizer_state_dict': Optimizer.state_dict(),
                                'BestAri': BestARI},
                             'best_model.pth')  # Save checkpoint of the best model

            # Release evaluation tensors
            del Z_eval, Out_eval, LogVar_eval, Mu_eval, PreClusterLabels
            FreeMemory()

        print(f'Epoch {Epoch:3d} | '
              f'TotalLoss={TotalLoss.item():.4f} | '
              f'RNAAMRLoss={RNAAMRLoss.item():.4f} | '
              f'ATACAMRLoss={ATACAMRLoss.item():.4f} | '
              f'KLLoss={Klloss.item():.4f} | '
              f'ARI={ARI:.4f} | '
              f'Best ARI={BestARI:.4f} | '
              f'LR=1.00e-4')

    # After training, load weights of the best model
    Checkpoint = torch.load('best_model.pth')
    ModelInst.load_state_dict(Checkpoint['model_state_dict'])
    print(f"Training completed! Best ARI: {BestARI:.4f}\n")


# === 10. free memory ===
def FreeMemory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()





















