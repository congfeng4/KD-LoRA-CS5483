import torch

def decompose_to_decaying_ranks(matrix, start_rank=32):
    # 1. Perform standard SVD
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    
    components = []
    start_idx = 0
    current_rank = start_rank
    
    # 2. Slice the SVD into blocks
    while start_idx < len(S):
        # Determine the end index for this block
        end_idx = min(start_idx + current_rank, len(S))
        
        # Construct the component matrix C_j
        # We slice U, S, and Vh to get just the vectors for this block
        S_block = torch.diag(S[start_idx:end_idx])
        U_block = U[:, start_idx:end_idx]
        Vh_block = Vh[start_idx:end_idx, :]
        
        # Recombine into a dense matrix (Rank = current_rank)
        C_j = U_block @ S_block @ Vh_block
        components.append(C_j)
        
        # Update indices for the next block (exponential decay)
        start_idx = end_idx
        current_rank = max(1, current_rank // 2) # Decay rank by half
        
        if start_idx >= len(S): break
            
    return components

# Example Usage
M = torch.randn(100, 100)
blocks = decompose_to_decaying_ranks(M, start_rank=64)

print(f"Original Rank: {torch.linalg.matrix_rank(M)}")
for i, block in enumerate(blocks):
    print(f"Block {i+1} Rank: {torch.linalg.matrix_rank(block)}")