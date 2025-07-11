import numpy as np

def sinus_pos_embedding(embed_dims: int, grid_size: int):
    """Generate 2D sinusoidal position embeddings
    
    Args:
        embed_dims (int): Output embedding dimension (must be divisible by 4)
        grid_size (int): Size of the grid (will create grid_size × grid_size positions)
    
    Returns:
        np.ndarray: Position embeddings of shape (grid_size*grid_size, embed_dims)
    """
    # Create grid coordinates
    grid = np.arange(grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(grid, grid)  # Both (grid_size, grid_size)
    
    assert embed_dims % 4 == 0, "Embedding dimension must be divisible by 4 for x/y split"
    coord_dim = embed_dims // 2  
    
    # Frequency calculation (per coordinate)
    i = np.arange(coord_dim // 2, dtype=np.float32)  
    freqs = 1.0 / (10000 ** (2 * i / coord_dim))   
    
    
    x_emb = grid_x[..., None] * freqs  
    x_sin = np.sin(x_emb)
    x_cos = np.cos(x_emb)
    x_embed = np.stack([x_sin, x_cos], axis=-1)   
    
    x_embed = x_embed.reshape(grid_size, grid_size, -1)  
    y_emb = grid_y[..., None] * freqs
    y_embed = np.stack([np.sin(y_emb), np.cos(y_emb)], axis=-1)
    y_embed = y_embed.reshape(grid_size, grid_size, -1)
    
    # Concatenate and flatten
    pos_embed = np.concatenate([x_embed, y_embed], axis=-1)  
    return pos_embed.reshape(-1, embed_dims)  

