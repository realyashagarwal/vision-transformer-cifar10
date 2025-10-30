import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=256, use_conv=True):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        
        if use_conv:
            # Use Conv2d for patch embedding (more efficient)
            self.proj = nn.Conv2d(in_channels, embed_dim, 
                                 kernel_size=patch_size, stride=patch_size)
        else:
            # Linear projection of flattened patches
            self.proj = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                         p1=patch_size, p2=patch_size),
                nn.Linear(patch_size * patch_size * in_channels, embed_dim),
            )
        
        self.use_conv = use_conv
    
    def forward(self, x):
        if self.use_conv:
            x = self.proj(x)  # (B, embed_dim, n_patches_h, n_patches_w)
            x = rearrange(x, 'b e h w -> b (h w) e')
        else:
            x = self.proj(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self Attention module with dropout and layer scale."""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, layer_scale_init=1e-4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Layer scale for better training stability
        if layer_scale_init is not None:
            self.layer_scale = nn.Parameter(torch.ones(embed_dim) * layer_scale_init)
        else:
            self.layer_scale = None
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)