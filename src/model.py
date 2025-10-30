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
            self.proj = nn.Conv2d(in_channels, embed_dim, 
                                 kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                         p1=patch_size, p2=patch_size),
                nn.Linear(patch_size * patch_size * in_channels, embed_dim),
            )
        self.use_conv = use_conv
    
    def forward(self, x):
        if self.use_conv:
            x = self.proj(x)
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
        
        if layer_scale_init is not None:
            self.layer_scale = nn.Parameter(torch.ones(embed_dim) * layer_scale_init)
        else:
            self.layer_scale = None
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if self.layer_scale is not None:
            x = x * self.layer_scale
        return x

class FeedForward(nn.Module):
    """MLP module with GELU activation and dropout."""
    def __init__(self, embed_dim, hidden_dim=None, dropout=0.1, layer_scale_init=1e-4):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        if layer_scale_init is not None:
            self.layer_scale = nn.Parameter(torch.ones(embed_dim) * layer_scale_init)
        else:
            self.layer_scale = None
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        
        if self.layer_scale is not None:
            x = x * self.layer_scale
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization."""
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1, 
                 attn_dropout=0.1, drop_path=0.1, layer_scale_init=1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout, layer_scale_init)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout, layer_scale_init)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer with improved training stability and performance."""
    def __init__(self, image_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=256, depth=12, num_heads=8, mlp_ratio=4.0,
                 dropout=0.1, attn_dropout=0.1, drop_path_rate=0.1,
                 layer_scale_init=1e-4, use_conv_patch=True):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, 
                                         embed_dim, use_conv_patch)
        num_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, 
                           attn_dropout, dpr[i], layer_scale_init)
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x