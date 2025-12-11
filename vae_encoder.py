"""
ViT-based Variational Autoencoder Encoder for Pong
Based on OASIS architecture
"""
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with ViT"""
    def __init__(self, img_height=210, img_width=160, patch_size=10, in_channels=3, embed_dim=512):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.num_patches_h = img_height // patch_size
        self.num_patches_w = img_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Conv2d for patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, num_patches_h, num_patches_w)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head Self-Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP as used in Vision Transformer"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTVAEEncoder(nn.Module):
    """
    ViT-based VAE Encoder (similar to OASIS)
    Encodes Pong frames into latent space with mean and logvar for reparameterization
    """
    def __init__(
        self, 
        img_height=210,
        img_width=160,
        patch_size=10,
        in_channels=3,
        embed_dim=512,
        depth=6,
        num_heads=8,
        mlp_ratio=4.,
        latent_dim=16,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.
    ):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_patches_h = img_height // patch_size
        self.num_patches_w = img_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_height, img_width, patch_size, in_channels, embed_dim)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Project to latent space (mean and logvar)
        # Output shape: (B, num_patches, latent_dim)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - input image
        Returns:
            mu: (B, num_patches, latent_dim) - mean
            logvar: (B, num_patches, latent_dim) - log variance
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Project to latent space
        mu = self.fc_mu(x)  # (B, num_patches, latent_dim)
        logvar = self.fc_logvar(x)  # (B, num_patches, latent_dim)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE
        Args:
            mu: (B, num_patches, latent_dim)
            logvar: (B, num_patches, latent_dim)
        Returns:
            z: (B, num_patches, latent_dim) - sampled latent
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def encode(self, x):
        """
        Encode and sample
        Args:
            x: (B, C, H, W)
        Returns:
            z: (B, num_patches, latent_dim)
            mu: (B, num_patches, latent_dim)
            logvar: (B, num_patches, latent_dim)
        """
        mu, logvar = self.forward(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def encode_frame(self, frame):
        """
        Convenience function to encode a single frame
        Args:
            frame: numpy array (H, W, C) or torch tensor
        Returns:
            z: latent representation (1, num_patches, latent_dim)
        """
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame).float()
        
        # Adjust dimensions: (H, W, C) -> (1, C, H, W)
        if frame.dim() == 3:
            frame = frame.permute(2, 0, 1).unsqueeze(0)
        elif frame.size(1) != 3:
            frame = frame.permute(0, 3, 1, 2)
        
        # Normalize to [0, 1]
        if frame.max() > 1.0:
            frame = frame / 255.0
        
        frame = frame.to(next(self.parameters()).device)
        
        with torch.no_grad():
            z, _, _ = self.encode(frame)
        
        return z


def create_vae_encoder(config=None):
    """
    Factory function to create VAE encoder
    
    Usage:
        encoder = create_vae_encoder()
        encoder = create_vae_encoder({'depth': 8, 'latent_dim': 32})
    """
    default_config = {
        'img_height': 210,
        'img_width': 160,
        'patch_size': 10,
        'in_channels': 3,
        'embed_dim': 512,
        'depth': 6,
        'num_heads': 8,
        'mlp_ratio': 4.,
        'latent_dim': 16,
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0
    }
    
    if config:
        default_config.update(config)
    
    return ViTVAEEncoder(**default_config)


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("ViT VAE Encoder Test")
    print("=" * 60)
    
    encoder = create_vae_encoder()
    encoder.eval()
    
    print(f"\n📊 Encoder Information:")
    print(f"- Input: 210×160×3")
    print(f"- Patch size: {encoder.patch_size}")
    print(f"- Number of patches: {encoder.num_patches_h}×{encoder.num_patches_w} = {encoder.num_patches}")
    print(f"- Embedding dim: {encoder.embed_dim}")
    print(f"- Latent dim: {encoder.latent_dim}")
    print(f"- Depth: {len(encoder.blocks)}")
    print(f"- Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test
    dummy_input = torch.randn(2, 3, 210, 160)
    print(f"\nInput shape: {dummy_input.shape}")
    
    mu, logvar = encoder(dummy_input)
    z = encoder.reparameterize(mu, logvar)
    
    print(f"✅ Encoding successful!")
    print(f"mu shape: {mu.shape}")
    print(f"logvar shape: {logvar.shape}")
    print(f"z shape: {z.shape}")
    print(f"Latent spatial dimensions: {encoder.num_patches_h}×{encoder.num_patches_w}")
