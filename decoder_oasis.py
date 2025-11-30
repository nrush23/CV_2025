"""
OASIS-style Decoder and DiT for Pong Game
Upgraded with VAE Decoder and Temporal Attention DiT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================
# Part 1: VAE Decoder (UPGRADED)
# ============================================================

class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block"""
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTDecoder(nn.Module):
    """
    Vision Transformer VAE Decoder for Pong (OASIS version)
    Reconstructs the game frame from the latent representation
    
    KEY UPGRADE: Works with VAE latent distribution
    """
    def __init__(
        self,
        img_height=210,
        img_width=160,
        patch_size=10,
        out_channels=3,
        latent_dim=128,
        embed_dim=256,
        depth=6,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        
        # Calculate the number of patches
        self.n_patches_h = img_height // patch_size
        self.n_patches_w = img_width // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w
        
        # 1. Project from latent space back to embed space
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, embed_dim)
        )
        
        # 2. Positional Embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim)
        )
        
        # 3. Transformer Decoder Blocks
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # 4. Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # 5. Project to patch pixel space
        patch_dim = out_channels * patch_size * patch_size
        self.to_pixels = nn.Linear(embed_dim, patch_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        """
        Args:
            z: (batch, latent_dim, n_patches_h, n_patches_w) - Latent representation
            
        Returns:
            image: (batch, channels, height, width) - Reconstructed image
        """
        batch_size = z.shape[0]
        
        # Reshape z: (B, C, H, W) -> (B, H*W, C)
        z = z.flatten(2).transpose(1, 2)  # (B, n_patches, latent_dim)
        
        # 1. Project from latent space back to embed space
        x = self.from_latent(z)  # (batch, n_patches, embed_dim)
        
        # 2. Add positional encoding
        x = x + self.pos_embed
        
        # 3. Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # 4. Layer norm
        x = self.norm(x)
        
        # 5. Project to pixel space
        x = self.to_pixels(x)  # (batch, n_patches, patch_dim)
        
        # 6. Reshape to image
        x = x.reshape(
            batch_size, 
            self.n_patches_h, 
            self.n_patches_w,
            self.out_channels, 
            self.patch_size, 
            self.patch_size
        )
        
        # Permute dimensions: (batch, C, n_patches_h, P, n_patches_w, P)
        x = x.permute(0, 3, 1, 4, 2, 5)
        
        # Merge patches: (batch, C, H, W)
        x = x.reshape(
            batch_size,
            self.out_channels,
            self.n_patches_h * self.patch_size,
            self.n_patches_w * self.patch_size
        )
        
        return x


# ============================================================
# Part 2: OASIS DiT with Temporal Attention (UPGRADED)
# ============================================================

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embeddings"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, timesteps):
        """
        Args:
            timesteps: (B,) - Timestep values
        Returns:
            emb: (B, dim) - Timestep embeddings
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.dim % 2 == 1:  # Zero pad if dim is odd
            emb = F.pad(emb, (0, 1))
        
        return emb


class SpatialAttention(nn.Module):
    """Spatial self-attention within a single frame"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
    def forward(self, x):
        """
        Args:
            x: (B*T, N, D) where B=batch, T=time, N=patches, D=embed_dim
        Returns:
            x: (B*T, N, D)
        """
        x, _ = self.attn(x, x, x)
        return x


class TemporalAttention(nn.Module):
    """
    Temporal attention across frames (OASIS KEY FEATURE)
    Allows the model to attend to previous frames
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
    def forward(self, x, num_frames):
        """
        Args:
            x: (B*T, N, D) - Flattened frames
            num_frames: T - Number of frames in the batch
        Returns:
            x: (B*T, N, D)
        """
        BT, N, D = x.shape
        B = BT // num_frames
        
        # Reshape to (B, T, N, D)
        x = x.reshape(B, num_frames, N, D)
        
        # Permute to (B, N, T, D) for temporal attention
        x = x.permute(0, 2, 1, 3)  # (B, N, T, D)
        
        # Flatten for attention: (B*N, T, D)
        x = x.reshape(B * N, num_frames, D)
        
        # Apply temporal attention
        x, _ = self.attn(x, x, x)
        
        # Reshape back: (B, N, T, D) -> (B, T, N, D) -> (B*T, N, D)
        x = x.reshape(B, N, num_frames, D)
        x = x.permute(0, 2, 1, 3)  # (B, T, N, D)
        x = x.reshape(B * num_frames, N, D)
        
        return x


class DiTBlock(nn.Module):
    """
    OASIS-style DiT Block with interleaved spatial and temporal attention
    KEY UPGRADE: Temporal attention for multi-frame context
    """
    def __init__(
        self, 
        embed_dim=256, 
        num_heads=8, 
        mlp_ratio=4.0,
        dropout=0.1,
        use_temporal=True
    ):
        super().__init__()
        
        self.use_temporal = use_temporal
        
        # Spatial attention
        self.norm1_spatial = nn.LayerNorm(embed_dim)
        self.spatial_attn = SpatialAttention(embed_dim, num_heads, dropout)
        
        # Temporal attention (NEW!)
        if use_temporal:
            self.norm1_temporal = nn.LayerNorm(embed_dim)
            self.temporal_attn = TemporalAttention(embed_dim, num_heads, dropout)
        
        # MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Adaptive layer norm modulation (for conditioning)
        num_params = 6 if use_temporal else 4
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, num_params * embed_dim)
        )
        
    def forward(self, x, condition, num_frames=1):
        """
        Args:
            x: (B*T, N, D) - Input features
            condition: (B*T, D) - Conditioning vector (timestep + action)
            num_frames: Number of frames (for temporal attention)
        Returns:
            x: (B*T, N, D) - Output features
        """
        # Get modulation parameters
        if self.use_temporal:
            (shift_spatial, scale_spatial, 
             shift_temporal, scale_temporal,
             shift_mlp, scale_mlp) = self.adaLN_modulation(condition).chunk(6, dim=-1)
        else:
            (shift_spatial, scale_spatial,
             shift_mlp, scale_mlp) = self.adaLN_modulation(condition).chunk(4, dim=-1)
        
        # Spatial attention with adaptive norm
        x_norm = self.norm1_spatial(x)
        x_norm = x_norm * (1 + scale_spatial.unsqueeze(1)) + shift_spatial.unsqueeze(1)
        x = x + self.spatial_attn(x_norm)
        
        # Temporal attention with adaptive norm (NEW!)
        if self.use_temporal and num_frames > 1:
            x_norm = self.norm1_temporal(x)
            x_norm = x_norm * (1 + scale_temporal.unsqueeze(1)) + shift_temporal.unsqueeze(1)
            x = x + self.temporal_attn(x_norm, num_frames)
        
        # MLP with adaptive norm
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + self.mlp(x_norm)
        
        return x


class DiT(nn.Module):
    """
    OASIS Diffusion Transformer (UPGRADED)
    Features:
    - Interleaved spatial and temporal attention
    - Action conditioning
    - Frame history context
    - Diffusion-based generation
    """
    def __init__(
        self,
        latent_dim=128,
        embed_dim=256,
        depth=8,
        num_heads=8,
        num_actions=6,
        mlp_ratio=4.0,
        dropout=0.1,
        max_frames=4
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.max_frames = max_frames
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Action embedding
        self.action_embed = nn.Embedding(num_actions, embed_dim)
        
        # Latent input projection
        self.input_proj = nn.Linear(latent_dim, embed_dim)
        
        # Positional embedding (will be initialized on first forward)
        self.pos_embed = None
        
        # DiT blocks with temporal attention (KEY UPGRADE)
        self.blocks = nn.ModuleList([
            DiTBlock(
                embed_dim, 
                num_heads, 
                mlp_ratio, 
                dropout,
                use_temporal=True  # Enable temporal attention
            )
            for _ in range(depth)
        ])
        
        # Final norm and projection
        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_proj = nn.Linear(embed_dim, latent_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x_t, timesteps, actions, x_prev=None):
        """
        Args:
            x_t: (B, T, C, H, W) or (B, C, H, W) - Noisy latent frames
            timesteps: (B*T,) or (B,) - Diffusion timesteps
            actions: (B*T,) or (B,) - Actions for each frame
            x_prev: (B, T, C, H, W) or (B, C, H, W) - Previous frame latents (optional)
        
        Returns:
            pred: (B, T, C, H, W) or (B, C, H, W) - Predicted noise or latent
        """
        # Handle both 4D and 5D inputs
        if x_t.dim() == 4:
            # Single frame: (B, C, H, W) -> (B, 1, C, H, W)
            x_t = x_t.unsqueeze(1)
            single_frame = True
        else:
            single_frame = False
        
        B, T, C, H, W = x_t.shape
        
        # Flatten time dimension: (B*T, C, H, W)
        x_t = x_t.reshape(B * T, C, H, W)
        
        # Flatten spatial: (B*T, C, H*W) -> (B*T, H*W, C)
        x = x_t.flatten(2).transpose(1, 2)
        N = x.shape[1]  # Number of patches
        
        # Initialize positional embedding if needed
        if self.pos_embed is None:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, N, self.embed_dim, device=x.device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Project to embed dimension
        x = self.input_proj(x)  # (B*T, N, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Compute conditioning
        # Timestep embedding
        t_emb = self.time_embed(timesteps)  # (B*T, embed_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Action embedding
        a_emb = self.action_embed(actions)  # (B*T, embed_dim)
        
        # Combine conditions
        condition = t_emb + a_emb  # (B*T, embed_dim)
        
        # If we have previous frames, incorporate them
        if x_prev is not None:
            if x_prev.dim() == 4:
                x_prev = x_prev.unsqueeze(1)
            x_prev = x_prev.reshape(B * T, C, H, W)
            x_prev = x_prev.flatten(2).transpose(1, 2)
            x_prev = self.input_proj(x_prev)
            condition = condition + x_prev.mean(dim=1)
        
        # Pass through DiT blocks (with temporal attention)
        for block in self.blocks:
            x = block(x, condition, num_frames=T)
        
        # Final processing
        x = self.final_norm(x)
        x = self.final_proj(x)  # (B*T, N, latent_dim)
        
        # Reshape back to (B, T, C, H, W)
        x = x.transpose(1, 2)  # (B*T, latent_dim, N)
        x = x.reshape(B, T, C, H, W)
        
        # If input was single frame, remove time dimension
        if single_frame:
            x = x.squeeze(1)
        
        return x
    
    def inference(self, start, action):
        """
        Simplified inference for single frame prediction
        NOTE: For full diffusion inference, use the diffusion_utils
        
        Args:
            start: (H, W, C) or (1, C, H, W) - Start frame
            action: int - Action
        Returns:
            prediction: Latent prediction
        """
        # This is a placeholder - actual inference should use diffusion sampling
        # See inference_oasis.py for complete implementation
        raise NotImplementedError(
            "Use OASISInference from inference_oasis.py for proper diffusion-based inference"
        )


# ============================================================
# Part 3: Full Autoencoder (Encoder + Decoder)
# ============================================================

class PongAutoencoder(nn.Module):
    """
    Full VAE Autoencoder for Pong (OASIS version)
    Combines VAE Encoder and Decoder
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x, sample_posterior=True):
        """
        Args:
            x: (batch, 3, 210, 160) - Original frame
            sample_posterior: Whether to sample from posterior
            
        Returns:
            recon: (batch, 3, 210, 160) - Reconstructed frame
            posterior: DiagonalGaussianDistribution - Latent distribution
        """
        posterior = self.encoder(x)
        
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mean
        
        recon = self.decoder(z)
        
        return recon, posterior


# ============================================================
# Helper Functions
# ============================================================

def create_decoder(config=None):
    """Creates the Decoder"""
    default_config = {
        'img_height': 210,
        'img_width': 160,
        'patch_size': 10,
        'out_channels': 3,
        'latent_dim': 128,
        'embed_dim': 256,
        'depth': 6,
        'num_heads': 8,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return ViTDecoder(**default_config)


def create_dit(config=None):
    """Creates the OASIS DiT"""
    default_config = {
        'latent_dim': 128,
        'embed_dim': 256,
        'depth': 8,
        'num_heads': 8,
        'num_actions': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'max_frames': 4
    }
    
    if config:
        default_config.update(config)
    
    return DiT(**default_config)


def create_autoencoder(encoder, decoder=None):
    """Creates the full Autoencoder"""
    if decoder is None:
        decoder = create_decoder()
    return PongAutoencoder(encoder, decoder)


# ============================================================
# Test Code
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("OASIS Decoder & DiT Test")
    print("=" * 70)
    
    # Test Decoder
    print("\n📦 Part 1: Test VAE Decoder")
    print("-" * 70)
    
    decoder = create_decoder()
    decoder.eval()
    
    print(f"Decoder Architecture:")
    print(f"- Latent dim: 128")
    print(f"- Output: 210×160×3")
    print(f"- Transformer layers: 6")
    print(f"- Parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Test decoder with latent
    dummy_latent = torch.randn(2, 128, 21, 16)  # (B, C, H, W)
    print(f"\nInput latent shape: {dummy_latent.shape}")
    
    with torch.no_grad():
        reconstructed = decoder(dummy_latent)
    
    print(f"✅ Decoding successful!")
    print(f"Output shape: {reconstructed.shape}")
    
    # Test DiT
    print("\n\n🌟 Part 2: Test OASIS DiT")
    print("-" * 70)
    
    dit = create_dit()
    dit.eval()
    
    print(f"DiT Architecture:")
    print(f"- Latent dim: 128")
    print(f"- DiT blocks: 8")
    print(f"- With temporal attention: ✅")
    print(f"- Parameters: {sum(p.numel() for p in dit.parameters()):,}")
    
    # Test DiT with sequence
    B, T = 2, 3
    x_noisy = torch.randn(B, T, 128, 21, 16)
    timesteps = torch.randint(0, 1000, (B * T,))
    actions = torch.randint(0, 6, (B * T,))
    x_prev = torch.randn(B, T, 128, 21, 16)
    
    print(f"\nInput shapes:")
    print(f"- Noisy latent: {x_noisy.shape}")
    print(f"- Timesteps: {timesteps.shape}")
    print(f"- Actions: {actions.shape}")
    print(f"- Previous frames: {x_prev.shape}")
    
    with torch.no_grad():
        pred = dit(x_noisy, timesteps, actions, x_prev)
    
    print(f"\n✅ DiT prediction successful!")
    print(f"Output shape: {pred.shape}")
    
    # Test full pipeline
    print("\n\n🔄 Part 3: Test Full Pipeline")
    print("-" * 70)
    
    from encoder import create_encoder
    
    encoder = create_encoder()
    autoencoder = create_autoencoder(encoder, decoder)
    autoencoder.eval()
    
    print(f"Total parameters: {sum(p.numel() for p in autoencoder.parameters()):,}")
    
    dummy_frame = torch.rand(2, 3, 210, 160)
    print(f"\nInput frame shape: {dummy_frame.shape}")
    
    with torch.no_grad():
        reconstructed, posterior = autoencoder(dummy_frame)
    
    print(f"✅ Full pipeline successful!")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"KL divergence: {posterior.kl().mean().item():.4f}")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed! OASIS models ready")
    print("=" * 70)
    print("\n💡 Key Upgrades:")
    print("  ✨ VAE-based encoder/decoder")
    print("  ✨ Temporal attention in DiT")
    print("  ✨ Multi-frame context support")
    print("  ✨ Adaptive layer norm conditioning")
    print("  ✨ Ready for diffusion training")
