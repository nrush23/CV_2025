import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
    """Splits the image into patches and converts them into embeddings"""
    def __init__(self, img_height=210, img_width=160, patch_size=14, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        
        # Calculate the number of patches (supports non-square images)
        self.n_patches_h = img_height // patch_size
        self.n_patches_w = img_width // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w
        
        # Use a convolutional layer to split patches and project to embed_dim
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.proj(x)  # (batch, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)  # (batch, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer Encoder Block"""
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
        # Multi-head Self-Attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder for Pong frames
    Encodes the game frame into a latent representation
    """
    def __init__(
        self, 
        img_height=210,           # Pong frame height
        img_width=160,            # Pong frame width
        patch_size=14,            # Size of each patch
        in_channels=3,            # RGB channels
        embed_dim=256,            # Embedding dimension
        depth=6,                  # Number of Transformer layers
        num_heads=8,              # Number of Attention heads
        latent_dim=128,           # Final latent vector dimension
        dropout=0.1
    ):
        super().__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        
        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_height, img_width, patch_size, in_channels, embed_dim
        )
        n_patches = self.patch_embed.n_patches
        
        # 2. Positional Embedding (learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches, embed_dim)
        )
        
        # 3. Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # 4. Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # 5. Project to latent space
        self.to_latent = nn.Sequential(
            nn.Linear(embed_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initializes weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width) - Game frame
            
        Returns:
            latent: (batch, n_patches, latent_dim) - Latent representation
        """
        # 1. Split into patches and embed
        x = self.patch_embed(x)  # (batch, n_patches, embed_dim)
        
        # 2. Add positional encoding
        x = x + self.pos_embed
        
        # 3. Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # 4. Layer norm
        x = self.norm(x)
        
        # 5. Project to latent space
        latent = self.to_latent(x)  # (batch, n_patches, latent_dim)
        
        return latent
    
    def encode_frame(self, frame):
        """
        Convenience function: Encodes a single frame
        
        Args:
            frame: numpy array (height, width, channels) or tensor
            
        Returns:
            latent: Latent representation
        """
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame).float()
        
        # Adjust dimensions: (H, W, C) -> (1, C, H, W)
        if frame.dim() == 3:
            frame = frame.permute(2, 0, 1).unsqueeze(0)
        
        # Normalize to [0, 1]
        if frame.max() > 1.0:
            frame = frame / 255.0
        
        with torch.no_grad():
            latent = self.forward(frame)
        
        return latent


# ============ Helper Functions ============

def create_encoder(config=None):
    """
    Factory function to create the Encoder
    
    Usage:
        encoder = create_encoder()
        encoder = create_encoder({'depth': 8, 'latent_dim': 256})
    """
    default_config = {
        'img_height': 210,
        'img_width': 160,
        'patch_size': 10,  # Changed to 10 to ensure divisibility of 210 and 160
        'in_channels': 3,
        'embed_dim': 256,
        'depth': 6,
        'num_heads': 8,
        'latent_dim': 128,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return ViTEncoder(**default_config)


def preprocess_pong_frame(frame):
    """
    Preprocesses the Pong game frame into the Encoder input format
    
    Args:
        frame: numpy array (210, 160, 3) RGB frame obtained from ALE
        
    Returns:
        tensor: (1, 3, 210, 160) Normalized tensor
    """
    # Convert to tensor
    frame_tensor = torch.from_numpy(frame).float()
    
    # Normalize to [0, 1]
    frame_tensor = frame_tensor / 255.0
    
    # Adjust dimensions: (H, W, C) -> (1, C, H, W)
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
    
    return frame_tensor


def encode_pong_observation(encoder, obs):
    """
    Encodes the Pong observation
    
    Args:
        encoder: ViTEncoder instance
        obs: numpy array observation obtained from the environment
        
    Returns:
        latent: Encoded latent representation
    """
    frame_tensor = preprocess_pong_frame(obs)
    
    with torch.no_grad():
        latent = encoder(frame_tensor)
    
    return latent


# ============ Usage Example ============
if __name__ == "__main__":
    print("=" * 60)
    print("Pong Encoder Test")
    print("=" * 60)
    
    # Create encoder
    encoder = create_encoder()
    encoder.eval()
    
    print(f"\n📊 Encoder Architecture Information:")
    print(f"- Input frame size: 210×160×3")
    print(f"- Patch size: 10×10")
    print(f"- Number of patches: {encoder.patch_embed.n_patches_h}×{encoder.patch_embed.n_patches_w} = {encoder.patch_embed.n_patches}")
    print(f"- Embedding dimension: {encoder.embed_dim}")
    print(f"- Number of Transformer layers: {len(encoder.blocks)}")
    print(f"- Latent vector dimension: {encoder.latent_dim}")
    print(f"- Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test encoding
    print("\n" + "=" * 60)
    print("🧪 Encoding Process Test")
    print("=" * 60)
    
    # Test with correct frame size
    dummy_frame = torch.randn(1, 3, 210, 160)
    print(f"\nInput shape: {dummy_frame.shape}")
    
    latent = encoder(dummy_frame)
    print(f"✅ Encoding successful!")
    print(f"Output latent representation shape: {latent.shape}")
    print(f"  - Batch size: {latent.shape[0]}")
    print(f"  - Number of patches: {latent.shape[1]}")
    print(f"  - Latent vector dimension: {latent.shape[2]}")
    
    # Test single frame encoding
    print("\n" + "=" * 60)
    print("🎮 Pong Frame Encoding Test")
    print("=" * 60)
    
    frame_np = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
    print(f"\nSimulated Pong frame shape: {frame_np.shape}")
    
    latent_single = encode_pong_observation(encoder, frame_np)
    print(f"✅ Encoding successful!")
    print(f"Latent representation shape: {latent_single.shape}")
    print(f"\nLatent representation statistics:")
    print(f"  - Mean: {latent_single.mean().item():.4f}")
    print(f"  - Std: {latent_single.std().item():.4f}")
    print(f"  - Min: {latent_single.min().item():.4f}")
    print(f"  - Max: {latent_single.max().item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Encoder is ready")
    print("=" * 60)
