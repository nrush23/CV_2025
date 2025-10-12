import torch
import torch.nn as nn
import numpy as np

#TODO: Add padding if our image dimensions are too small for the patch size
#With a default of 14x14 patches, our height is okay but width is 8 short
PADDING_WIDTH = 8
PADDING_HEIGHT = 0

class PatchEmbedding(nn.Module):
    """將圖像切分成 patches 並轉換成 embeddings"""
    def __init__(self, img_height=210, img_width=160, patch_size=14, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        
        # 計算 patches 數量（支援非正方形圖像）
        self.n_patches_h = img_height // patch_size
        self.n_patches_w = img_width // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w
        
        # 使用卷積層來切分 patches 並投影到 embed_dim
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
    """標準的 Transformer Encoder Block"""
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
    將遊戲畫面編碼成潛在表示
    """
    def __init__(
        self, 
        img_height=210,         # Pong 畫面高度
        img_width=160,          # Pong 畫面寬度
        patch_size=14,          # 每個 patch 的大小
        in_channels=3,          # RGB 通道
        embed_dim=256,          # Embedding 維度
        depth=6,                # Transformer 層數
        num_heads=8,            # Attention heads 數量
        latent_dim=128,         # 最終潛在向量維度
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
        
        # 2. Positional Embedding (可學習)
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
        
        # 5. 投影到潛在空間
        self.to_latent = nn.Sequential(
            nn.Linear(embed_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """初始化權重"""
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
            x: (batch, channels, height, width) - 遊戲畫面
            
        Returns:
            latent: (batch, n_patches, latent_dim) - 潛在表示
        """
        # 1. 切分成 patches 並 embed
        x = self.patch_embed(x)  # (batch, n_patches, embed_dim)
        
        # 2. 加上 positional encoding
        x = x + self.pos_embed
        
        # 3. 通過 Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # 4. Layer norm
        x = self.norm(x)
        
        # 5. 投影到潛在空間
        latent = self.to_latent(x)  # (batch, n_patches, latent_dim)
        
        return latent
    
    def encode_frame(self, frame):
        """
        方便的函數：編碼單一畫面
        
        Args:
            frame: numpy array (height, width, channels) 或 tensor
            
        Returns:
            latent: 潛在表示
        """
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame).float()
        
        # 調整維度: (H, W, C) -> (1, C, H, W)
        if frame.dim() == 3:
            frame = frame.permute(2, 0, 1).unsqueeze(0)
        
        # 正規化到 [0, 1]
        if frame.max() > 1.0:
            frame = frame / 255.0
        
        with torch.no_grad():
            latent = self.forward(frame)
        
        return latent


# ============ 輔助函數 ============

def create_encoder(config=None):
    """
    創建 Encoder 的工廠函數
    
    使用方式:
        encoder = create_encoder()
        encoder = create_encoder({'depth': 8, 'latent_dim': 256})
    """
    default_config = {
        'img_height': 210,
        'img_width': 160,
        'patch_size': 14,
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
    將 Pong 遊戲畫面預處理成 Encoder 輸入格式
    
    Args:
        frame: numpy array (210, 160, 3) 從 ALE 獲得的 RGB 畫面
        
    Returns:
        tensor: (1, 3, 210, 160) 正規化後的 tensor
    """
    # 轉成 tensor
    frame_tensor = torch.from_numpy(frame).float()
    
    # 正規化到 [0, 1]
    frame_tensor = frame_tensor / 255.0
    
    # 調整維度: (H, W, C) -> (1, C, H, W)
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
    
    return frame_tensor


def encode_pong_observation(encoder, obs):
    """
    編碼 Pong 觀察值
    
    Args:
        encoder: ViTEncoder 實例
        obs: numpy array 從環境獲得的觀察值
        
    Returns:
        latent: 編碼後的潛在表示
    """
    frame_tensor = preprocess_pong_frame(obs)
    
    with torch.no_grad():
        latent = encoder(frame_tensor)
    
    return latent


# ============ 使用範例 ============
if __name__ == "__main__":
    print("=" * 60)
    print("Pong Encoder Test")
    print("=" * 60)
    
    # 創建 encoder
    encoder = create_encoder()
    encoder.eval()
    
    print(f"\n📊 Encoder Architecture Information:")
    print(f"- Input frame size: 210×160×3")
    print(f"- Patch 大小: 14×14")
    print(f"- Number of patches: {encoder.patch_embed.n_patches_h}×{encoder.patch_embed.n_patches_w} = {encoder.patch_embed.n_patches}")
    print(f"- Embedding dimension: {encoder.embed_dim}")
    print(f"- Number of Transformer layers: {len(encoder.blocks)}")
    print(f"- Latent vector dimension: {encoder.latent_dim}")
    print(f"- Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # 測試編碼
    print("\n" + "=" * 60)
    print("🧪 Encoding Process Test")
    print("=" * 60)
    
    # 測試正確的畫面大小
    dummy_frame = torch.randn(1, 3, 210, 160)
    print(f"\nInput shape: {dummy_frame.shape}")
    
    latent = encoder(dummy_frame)
    print(f"✅ Encoding successful!")
    print(f"Output latent representation shape: {latent.shape}")
    print(f"  - Batch size: {latent.shape[0]}")
    print(f"  - Number of patches: {latent.shape[1]}")
    print(f"  - Latent vector dimension: {latent.shape[2]}")
    
    # 測試單一畫面編碼
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
