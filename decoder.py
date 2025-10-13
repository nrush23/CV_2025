"""
Decoder for Pong Game - Vision Transformer + DiT Implementation
包含：
1. ViT Decoder: 將潛在表示重建回遊戲畫面
2. DiT: Diffusion Transformer 用於生成新畫面
"""

import torch
import torch.nn as nn
import numpy as np
import math


# ============================================================
# Part 1: Standard ViT Decoder (用於 Autoencoder 訓練)
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
    Vision Transformer Decoder for Pong
    將潛在表示重建回遊戲畫面
    """
    def __init__(
        self,
        img_height=210,
        img_width=160,
        patch_size=14,
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
        
        # 計算 patches 數量
        self.n_patches_h = img_height // patch_size
        self.n_patches_w = img_width // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w
        
        # 1. 從潛在空間投影回 embed 空間
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
        
        # 5. 投影到 patch 像素空間
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
    
    def forward(self, latent):
        """
        Args:
            latent: (batch, n_patches, latent_dim) - 潛在表示
            
        Returns:
            image: (batch, channels, height, width) - 重建的圖像
        """
        batch_size = latent.shape[0]
        
        # 1. 從潛在空間投影回 embed 空間
        x = self.from_latent(latent)  # (batch, n_patches, embed_dim)
        
        # 2. 加上 positional encoding
        x = x + self.pos_embed
        
        # 3. 通過 Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # 4. Layer norm
        x = self.norm(x)
        
        # 5. 投影到像素空間
        x = self.to_pixels(x)  # (batch, n_patches, patch_dim)
        
        # 6. 重組成圖像
        # (batch, n_patches, C*P*P) -> (batch, n_patches_h, n_patches_w, C, P, P)
        x = x.reshape(
            batch_size, 
            self.n_patches_h, 
            self.n_patches_w,
            self.out_channels, 
            self.patch_size, 
            self.patch_size
        )
        
        # 重排維度: (batch, C, n_patches_h, P, n_patches_w, P)
        x = x.permute(0, 3, 1, 4, 2, 5)
        
        # 合併 patches: (batch, C, H, W)
        x = x.reshape(
            batch_size,
            self.out_channels,
            self.n_patches_h * self.patch_size,
            self.n_patches_w * self.patch_size
        )
        
        # Sigmoid 確保輸出在 [0, 1]
        x = torch.sigmoid(x)
        
        return x


# ============================================================
# Part 2: DiT (Diffusion Transformer) - 用於生成新畫面
# ============================================================

class TimestepEmbedding(nn.Module):
    """將 timestep 編碼成向量"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, timesteps):
        """
        Args:
            timesteps: (batch,) - timestep 值
        Returns:
            emb: (batch, dim) - 編碼後的向量
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block
    與標準 Transformer 不同，這個 block 會接收條件信息
    """
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        # Self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
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
        
        # Adaptive Layer Norm (用於條件注入)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim)
        )
    
    def forward(self, x, condition):
        """
        Args:
            x: (batch, n_patches, embed_dim) - 輸入特徵
            condition: (batch, embed_dim) - 條件向量 (timestep + action)
            
        Returns:
            x: (batch, n_patches, embed_dim) - 輸出特徵
        """
        # 計算調製參數
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(condition).chunk(6, dim=1)
        
        # Self-attention with adaptive layer norm
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP with adaptive layer norm
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer for Pong
    在潛在空間中生成新的遊戲畫面
    
    用途：給定當前畫面的潛在表示和玩家動作，預測下一幀
    """
    def __init__(
        self,
        latent_dim=128,
        embed_dim=256,
        depth=8,
        num_heads=8,
        num_actions=6,  # Pong 有 6 個可能的動作
        dropout=0.1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.num_actions = num_actions
        
        # 1. 從潛在空間投影到 embed 空間
        self.latent_proj = nn.Linear(latent_dim, embed_dim)
        
        # 2. Timestep embedding (用於 diffusion)
        self.time_embed = nn.Sequential(
            TimestepEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 3. Action embedding
        self.action_embed = nn.Embedding(num_actions, embed_dim)
        
        # 4. Positional embedding
        # 假設最多 400 個 patches (足夠容納 21×16=336)
        self.pos_embed = nn.Parameter(torch.zeros(1, 400, embed_dim))
        
        # 5. DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # 6. Final norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # 7. 投影回潛在空間
        self.final_proj = nn.Linear(embed_dim, latent_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, latent, timesteps, actions):
        """
        Args:
            latent: (batch, n_patches, latent_dim) - 當前幀的潛在表示
            timesteps: (batch,) - diffusion timesteps
            actions: (batch,) - 玩家動作 (0-5)
            
        Returns:
            pred_latent: (batch, n_patches, latent_dim) - 預測的下一幀潛在表示
        """
        batch_size, n_patches, _ = latent.shape
        
        # 1. 投影到 embed 空間
        x = self.latent_proj(latent)  # (batch, n_patches, embed_dim)
        
        # 2. 加上 positional embedding
        x = x + self.pos_embed[:, :n_patches, :]
        
        # 3. 計算條件向量 (timestep + action)
        time_emb = self.time_embed(timesteps)  # (batch, embed_dim)
        action_emb = self.action_embed(actions)  # (batch, embed_dim)
        condition = time_emb + action_emb  # (batch, embed_dim)
        
        # 4. 通過 DiT blocks
        for block in self.blocks:
            x = block(x, condition)
        
        # 5. Final processing
        x = self.final_norm(x)
        pred_latent = self.final_proj(x)  # (batch, n_patches, latent_dim)
        
        return pred_latent


# ============================================================
# Part 3: 完整的 Autoencoder (Encoder + Decoder)
# ============================================================

class PongAutoencoder(nn.Module):
    """
    完整的 Autoencoder for Pong
    結合 Encoder 和 Decoder
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        """
        Args:
            x: (batch, 3, 210, 160) - 原始畫面
            
        Returns:
            recon: (batch, 3, 210, 160) - 重建的畫面
            latent: (batch, n_patches, latent_dim) - 潛在表示
        """
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


# ============================================================
# 輔助函數
# ============================================================

def create_decoder(config=None):
    """創建 Decoder"""
    default_config = {
        'img_height': 210,
        'img_width': 160,
        'patch_size': 10,  # 改為 10，確保能整除
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
    """創建 DiT"""
    default_config = {
        'latent_dim': 128,
        'embed_dim': 256,
        'depth': 8,
        'num_heads': 8,
        'num_actions': 6,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return DiT(**default_config)


def create_autoencoder(encoder, decoder=None):
    """創建完整的 Autoencoder"""
    if decoder is None:
        decoder = create_decoder()
    return PongAutoencoder(encoder, decoder)


# ============================================================
# 測試代碼
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Pong Decoder & DiT Test")
    print("=" * 70)
    
    # ========== 測試 Decoder ==========
    print("\n📦 Part 1: Test ViT Decoder")
    print("-" * 70)
    
    decoder = create_decoder()
    decoder.eval()
    
    print(f"Decoder Architecture Information:")
    print(f"- Latent space dimension: 128")
    print(f"- Output image size: 210×160×3")
    print(f"- Number of Transformer layers: 6")
    print(f"- Total parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # 測試 decoder
    dummy_latent = torch.randn(2, 336, 128)  # 2 個樣本，336 patches (21×16)
    print(f"\nInput latent representation shape: {dummy_latent.shape}")
    
    with torch.no_grad():
        reconstructed = decoder(dummy_latent)
    
    print(f"✅ Reconstruction successful!")
    print(f"Output image shape: {reconstructed.shape}")
    print(f"Output value range: [{reconstructed.min().item():.3f}, {reconstructed.max().item():.3f}]")
    
    # ========== 測試 DiT ==========
    print("\n\n🌟 Part 2: Testing Diffusion Transformer (DiT)")
    print("-" * 70)
    
    dit = create_dit()
    dit.eval()
    
    print(f"DiT Architecture Information:")
    print(f"- Latent space dimension: 128")
    print(f"- Number of DiT layers: 8")
    print(f"- Supported action count: 6")
    print(f"- Total parameters: {sum(p.numel() for p in dit.parameters()):,}")
    
    # 測試 DiT
    dummy_latent = torch.randn(2, 336, 128)  # 336 patches
    dummy_timesteps = torch.randint(0, 1000, (2,))
    dummy_actions = torch.randint(0, 6, (2,))
    
    print(f"\Input:")
    print(f"- Latent representation shape: {dummy_latent.shape}")
    print(f"- Timesteps: {dummy_timesteps}")
    print(f"- Actions: {dummy_actions}")
    
    with torch.no_grad():
        pred_latent = dit(dummy_latent, dummy_timesteps, dummy_actions)
    
    print(f"\n✅ DiT prediction successful!")
    print(f"Output latent representation shape: {pred_latent.shape}")
    
    # ========== 測試完整 Pipeline ==========
    print("\n\n🔄 Part 3: Testing the Full Autoencoder Pipeline")
    print("-" * 70)
    
    # 需要 encoder，這裡模擬一下
    from encoder import create_encoder
    
    encoder = create_encoder()
    autoencoder = create_autoencoder(encoder, decoder)
    autoencoder.eval()
    
    print(f"Total Autoencoder parameters: {sum(p.numel() for p in autoencoder.parameters()):,}")
    
    # 測試完整流程
    dummy_frame = torch.rand(2, 3, 210, 160)  # 2 個遊戲畫面
    print(f"\nInput original image shape: {dummy_frame.shape}")
    
    with torch.no_grad():
        reconstructed, latent = autoencoder(dummy_frame)
    
    print(f"✅ Full pipeline successful!")
    print(f"Latent representation shape: {latent.shape}")
    print(f"Reconstructed image shape: {reconstructed.shape}")
    
    # 計算重建誤差
    mse = torch.mean((dummy_frame - reconstructed) ** 2)
    print(f"\nRebuilding MSE (untrained): {mse.item():.4f}")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed! Decoder and DiT are ready.")
    print("=" * 70)
    
    print("\n💡 Next steps:")
    print("1. Train the Autoencoder (encoder + decoder)")
    print("2. Collect gameplay data and train the DiT")
    print("3. Use the DiT to generate new game frames")
