"""
ViT-based Variational Autoencoder Decoder for Pong
Based on OASIS architecture
"""
import torch
import torch.nn as nn
from einops import rearrange


class TransformerBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTVAEDecoder(nn.Module):
    """
    ViT-based VAE Decoder (similar to OASIS)
    Decodes latent space back to image space
    """
    def __init__(
        self,
        img_height=210,
        img_width=160,
        patch_size=10,
        out_channels=3,
        embed_dim=512,
        depth=12,
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
        
        # Project from latent to embedding space
        self.latent_to_embed = nn.Linear(latent_dim, embed_dim)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks (typically deeper than encoder)
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
        
        # Project back to pixel space
        # Each patch token predicts patch_size * patch_size * out_channels pixels
        self.head = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        
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
    
    def forward(self, z):
        """
        Args:
            z: (B, num_patches, latent_dim) - latent representation
        Returns:
            x: (B, C, H, W) - reconstructed image
        """
        B = z.shape[0]
        
        # Project latent to embedding space
        x = self.latent_to_embed(z)  # (B, num_patches, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Project to pixel space
        x = self.head(x)  # (B, num_patches, patch_size * patch_size * out_channels)
        
        # Reshape to image
        # (B, num_patches, patch_size * patch_size * C) -> (B, num_patches_h, num_patches_w, patch_size, patch_size, C)
        x = rearrange(
            x, 
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=self.num_patches_h,
            w=self.num_patches_w,
            p1=self.patch_size,
            p2=self.patch_size,
            c=3
        )
        
        # Apply sigmoid to get values in [0, 1]
        x = torch.sigmoid(x)
        
        return x
    
    def decode_frame(self, z):
        """
        Convenience function to decode a single latent
        Args:
            z: (1, num_patches, latent_dim) or (num_patches, latent_dim)
        Returns:
            frame: (H, W, C) numpy array
        """
        if z.dim() == 2:
            z = z.unsqueeze(0)
        
        with torch.no_grad():
            x = self.forward(z)
        
        # Convert to numpy and adjust dimensions
        frame = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype('uint8')
        
        return frame


class ViTVAE(nn.Module):
    """
    Complete ViT-based VAE (Encoder + Decoder)
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - input image
        Returns:
            recon: (B, C, H, W) - reconstructed image
            mu: (B, num_patches, latent_dim)
            logvar: (B, num_patches, latent_dim)
        """
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def encode(self, x):
        """Encode input to latent"""
        return self.encoder.encode(x)
    
    def decode(self, z):
        """Decode latent to image"""
        return self.decoder(z)
    
    def reconstruct(self, x):
        """Encode and decode"""
        z, _, _ = self.encode(x)
        return self.decode(z)


def vae_loss(recon, x, mu, logvar, kld_weight=0.00025):
    """
    VAE loss function
    Args:
        recon: (B, C, H, W) - reconstructed image
        x: (B, C, H, W) - original image
        mu: (B, num_patches, latent_dim)
        logvar: (B, num_patches, latent_dim)
        kld_weight: weight for KLD loss
    Returns:
        loss: total loss
        recon_loss: reconstruction loss
        kld_loss: KL divergence loss
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
    
    # KL divergence loss
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss = kld_loss / x.size(0)  # Average over batch
    
    # Total loss
    loss = recon_loss + kld_weight * kld_loss
    
    return loss, recon_loss, kld_loss


def create_vae_decoder(config=None):
    """
    Factory function to create VAE decoder
    
    Usage:
        decoder = create_vae_decoder()
        decoder = create_vae_decoder({'depth': 8, 'latent_dim': 32})
    """
    default_config = {
        'img_height': 210,
        'img_width': 160,
        'patch_size': 10,
        'out_channels': 3,
        'embed_dim': 512,
        'depth': 12,
        'num_heads': 8,
        'mlp_ratio': 4.,
        'latent_dim': 16,
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0
    }
    
    if config:
        default_config.update(config)
    
    return ViTVAEDecoder(**default_config)


def create_vae(encoder_config=None, decoder_config=None):
    """
    Create complete VAE model
    """
    from vae_encoder import create_vae_encoder
    
    encoder = create_vae_encoder(encoder_config)
    decoder = create_vae_decoder(decoder_config)
    
    return ViTVAE(encoder, decoder)


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("ViT VAE Decoder Test")
    print("=" * 60)
    
    decoder = create_vae_decoder()
    decoder.eval()
    
    print(f"\n📊 Decoder Information:")
    print(f"- Output: 210×160×3")
    print(f"- Patch size: {decoder.patch_size}")
    print(f"- Number of patches: {decoder.num_patches_h}×{decoder.num_patches_w} = {decoder.num_patches}")
    print(f"- Embedding dim: {decoder.embed_dim}")
    print(f"- Latent dim: {decoder.latent_dim}")
    print(f"- Depth: {len(decoder.blocks)}")
    print(f"- Total parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Test
    dummy_latent = torch.randn(2, 21 * 16, 16)  # (B, num_patches, latent_dim)
    print(f"\nInput latent shape: {dummy_latent.shape}")
    
    output = decoder(dummy_latent)
    
    print(f"✅ Decoding successful!")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test complete VAE
    print("\n" + "=" * 60)
    print("Complete VAE Test")
    print("=" * 60)
    
    from vae_encoder import create_vae_encoder
    encoder = create_vae_encoder()
    vae = ViTVAE(encoder, decoder)
    vae.eval()
    
    dummy_img = torch.randn(2, 3, 210, 160)
    print(f"Input image shape: {dummy_img.shape}")
    
    recon, mu, logvar = vae(dummy_img)
    print(f"✅ VAE forward pass successful!")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"mu shape: {mu.shape}")
    print(f"logvar shape: {logvar.shape}")
    
    # Test loss
    loss, recon_loss, kld_loss = vae_loss(recon, dummy_img, mu, logvar)
    print(f"\nLoss values:")
    print(f"Total loss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KLD loss: {kld_loss.item():.4f}")
