"""
Diffusion Transformer (DiT) for Pong
Based on OASIS architecture
Action-conditioned frame generation in latent space
"""
import torch
import torch.nn as nn
import math
from einops import rearrange


def modulate(x, shift, scale):
    """Modulation operation for adaptive layer norm"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings
        Args:
            t: (N,) tensor of N indices, one per batch element
            dim: dimension of the output
            max_period: maximum period for sinusoidal encoding
        Returns:
            (N, dim) tensor of positional embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ActionEmbedder(nn.Module):
    """
    Embeds Pong actions into vector representations
    Pong has 4 possible actions: 0 (NOOP), 1 (FIRE), 2 (UP), 3 (DOWN)
    """
    def __init__(self, num_actions=4, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    
    def forward(self, actions):
        """
        Args:
            actions: (B,) tensor of action indices
        Returns:
            (B, hidden_size) tensor of action embeddings
        """
        x = self.embedding(actions)
        x = self.mlp(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm conditioning
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, 
            num_heads, 
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        Args:
            x: (B, N, D) - input sequence
            c: (B, D) - conditioning vector (timestep + action)
        Returns:
            (B, N, D) - output sequence
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention with modulation
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP with modulation
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer for Pong
    Predicts noise in latent space conditioned on:
    - Current timestep
    - Action taken
    - Previous frame latent (optional)
    """
    def __init__(
        self,
        latent_dim=16,
        num_patches_h=21,
        num_patches_w=16,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        num_actions=4,
        learn_sigma=False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.num_patches = num_patches_h * num_patches_w
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.learn_sigma = learn_sigma
        
        # Input projection: latent_dim -> hidden_size
        self.x_embedder = nn.Linear(latent_dim, hidden_size, bias=True)
        
        # Condition embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.action_embedder = ActionEmbedder(num_actions, hidden_size)
        
        # Previous frame embedder (optional, can condition on previous frame)
        self.prev_frame_embedder = nn.Linear(latent_dim, hidden_size, bias=True)
        
        # Positional embedding for spatial locations
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        
        # Final layer
        out_channels = latent_dim * 2 if learn_sigma else latent_dim
        self.final_layer = FinalLayer(hidden_size, out_channels)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize action embedding
        nn.init.normal_(self.action_embedder.embedding.weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, action, prev_latent=None):
        """
        Args:
            x: (B, num_patches, latent_dim) - noisy latent
            t: (B,) - diffusion timesteps
            action: (B,) - action indices
            prev_latent: (B, num_patches, latent_dim) - optional previous frame latent
        Returns:
            (B, num_patches, latent_dim) - predicted noise
        """
        # Embed inputs
        x = self.x_embedder(x) + self.pos_embed  # (B, num_patches, hidden_size)
        
        # Embed conditioning
        t_emb = self.t_embedder(t)  # (B, hidden_size)
        action_emb = self.action_embedder(action)  # (B, hidden_size)
        
        # Combine conditioning
        c = t_emb + action_emb  # (B, hidden_size)
        
        # Optionally add previous frame information
        if prev_latent is not None:
            prev_emb = self.prev_frame_embedder(prev_latent)  # (B, num_patches, hidden_size)
            # Add as extra tokens or average pool
            prev_emb_pooled = prev_emb.mean(dim=1)  # (B, hidden_size)
            c = c + prev_emb_pooled
        
        # Apply DiT blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final layer
        x = self.final_layer(x, c)  # (B, num_patches, latent_dim)
        
        return x


def create_dit(config=None):
    """
    Factory function to create DiT model
    
    Usage:
        dit = create_dit()
        dit = create_dit({'depth': 16, 'hidden_size': 512})
    """
    default_config = {
        'latent_dim': 16,
        'num_patches_h': 21,
        'num_patches_w': 16,
        'hidden_size': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'num_actions': 6,
        'learn_sigma': False
    }
    
    if config:
        default_config.update(config)
    
    return DiT(**default_config)


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("DiT Model Test")
    print("=" * 60)
    
    dit = create_dit()
    dit.eval()
    
    print(f"\n📊 DiT Information:")
    print(f"- Latent dim: {dit.latent_dim}")
    print(f"- Number of patches: {dit.num_patches_h}×{dit.num_patches_w} = {dit.num_patches}")
    print(f"- Hidden size: {dit.hidden_size}")
    print(f"- Depth: {len(dit.blocks)}")
    print(f"- Number of heads: {dit.num_heads}")
    print(f"- Total parameters: {sum(p.numel() for p in dit.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    noisy_latent = torch.randn(batch_size, dit.num_patches, dit.latent_dim)
    timesteps = torch.randint(0, 1000, (batch_size,))
    actions = torch.randint(0, 4, (batch_size,))
    prev_latent = torch.randn(batch_size, dit.num_patches, dit.latent_dim)
    
    print(f"\nInput shapes:")
    print(f"- Noisy latent: {noisy_latent.shape}")
    print(f"- Timesteps: {timesteps.shape}")
    print(f"- Actions: {actions.shape}")
    print(f"- Previous latent: {prev_latent.shape}")
    
    with torch.no_grad():
        output = dit(noisy_latent, timesteps, actions, prev_latent)
    
    print(f"\n✅ Forward pass successful!")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
