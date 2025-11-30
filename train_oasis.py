"""
OASIS Training Script for Pong (UPGRADED)
Two-stage training: VAE → DiT with Diffusion
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from encoder import create_encoder
from decoder import create_decoder, create_dit, create_autoencoder
from pong import Pong


# ============================================================
# Visualization Helper
# ============================================================

def save_comparison_grid(original, reconstructed, epoch, save_dir, prefix='autoencoder', num_samples=8):
    """
    Saves a comparison grid of original vs reconstructed frames
    """
    os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)
    
    # Take only num_samples
    num_samples = min(num_samples, original.size(0))
    original = original[:num_samples]
    reconstructed = reconstructed[:num_samples]
    
    # Convert to numpy and denormalize
    original_np = original.detach().cpu().permute(0, 2, 3, 1).numpy()
    reconstructed_np = reconstructed.detach().cpu().permute(0, 2, 3, 1).numpy()
    
    # Clip to [0, 1] range
    original_np = np.clip(original_np, 0, 1)
    reconstructed_np = np.clip(reconstructed_np, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    
    # Handle case with single sample
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        # Original frame
        axes[0, i].imshow(original_np[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstructed frame
        axes[1, i].imshow(reconstructed_np[i])
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.suptitle(f'{prefix.capitalize()} - Epoch {epoch}', fontsize=12, y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'visualizations', f'{prefix}_epoch_{epoch:03d}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"    📸 Visualization saved to {save_path}")


# ============================================================
# Dataset (UPGRADED for sequences)
# ============================================================

class PongFrameDataset(Dataset):
    """
    Pong frame dataset for VAE training
    """
    def __init__(self, frames, actions=None):
        """
        Args:
            frames: numpy array (N, H, W, C) or list of frames
            actions: numpy array (N,) corresponding actions (optional)
        """
        self.frames = frames
        self.actions = actions
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        # Convert to tensor and normalize
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame).float()
        
        # Ensure it's in (C, H, W) format
        if frame.shape[0] != 3:
            frame = frame.permute(2, 0, 1)
        
        # Normalize to [0, 1]
        if frame.max() > 1.0:
            frame = frame / 255.0
        
        if self.actions is not None:
            action = self.actions[idx]
            return frame, action
        
        return frame


class PongSequenceDataset(Dataset):
    """
    Pong sequence dataset for DiT training (UPGRADED)
    Returns sequences of frames for temporal modeling
    """
    def __init__(self, frames, actions, sequence_length=4):
        """
        Args:
            frames: (N, H, W, C) - All frames
            actions: (N,) - Actions for each frame
            sequence_length: Number of consecutive frames
        """
        self.frames = frames
        self.actions = actions
        self.sequence_length = sequence_length
        
        # Calculate valid starting indices
        self.valid_indices = list(range(len(frames) - sequence_length))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        # Get sequence of frames
        frame_seq = []
        action_seq = []
        
        for i in range(self.sequence_length):
            frame = self.frames[start_idx + i]
            action = self.actions[start_idx + i]
            
            # Convert to tensor
            if isinstance(frame, np.ndarray):
                frame = torch.from_numpy(frame).float()
            
            # Ensure (C, H, W) format
            if frame.shape[0] != 3:
                frame = frame.permute(2, 0, 1)
            
            # Normalize to [0, 1]
            if frame.max() > 1.0:
                frame = frame / 255.0
            
            frame_seq.append(frame)
            action_seq.append(action)
        
        # Stack into tensors
        frames_tensor = torch.stack(frame_seq)  # (T, C, H, W)
        actions_tensor = torch.tensor(action_seq, dtype=torch.long)  # (T,)
        
        return frames_tensor, actions_tensor


# ============================================================
# Diffusion Utils (simplified)
# ============================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule for diffusion"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class SimpleDiffusion:
    """
    Simplified Diffusion for training
    """
    def __init__(self, timesteps=1000, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        # Pre-compute diffusion schedule
        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """Add noise to x_0"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_0.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise


# ============================================================
# Stage 1: VAE Trainer (UPGRADED)
# ============================================================

class AutoencoderTrainer:
    """
    VAE Autoencoder Trainer (UPGRADED)
    Now includes KL divergence loss
    """
    def __init__(self, encoder=None, decoder=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.autoencoder = create_autoencoder(encoder, decoder).to(device)
        self.optimizer = optim.AdamW(
            self.autoencoder.parameters(), 
            lr=1e-4, 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        self.train_losses = []
        self.val_losses = []
        self.recon_losses = []
        self.kl_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, dataloader, kl_weight=1e-6):
        """Train for one epoch with KL loss"""
        self.autoencoder.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch in tqdm(dataloader, desc="Training VAE"):
            if isinstance(batch, (list, tuple)):
                frames = batch[0]
            else:
                frames = batch
            
            frames = frames.to(self.device)
            
            # Forward pass
            reconstructed, posterior = self.autoencoder(frames, sample_posterior=True)
            
            # Compute losses
            recon_loss = F.mse_loss(reconstructed, frames)
            kl_loss = posterior.kl().mean()
            
            # Total VAE loss
            loss = recon_loss + kl_weight * kl_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        
        self.train_losses.append(avg_loss)
        self.recon_losses.append(avg_recon)
        self.kl_losses.append(avg_kl)
        
        return avg_loss, avg_recon, avg_kl
    
    def validate(self, dataloader, kl_weight=1e-6):
        """Validation"""
        self.autoencoder.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    frames = batch[0]
                else:
                    frames = batch
                
                frames = frames.to(self.device)
                
                reconstructed, posterior = self.autoencoder(frames, sample_posterior=True)
                recon_loss = F.mse_loss(reconstructed, frames)
                kl_loss = posterior.kl().mean()
                loss = recon_loss + kl_weight * kl_loss
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_dataset, val_dataset, epochs=50, batch_size=32, 
              save_dir='checkpoints', visualize_every=5, kl_weight=1e-6):
        """
        Full training loop
        """
        os.makedirs(save_dir, exist_ok=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"\n🚀 Starting VAE Training")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")
        print(f"   Batch size: {batch_size}")
        print(f"   KL weight: {kl_weight}")
        
        for epoch in range(epochs):
            # Train
            train_loss, recon_loss, kl_loss = self.train_epoch(train_loader, kl_weight)
            
            # Validate
            val_loss = self.validate(val_loader, kl_weight)
            
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f} (Recon: {recon_loss:.4f}, KL: {kl_loss:.4f})")
            print(f"  Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save(os.path.join(save_dir, 'best_autoencoder.pth'))
                print(f"  💾 New best model saved!")
            
            # Visualize
            if (epoch + 1) % visualize_every == 0:
                self.visualize(val_loader, epoch+1, save_dir)
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save(os.path.join(save_dir, f'autoencoder_epoch_{epoch+1}.pth'))
        
        # Save final
        self.save(os.path.join(save_dir, 'autoencoder_final.pth'))
        
        # Plot curves
        self.plot_curves(save_dir)
        
        print(f"\n✅ VAE Training Complete!")
    
    def visualize(self, dataloader, epoch, save_dir):
        """Visualize reconstructions"""
        self.autoencoder.eval()
        
        with torch.no_grad():
            batch = next(iter(dataloader))
            if isinstance(batch, (list, tuple)):
                frames = batch[0]
            else:
                frames = batch
            
            frames = frames.to(self.device)
            reconstructed, _ = self.autoencoder(frames, sample_posterior=False)
            
            save_comparison_grid(frames, reconstructed, epoch, save_dir, 'vae', num_samples=8)
    
    def plot_curves(self, save_dir):
        """Plot training curves"""
        plt.figure(figsize=(15, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Val')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(self.recon_losses)
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(self.kl_losses)
        plt.title('KL Divergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'vae_training_curves.png'))
        plt.close()
    
    def save(self, path):
        """Save model"""
        torch.save({
            'autoencoder': self.autoencoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'recon_losses': self.recon_losses,
            'kl_losses': self.kl_losses
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.autoencoder.load_state_dict(checkpoint['autoencoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"✅ VAE loaded from {path}")


# ============================================================
# Stage 2: DiT Trainer with Diffusion (UPGRADED)
# ============================================================

class DiTTrainer:
    """
    DiT Trainer with Diffusion (UPGRADED)
    Trains in latent space with temporal attention
    """
    def __init__(self, encoder, dit=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.encoder = encoder.to(device)
        self.encoder.eval()  # Frozen
        
        if dit is None:
            dit = create_dit()
        self.dit = dit.to(device)
        
        self.diffusion = SimpleDiffusion(timesteps=1000, device=device)
        
        self.optimizer = optim.AdamW(
            self.dit.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        self.train_losses = []
    
    def train_step(self, batch):
        """Single training step"""
        self.optimizer.zero_grad()
        
        frames, actions = batch
        frames = frames.to(self.device)  # (B, T, C, H, W)
        actions = actions.to(self.device)  # (B, T)
        
        B, T, C, H, W = frames.shape
        
        # Encode all frames to latent
        with torch.no_grad():
            frames_flat = frames.reshape(B * T, C, H, W)
            posterior = self.encoder(frames_flat)
            latents = posterior.mean  # Use mean for stability
            
            # Reshape back
            _, latent_dim, h, w = latents.shape
            latents = latents.reshape(B, T, latent_dim, h, w)
        
        # For diffusion, predict next frame from current
        if T > 1:
            x_prev = latents[:, :-1]  # (B, T-1, C, h, w)
            x_target = latents[:, 1:]  # (B, T-1, C, h, w)
            actions_used = actions[:, 1:]  # (B, T-1)
            
            # Flatten batch and time
            x_prev_flat = x_prev.reshape(B * (T-1), latent_dim, h, w)
            x_target_flat = x_target.reshape(B * (T-1), latent_dim, h, w)
            actions_flat = actions_used.reshape(B * (T-1))
            
            # Sample timesteps
            t = torch.randint(0, self.diffusion.timesteps, (B * (T-1),), device=self.device)
            
            # Add noise
            noise = torch.randn_like(x_target_flat)
            x_noisy = self.diffusion.q_sample(x_target_flat, t, noise)
            
            # Predict noise
            x_noisy_5d = x_noisy.unsqueeze(1)  # Add time dim
            x_prev_5d = x_prev_flat.unsqueeze(1)
            
            noise_pred = self.dit(x_noisy_5d, t, actions_flat, x_prev_5d)
            noise_pred = noise_pred.squeeze(1)
            
            # Compute loss
            loss = F.mse_loss(noise_pred, noise)
        else:
            # Single frame case
            x_target = latents
            actions_flat = actions.reshape(B * T)
            
            t = torch.randint(0, self.diffusion.timesteps, (B * T,), device=self.device)
            noise = torch.randn_like(x_target.reshape(B * T, latent_dim, h, w))
            x_noisy = self.diffusion.q_sample(x_target.reshape(B * T, latent_dim, h, w), t, noise)
            
            noise_pred = self.dit(x_noisy.unsqueeze(1), t, actions_flat, None).squeeze(1)
            loss = F.mse_loss(noise_pred, noise)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dit.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, dataset, epochs=30, batch_size=16, save_dir='checkpoints'):
        """Full training loop"""
        os.makedirs(save_dir, exist_ok=True)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        print(f"\n🌟 Starting DiT Training")
        print(f"   Samples: {len(dataset)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Temporal attention: ✅")
        
        for epoch in range(epochs):
            self.dit.train()
            epoch_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_loss += loss
                pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            avg_loss = epoch_loss / len(dataloader)
            self.train_losses.append(avg_loss)
            
            print(f"\nEpoch {epoch+1}: Loss = {avg_loss:.4f}")
            
            # Save periodic
            if (epoch + 1) % 10 == 0:
                self.save(os.path.join(save_dir, f'dit_epoch_{epoch+1}.pth'))
        
        # Save final
        self.save(os.path.join(save_dir, 'dit_final.pth'))
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_losses)
        plt.title('DiT Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'dit_training_curves.png'))
        plt.close()
        
        print(f"\n✅ DiT Training Complete!")
    
    def save(self, path):
        """Save model"""
        torch.save({
            'dit': self.dit.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.train_losses
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.dit.load_state_dict(checkpoint['dit'])
        print(f"✅ DiT loaded from {path}")


# ============================================================
# Main Training Functions
# ============================================================

def collect_pong_data(num_frames=1000, view=False):
    """Collect Pong game data"""
    print(f"📊 Collecting {num_frames} frames of Pong data...")
    
    game = Pong(VIEW=view, PLAY=False, EPS=0.01)
    frames, actions = game.simulate(num_frames, COLLECT=True, CLOSE=True)
    
    print(f"✅ Data collection complete")
    print(f"   Frames shape: {frames.shape}")
    print(f"   Actions shape: {actions.shape}")
    
    return frames, actions


def train(NUM_FRAMES=5000, AUTOENCODER_EPOCHS=50, DIT_EPOCHS=30, BATCH_SIZE=16, 
          SEQUENCE_LENGTH=4, VISUALIZE_EVERY=5, save_dir='checkpoints'):
    """
    Complete OASIS training pipeline
    
    Args:
        NUM_FRAMES: Number of frames to collect
        AUTOENCODER_EPOCHS: VAE training epochs
        DIT_EPOCHS: DiT training epochs
        BATCH_SIZE: Batch size
        SEQUENCE_LENGTH: Sequence length for temporal modeling
        VISUALIZE_EVERY: Visualization frequency
        save_dir: Save directory
    """
    print("=" * 70)
    print("🎮 OASIS Training Pipeline for Pong")
    print("=" * 70)
    
    # Step 1: Collect Data
    print("\n📊 Step 1: Collecting Game Data")
    frames, actions = collect_pong_data(num_frames=NUM_FRAMES, view=False)
    
    # Split train/val
    split_idx = int(len(frames) * 0.9)
    train_frames = frames[:split_idx]
    val_frames = frames[split_idx:]
    train_actions = actions[:split_idx]
    val_actions = actions[split_idx:]
    
    # Step 2: Train VAE
    print("\n🔧 Step 2: Training VAE Autoencoder")
    train_dataset = PongFrameDataset(train_frames)
    val_dataset = PongFrameDataset(val_frames)
    
    encoder = create_encoder()
    decoder = create_decoder()
    ae_trainer = AutoencoderTrainer(encoder, decoder)
    
    ae_trainer.train(
        train_dataset, 
        val_dataset,
        epochs=AUTOENCODER_EPOCHS,
        batch_size=BATCH_SIZE,
        save_dir=save_dir,
        visualize_every=VISUALIZE_EVERY,
        kl_weight=1e-6  # OASIS uses very small KL weight
    )
    
    # Step 3: Train DiT
    print("\n✨ Step 3: Training DiT with Diffusion")
    seq_dataset = PongSequenceDataset(train_frames, train_actions, sequence_length=SEQUENCE_LENGTH)
    
    dit_trainer = DiTTrainer(ae_trainer.autoencoder.encoder)
    dit_trainer.train(
        seq_dataset,
        epochs=DIT_EPOCHS,
        batch_size=BATCH_SIZE,
        save_dir=save_dir
    )
    
    print("\n" + "=" * 70)
    print("🎉 Training Complete!")
    print("=" * 70)
    print(f"\nModels saved to: {save_dir}/")
    print("\n💡 Key Upgrades:")
    print("  ✨ VAE with KL divergence")
    print("  ✨ Temporal attention in DiT")
    print("  ✨ Diffusion-based training")
    print("  ✨ Multi-frame sequences")
    
    return ae_trainer, dit_trainer


# ============================================================
# Main Program
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OASIS Pong Training')
    parser.add_argument('--frames', type=int, default=5000, help='Number of frames')
    parser.add_argument('--vae-epochs', type=int, default=50, help='VAE epochs')
    parser.add_argument('--dit-epochs', type=int, default=30, help='DiT epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--sequence-length', type=int, default=4, help='Sequence length')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Save directory')
    
    args = parser.parse_args()
    
    # Run training
    ae_trainer, dit_trainer = train(
        NUM_FRAMES=args.frames,
        AUTOENCODER_EPOCHS=args.vae_epochs,
        DIT_EPOCHS=args.dit_epochs,
        BATCH_SIZE=args.batch_size,
        SEQUENCE_LENGTH=args.sequence_length,
        save_dir=args.save_dir
    )
