"""
VAE Trainer for Pong
Trains the ViT-based VAE to compress and reconstruct frames
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os


class PongFrameDataset(Dataset):
    """
    Dataset for VAE training
    Stores individual frames
    """
    def __init__(self, frames):
        """
        Args:
            frames: List of numpy arrays (H, W, C) or torch tensors
        """
        self.frames = frames
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        # Convert to tensor if numpy array
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame).float().permute(2, 0, 1)  # (C, H, W)
            if frame.max() > 1.0:
                frame = frame / 255.0
        
        return frame


class VAETrainer:
    """
    Trainer for ViT VAE model
    """
    def __init__(self, vae, device='cuda', kld_weight=0.00025):
        self.vae = vae.to(device)
        self.device = device
        self.kld_weight = kld_weight
    
    def compute_loss(self, recon, x, mu, logvar):
        """
        Compute VAE loss
        Args:
            recon: (B, C, H, W) - reconstructed image
            x: (B, C, H, W) - original image
            mu: (B, num_patches, latent_dim)
            logvar: (B, num_patches, latent_dim)
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
        loss = recon_loss + self.kld_weight * kld_loss
        
        return loss, recon_loss, kld_loss
    
    def train_step(self, batch):
        """Single training step"""
        x = batch.to(self.device)  # (B, C, H, W)
        
        # Forward pass
        recon, mu, logvar = self.vae(x)
        
        # Compute loss
        loss, recon_loss, kld_loss = self.compute_loss(recon, x, mu, logvar)
        
        return loss, recon_loss, kld_loss
    
    def validate(self, val_loader):
        """Validation"""
        self.vae.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                loss, recon_loss, kld_loss = self.train_step(batch)
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kld += kld_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_recon = total_recon / len(val_loader)
        avg_kld = total_kld / len(val_loader)
        
        self.vae.train()
        return avg_loss, avg_recon, avg_kld
    
    def train(
        self,
        train_dataset,
        val_dataset=None,
        epochs=20,
        batch_size=16,
        lr=1e-4,
        save_dir='checkpoints',
        save_every=5
    ):
        """
        Train VAE model
        Args:
            train_dataset: training dataset
            val_dataset: validation dataset (optional)
            epochs: number of epochs
            batch_size: batch size
            lr: learning rate
            save_dir: directory to save checkpoints
            save_every: save checkpoint every N epochs
        """
        os.makedirs(save_dir, exist_ok=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if val_dataset else None
        
        optimizer = torch.optim.AdamW(self.vae.parameters(), lr=lr)
        
        print(f"\n🚀 Starting VAE training...")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {batch_size}")
        print(f"- Learning rate: {lr}")
        print(f"- KLD weight: {self.kld_weight}")
        print(f"- Train dataset size: {len(train_dataset)}")
        if val_dataset:
            print(f"- Val dataset size: {len(val_dataset)}")
        
        self.vae.train()
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kld = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                optimizer.zero_grad()
                loss, recon_loss, kld_loss = self.train_step(batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kld += kld_loss.item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'recon': recon_loss.item(),
                    'kld': kld_loss.item()
                })
            
            avg_loss = epoch_loss / len(train_loader)
            avg_recon = epoch_recon / len(train_loader)
            avg_kld = epoch_kld / len(train_loader)
            train_losses.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f} (Recon: {avg_recon:.6f}, KLD: {avg_kld:.6f})")
            
            # Validation
            if val_loader:
                val_loss, val_recon, val_kld = self.validate(val_loader)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.6f} (Recon: {val_recon:.6f}, KLD: {val_kld:.6f})")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(save_dir, 'best_vae.pth')
                    torch.save(self.vae.state_dict(), best_path)
                    print(f"💾 Saved best model: {best_path}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(save_dir, f'vae_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.vae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'best_val_loss': best_val_loss
                }, checkpoint_path)
                print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Save final model
        final_path = os.path.join(save_dir, 'vae_final.pth')
        torch.save(self.vae.state_dict(), final_path)
        print(f"💾 Saved final model: {final_path}")
        
        return train_losses, val_losses
    
    def load(self, path):
        """Load model weights"""
        print(f"📂 Loading VAE from {path}...")
        state_dict = torch.load(path, map_location=self.device)
        if 'model_state_dict' in state_dict:
            self.vae.load_state_dict(state_dict['model_state_dict'])
        else:
            self.vae.load_state_dict(state_dict)
        print("✅ VAE loaded successfully")
    
    def save(self, path):
        """Save model weights"""
        torch.save(self.vae.state_dict(), path)
        print(f"💾 Saved VAE to {path}")


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("VAE Trainer Test")
    print("=" * 60)
    
    from vae_encoder import create_vae_encoder
    from vae_decoder import create_vae_decoder, ViTVAE
    
    # Create VAE
    encoder = create_vae_encoder()
    decoder = create_vae_decoder()
    vae = ViTVAE(encoder, decoder)
    
    # Create trainer
    trainer = VAETrainer(vae, device='cpu')
    
    print("✅ Trainer created successfully")
    
    # Test with dummy data
    dummy_frames = [np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8) for _ in range(20)]
    train_dataset = PongFrameDataset(dummy_frames[:15])
    val_dataset = PongFrameDataset(dummy_frames[15:])
    
    print(f"\nDataset sizes:")
    print(f"- Train: {len(train_dataset)}")
    print(f"- Val: {len(val_dataset)}")
    
    # Test train step
    batch = torch.randn(2, 3, 210, 160)
    loss, recon_loss, kld_loss = trainer.train_step(batch)
    print(f"\n✅ Train step successful")
    print(f"- Total loss: {loss.item():.4f}")
    print(f"- Recon loss: {recon_loss.item():.4f}")
    print(f"- KLD loss: {kld_loss.item():.4f}")
