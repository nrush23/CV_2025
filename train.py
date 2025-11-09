"""
Training script for Pong Autoencoder and Diffusion Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from encoder import create_encoder
from decoder import create_decoder, create_dit
from pong import Pong


# ============================================================
# Dataset
# ============================================================

class PongFrameDataset(Dataset):
    """Pong game frame dataset"""
    def __init__(self, frames, actions=None):
        """
        Args:
            frames: numpy array (N, H, W, C) or list of frames
            actions: numpy array (N,) corresponding actions (for DiT training)
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


# ============================================================
# Part 3: Full Autoencoder (Encoder + Decoder)
# ============================================================

class PongAutoencoder(nn.Module):
    """
    Full Autoencoder for Pong
    Combines Encoder and Decoder
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        """
        Args:
            x: (batch, 3, 210, 160) - Original frame
            
        Returns:
            recon: (batch, 3, 210, 160) - Reconstructed frame
            latent: (batch, n_patches, latent_dim) - Latent representation
        """
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

def create_autoencoder(encoder=None, decoder=None):
    """Creates the full Autoencoder"""
    if encoder is None:
        encoder = create_encoder()
    if decoder is None:
        decoder = create_decoder()
    return PongAutoencoder(encoder, decoder)

# ============================================================
# Stage 1: Train Autoencoder
# ============================================================

class AutoencoderTrainer:
    """Autoencoder Trainer"""
    def __init__(self, encoder=None, decoder=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.autoencoder = create_autoencoder(encoder, decoder).to(device)
        self.optimizer = optim.AdamW(self.autoencoder.parameters(), lr=1e-4, weight_decay=0.01)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader):
        """Trains for one epoch"""
        self.autoencoder.train()
        total_loss = 0
        
        for frames in tqdm(dataloader, desc="Training"):
            frames = frames.to(self.device)
            
            # Forward pass
            reconstructed, latent = self.autoencoder(frames)
            loss = self.criterion(reconstructed, frames)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_dataset, val_dataset, epochs=50, batch_size=32, save_dir='checkpoints'):
        """
        Trains the Autoencoder using the given training and validation datasets, epochs, and batch_size
        Args:
            train_dataset (np.ndarray): Training set of frames of shape (N, 210, 160, 3)
            val_dataset (np.ndarray): Validation set of frames of shape (K, 210, 160, 3)
            epochs (int): Number of epochs.
            batch_size (int): Number of batches.
            save_dir (string): Folder location to save files.
        Returns:
            None
        """
        print("\n" + "=" * 70)
        print("🚀 Start training Autoencoder")
        print("=" * 70)
        
        # Create directory
        os.makedirs(save_dir, exist_ok=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\nModel parameters: {sum(p.numel() for p in self.autoencoder.parameters()):,}")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\n📍 Epoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f"    Train Loss: {train_loss:.6f}")
            print(f"    Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(os.path.join(save_dir, 'best_autoencoder.pth'))
                print(f"    🌟 New best model!")
            
            # Save checkpoint periodically
            if (epoch + 1) % 10 == 0:
                self.save(os.path.join(save_dir, f'autoencoder_epoch_{epoch+1}.pth'))
        
        # Plot training curves
        self.plot_losses(os.path.join(save_dir, 'autoencoder_curves.png'))
    
    def validate(self, dataloader):
        """Validation"""
        self.autoencoder.eval()
        total_loss = 0
        
        with torch.no_grad():
            for frames in tqdm(dataloader, desc="Validating"):
                frames = frames.to(self.device)
                reconstructed, latent = self.autoencoder(frames)
                loss = self.criterion(reconstructed, frames)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save(self, path):
        """Saves the model"""
        torch.save({
            'encoder': self.autoencoder.encoder.state_dict(),
            'decoder': self.autoencoder.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        print(f"✅ Model saved to {path}")
    
    def load(self, path):
        """Loads the model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.autoencoder.encoder.load_state_dict(checkpoint['encoder'])
        self.autoencoder.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"✅ Model loaded from {path}")
    
    def plot_losses(self, save_path='training_curves.png'):
        """Plots the training curves"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Training curves saved to {save_path}")


# ============================================================
# Stage 2: Train DiT
# ============================================================

class DiTTrainer:
    """DiT Trainer"""
    def __init__(self, encoder, dit=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.dit = create_dit().to(device) if dit is None else dit.to(device)
        self.encoder = encoder.to(device)
        self.encoder.eval()  # Encoder is already trained, set to evaluation mode
        
        self.optimizer = optim.AdamW(self.dit.parameters(), lr=1e-4, weight_decay=0.01)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
    
    def add_noise(self, latent, timesteps):
        """Adds noise (simplified diffusion)"""
        noise = torch.randn_like(latent)
        # Simple linear noise schedule
        alpha = 1.0 - timesteps.float().unsqueeze(-1).unsqueeze(-1) / 1000.0
        noisy_latent = alpha * latent + (1 - alpha) * noise
        return noisy_latent, noise
    
    def train_epoch(self, dataloader):
        """Trains for one epoch"""
        self.dit.train()
        total_loss = 0
        
        for frames, actions in tqdm(dataloader, desc="Training DiT"):
            frames = frames.to(self.device)
            actions = actions.to(self.device)
            
            # 1. Encode the frame using the encoder
            with torch.no_grad():
                latent = self.encoder(frames)
            
            # 2. Add noise
            batch_size = frames.shape[0]
            timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)
            noisy_latent, noise = self.add_noise(latent, timesteps)
            
            # 3. DiT predicts the noise
            pred_noise = self.dit(noisy_latent, timesteps, actions)
            
            # 4. Calculate the loss
            loss = self.criterion(pred_noise, noise)
            
            # 5. Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def train(self, dataset, epochs=30, batch_size=32, save_dir='checkpoints'):
        """Trains the DiT"""
        print("\n" + "=" * 70)
        print("🌟 Start training DiT")
        print("=" * 70)
        
        # Create dataset (requires paired frame and action)
        # dataset = PongFrameDataset(frames, actions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # # Create model
        # dit = create_dit()
        # trainer = DiTTrainer(self.encoder, dit)
        
        print(f"\nDiT parameters: {sum(p.numel() for p in self.dit.parameters()):,}")
        print(f"Number of training samples: {len(dataset)}")
        
        # Training loop
        for epoch in range(epochs):
            print(f"\n📍 Epoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch(dataloader)
            print(f"    Train Loss: {train_loss:.6f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % 5 == 0:
                self.save(os.path.join(save_dir, f'dit_epoch_{epoch+1}.pth'))
        
        # Save final model
        self.save(os.path.join(save_dir, 'dit_final.pth'))
    
    def save(self, path):
        """Saves the model"""
        torch.save({
            'dit': self.dit.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_losses': self.train_losses
        }, path)
        print(f"✅ DiT is saved to {path}")
    
    def load(self, path):
        """Loads the model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.dit.load_state_dict(checkpoint['dit'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_losses = checkpoint.get('train_losses', [])
        print(f"✅ DiT is loaded from {path}")


# ============================================================
# Main Training Functions
# ============================================================

def collect_pong_data(num_frames=1000, view=False):
    """Collects Pong game data
    Args:
        num_frames (int): Number of frames to collect from ALE.
        view (bool): Display the frames in real time, default false.
    Returns:
        Tuple (frames, actions):
        - frames (np.ndarray): Np array of the frames with shape (N, 210, 160, 3).
        - actions (np.ndarray): Np array of the actions associated at each frame with shape (N, ).  
    """
    print(f"📊 Collecting {num_frames} frames of Pong data...")

    PONG = Pong(VIEW=view, PLAY=False, EPS=0.01)
    frames, actions = PONG.simulate(num_frames, True)
    
    print(f"✅ Data collection complete.")
    print(f"    - Frames shape: {frames.shape}")
    print(f"    - Actions shape: {actions.shape}")
    
    return frames, actions


def train_autoencoder(frames, epochs=50, batch_size=32, save_dir='checkpoints'):
    """Trains the Autoencoder"""
    print("\n" + "=" * 70)
    print("🚀 Start training Autoencoder")
    print("=" * 70)
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Split training/validation sets
    split_idx = int(len(frames) * 0.9)
    train_frames = frames[:split_idx]
    val_frames = frames[split_idx:]
    
    # Create datasets
    train_dataset = PongFrameDataset(train_frames)
    val_dataset = PongFrameDataset(val_frames)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    encoder = create_encoder()
    decoder = create_decoder()
    trainer = AutoencoderTrainer(encoder, decoder)
    
    print(f"\nModel parameters: {sum(p.numel() for p in trainer.autoencoder.parameters()):,}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n📍 Epoch {epoch+1}/{epochs}")
        
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        
        print(f"    Train Loss: {train_loss:.6f}")
        print(f"    Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save(os.path.join(save_dir, 'best_autoencoder.pth'))
            print(f"    🌟 New best model!")
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            trainer.save(os.path.join(save_dir, f'autoencoder_epoch_{epoch+1}.pth'))
    
    # Plot training curves
    trainer.plot_losses(os.path.join(save_dir, 'autoencoder_curves.png'))
    
    return trainer


def train_dit(frames, actions, encoder, epochs=30, batch_size=32, save_dir='checkpoints'):
    """Trains the DiT"""
    print("\n" + "=" * 70)
    print("🌟 Start training DiT")
    print("=" * 70)
    
    # Create dataset (requires paired frame and action)
    dataset = PongFrameDataset(frames, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    dit = create_dit()
    trainer = DiTTrainer(encoder, dit)
    
    print(f"\nDiT parameters: {sum(p.numel() for p in dit.parameters()):,}")
    print(f"Number of training samples: {len(dataset)}")
    
    # Training loop
    for epoch in range(epochs):
        print(f"\n📍 Epoch {epoch+1}/{epochs}")
        
        train_loss = trainer.train_epoch(dataloader)
        print(f"    Train Loss: {train_loss:.6f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % 5 == 0:
            trainer.save(os.path.join(save_dir, f'dit_epoch_{epoch+1}.pth'))
    
    # Save final model
    trainer.save(os.path.join(save_dir, 'dit_final.pth'))
    
    return trainer

def train(NUM_FRAMES=5000, AUTOENCODER_EPOCHS=20, DIT_EPOCHS=15, BATCH_SIZE=16):
    """
    Train our model on the specified amount of frames, epochs, and batch size.
    Args:
        NUM_FRAMES (int): Number of frames.
        AUTOENCODER_EPOCHS (int): Autoencoder Epoch size.
        DIT_EPOCHS (int): DIT Epoch size.
        BATCH_SIZE (int): Batch size.
    Returns:
        Tuple (Autoencoder, DiT):
            - Autoencoder (Encoder): Autoencoder used during training.
            - DiT (DiT): DiT used during training.
    """
    print("=" * 70)
    print("🎮 Pong AI Training Pipeline")
    print("=" * 70)
    
    
    # Step 1: Collect Data
    print("\n📊 Step 1: Collecting Game Data")
    frames, actions = collect_pong_data(num_frames=NUM_FRAMES, view=False)
    
    # Step 2: Train Autoencoder
    print("\n🔧 Step 2: Training Autoencoder")
    ae_trainer = train_autoencoder(
        frames, 
        epochs=AUTOENCODER_EPOCHS, 
        batch_size=BATCH_SIZE
    )
    
    # Step 3: Train DiT
    print("\n✨ Step 3: Training DiT")
    dit_trainer = train_dit(
        frames, 
        actions, 
        ae_trainer.autoencoder.encoder,
        epochs=DIT_EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    print("\n" + "=" * 70)
    print("🎉 Training complete!")
    print("=" * 70)
    print("\nModel saved to the checkpoints/ directory")
    print("\nNext steps:")
    print("1. Check checkpoints/autoencoder_curves.png to review training performance")
    print("2. Use the trained model to generate new Pong frames")
    print("3. Try using DiT to generate a playable game!")
    return ae_trainer, dit_trainer




# ============================================================
# Main Program
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🎮 Pong AI Training Pipeline")
    print("=" * 70)
    
    # Settings
    NUM_FRAMES = 5000  # Number of frames to collect
    AUTOENCODER_EPOCHS = 20  # Number of Autoencoder training epochs
    DIT_EPOCHS = 15  # Number of DiT training epochs
    BATCH_SIZE = 16
    
    # Step 1: Collect Data
    print("\n📊 Step 1: Collecting Game Data")
    frames, actions = collect_pong_data(num_frames=NUM_FRAMES, view=False)
    
    # Step 2: Train Autoencoder
    print("\n🔧 Step 2: Training Autoencoder")
    ae_trainer = train_autoencoder(
        frames, 
        epochs=AUTOENCODER_EPOCHS, 
        batch_size=BATCH_SIZE
    )
    
    # Step 3: Train DiT
    print("\n✨ Step 3: Training DiT")
    dit_trainer = train_dit(
        frames, 
        actions, 
        ae_trainer.autoencoder.encoder,
        epochs=DIT_EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    print("\n" + "=" * 70)
    print("🎉 Training complete!")
    print("=" * 70)
    print("\nModel saved to the checkpoints/ directory")
    print("\nNext steps:")
    print("1. Check checkpoints/autoencoder_curves.png to review training performance")
    print("2. Use the trained model to generate new Pong frames")
    print("3. Try using DiT to generate a playable game!")
