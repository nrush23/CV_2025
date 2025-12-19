"""
OASIS-style Training Pipeline for Pong
Combines VAE and DiT training
"""
import torch
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from vae_encoder import create_vae_encoder
from vae_decoder import create_vae_decoder, ViTVAE
from dit_model import create_dit
from vae_trainer import VAETrainer, PongFrameDataset
from dit_trainer import DiTTrainer, PongLatentDataset


class OASISPipeline:
    """
    Complete OASIS-style pipeline for Pong
    Handles data collection, VAE training, DiT training, and inference
    """
    def __init__(
        self,
        vae_config=None,
        dit_config=None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        print("=" * 70)
        print("🎮 Initializing OASIS Pipeline for Pong")
        print("=" * 70)
        
        self.device = device
        print(f"Device: {device}")
        
        # Create models
        print("\n📦 Creating models...")
        self.encoder = create_vae_encoder(vae_config)
        self.decoder = create_vae_decoder(vae_config)
        self.vae = ViTVAE(self.encoder, self.decoder)
        self.dit = create_dit(dit_config)
        
        print(f"✅ VAE created - Encoder: {sum(p.numel() for p in self.encoder.parameters()):,} params")
        print(f"✅ VAE created - Decoder: {sum(p.numel() for p in self.decoder.parameters()):,} params")
        print(f"✅ DiT created - {sum(p.numel() for p in self.dit.parameters()):,} params")
        
        # Create trainers
        self.vae_trainer = VAETrainer(self.vae, device=device)
        self.dit_trainer = DiTTrainer(self.encoder, self.dit, device=device)
        
        self.trained = False
    
    def collect_pong_data(self, pong_env, num_frames=5000):
        """
        Collect Pong game data
        Args:
            pong_env: Pong environment instance
            num_frames: number of frames to collect
        Returns:
            frames: list of frames
            actions: list of actions
        """
        print(f"\n📊 Collecting {num_frames} frames from Pong...")
        frames, actions = pong_env.simulate(num_frames, COLLECT=True)
        print(f"✅ Collected {len(frames)} frames")
        return frames, actions
    
    def prepare_datasets(self, frames, actions, train_split=0.9):
        """
        Prepare datasets for training
        Args:
            frames: list of frames
            actions: list of actions
            train_split: fraction for training set
        Returns:
            vae_train: VAE training dataset
            vae_val: VAE validation dataset
            actions_train: training actions
            actions_val: validation actions
        """
        print(f"\n🔀 Preparing datasets (train split: {train_split})...")
        
        split_idx = int(len(frames) * train_split)
        
        # Split frames for VAE
        frames_train = frames[:split_idx]
        frames_val = frames[split_idx:]
        
        # Split actions
        actions_train = actions[:split_idx-1]  # -1 because we need pairs
        actions_val = actions[split_idx-1:]
        
        # Create datasets
        vae_train = PongFrameDataset(frames_train)
        vae_val = PongFrameDataset(frames_val)
        
        print(f"✅ VAE train: {len(vae_train)}, val: {len(vae_val)}")
        print(f"✅ Actions train: {len(actions_train)}, val: {len(actions_val)}")
        
        return vae_train, vae_val, actions_train, actions_val, frames_train
    
    def train_vae(
        self,
        train_dataset,
        val_dataset,
        epochs=20,
        batch_size=16,
        lr=1e-4,
        save_dir='checkpoints'
    ):
        """
        Train VAE
        Args:
            train_dataset: training dataset
            val_dataset: validation dataset
            epochs: number of epochs
            batch_size: batch size
            lr: learning rate
            save_dir: save directory
        Returns:
            train_losses: training losses
            val_losses: validation losses
        """
        print("\n" + "=" * 70)
        print("🔧 Step 1: Training VAE")
        print("=" * 70)
        
        train_losses, val_losses = self.vae_trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_dir=save_dir
        )
        
        print("✅ VAE training complete!")
        return train_losses, val_losses
    
    def train_dit(
        self,
        frames_train,
        actions_train,
        epochs=15,
        batch_size=16,
        lr=1e-4,
        save_dir='checkpoints'
    ):
        """
        Train DiT
        Args:
            frames_train: training frames
            actions_train: training actions
            epochs: number of epochs
            batch_size: batch size
            lr: learning rate
            save_dir: save directory
        Returns:
            train_losses: training losses
        """
        print("\n" + "=" * 70)
        print("✨ Step 2: Training DiT")
        print("=" * 70)
        
        # Prepare latent dataset
        dit_dataset = self.dit_trainer.prepare_latents(frames_train, actions_train)
        
        # Train
        train_losses = self.dit_trainer.train(
            dataset=dit_dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_dir=save_dir
        )
        
        print("✅ DiT training complete!")
        self.trained = True
        return train_losses
    
    def train(
        self,
        pong_env,
        num_frames=5000,
        vae_epochs=20,
        dit_epochs=15,
        batch_size=16,
        save_dir='checkpoints'
    ):
        """
        Complete training pipeline
        Args:
            pong_env: Pong environment
            num_frames: number of frames to collect
            vae_epochs: VAE training epochs
            dit_epochs: DiT training epochs
            batch_size: batch size
            save_dir: save directory
        """
        print("\n" + "=" * 70)
        print("🎮 Starting OASIS-Pong Training Pipeline")
        print("=" * 70)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Step 0: Collect data
        frames, actions = self.collect_pong_data(pong_env, num_frames)
        
        # Prepare datasets
        vae_train, vae_val, actions_train, actions_val, frames_train = self.prepare_datasets(
            frames, actions
        )
        
        # Step 1: Train VAE
        if vae_epochs > 0:
            vae_train_losses, vae_val_losses = self.train_vae(
                train_dataset=vae_train,
                val_dataset=vae_val,
                epochs=vae_epochs,
                batch_size=batch_size,
                save_dir=save_dir
            )
        
        # Step 2: Train DiT
        if dit_epochs > 0:
            dit_train_losses = self.train_dit(
                frames_train=frames_train,
                actions_train=actions_train,
                epochs=dit_epochs,
                batch_size=batch_size,
                save_dir=save_dir
            )
        
        print("\n" + "=" * 70)
        print("🎉 Training Complete!")
        print("=" * 70)
        print(f"\n💾 Models saved to {save_dir}/")
    
    def load_weights(self, vae_path=None, dit_path=None):
        """
        Load trained weights
        Args:
            vae_path: path to VAE weights
            dit_path: path to DiT weights
        """
        print("=" * 70)
        print("📂 Loading weights...")
        
        if vae_path:
            self.vae_trainer.load(vae_path)
        
        if dit_path:
            print(f"📂 Loading DiT from {dit_path}...")
            state_dict = torch.load(dit_path, map_location=self.device)
            if 'model_state_dict' in state_dict:
                self.dit.load_state_dict(state_dict['model_state_dict'])
            else:
                self.dit.load_state_dict(state_dict)
            print("✅ DiT loaded successfully")
        
        self.trained = True
        print("=" * 70)
    
    def inference(self, initial_frame, actions, num_denoising_steps=50):
        """
        Generate frames from initial frame and actions
        Args:
            initial_frame: starting frame (numpy array or tensor)
            actions: list of actions
            num_denoising_steps: number of denoising steps per frame
        Returns:
            generated_frames: list of generated frames
        """
        assert self.trained, "Model not trained or loaded!"
        
        print("=" * 70)
        print(f"🎬 Running inference for {len(actions)} frames...")
        print("=" * 70)
        
        self.vae.eval()
        self.dit.eval()
        
        # Encode initial frame
        if isinstance(initial_frame, np.ndarray):
            initial_frame = torch.from_numpy(initial_frame).float().permute(2, 0, 1).unsqueeze(0)
            if initial_frame.max() > 1.0:
                initial_frame = initial_frame / 255.0
        
        initial_frame = initial_frame.to(self.device)
        
        with torch.no_grad():
            initial_latent, _, _ = self.encoder.encode(initial_frame)
        
        # Generate sequence
        generated_latents = self.dit_trainer.generate(
            initial_latent=initial_latent,
            actions=actions,
            num_steps=num_denoising_steps
        )
        
        # Decode latents to frames
        print("🎨 Decoding latents to frames...")
        generated_frames = []
        for latent in generated_latents:
            with torch.no_grad():
                frame = self.decoder(latent)
            # Convert to numpy
            frame = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            generated_frames.append(frame)
        
        print("✅ Inference complete!")
        return generated_frames


def create_oasis_pipeline(vae_config=None, dit_config=None, device=None):
    """
    Factory function to create OASIS pipeline
    
    Usage:
        pipeline = create_oasis_pipeline()
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return OASISPipeline(vae_config=vae_config, dit_config=dit_config, device=device)


# Test code
if __name__ == "__main__":
    print("=" * 70)
    print("OASIS Pipeline Test")
    print("=" * 70)
    
    # Create pipeline
    pipeline = create_oasis_pipeline(device='cpu')
    
    print("\n✅ Pipeline created successfully!")
    print(f"- VAE parameters: {sum(p.numel() for p in pipeline.vae.parameters()):,}")
    print(f"- DiT parameters: {sum(p.numel() for p in pipeline.dit.parameters()):,}")
