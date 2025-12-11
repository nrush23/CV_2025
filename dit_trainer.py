"""
Diffusion Scheduler and Training for Pong DiT
Implements DDPM (Denoising Diffusion Probabilistic Model) scheduler
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os


class DDPMScheduler:
    """
    DDPM Scheduler for diffusion process
    """
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        device="cuda"
    ):
        self.num_train_timesteps = num_train_timesteps
        self.device = device
        
        # Beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32).to(device)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps).to(device)
        
        # Alpha schedule
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, original_samples, noise, timesteps):
        """
        Add noise to samples at given timesteps
        Args:
            original_samples: (B, ...) - clean samples
            noise: (B, ...) - noise to add
            timesteps: (B,) - timestep indices
        Returns:
            noisy_samples: (B, ...) - noisy samples
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def remove_noise(self, noisy_samples, noise_pred, timesteps):
        """
        Remove predicted noise from noisy samples
        Args:
            noisy_samples: (B, ...) - noisy samples at timestep t
            noise_pred: (B, ...) - predicted noise
            timesteps: (B,) - timestep indices
        Returns:
            denoised_samples: (B, ...) - predicted clean samples
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(noisy_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        # Predict x_0
        pred_original = (noisy_samples - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
        return pred_original
    
    def step(self, model_output, timestep, sample):
        """
        Predict the sample at the previous timestep (DDPM sampling)
        Args:
            model_output: predicted noise
            timestep: current timestep
            sample: current noisy sample
        Returns:
            prev_sample: predicted sample at t-1
        """
        t = timestep
        
        # 1. Compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_prod_t = 1 - alpha_prod_t
        
        # 2. Compute predicted original sample from predicted noise
        pred_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        
        # 3. Compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = torch.sqrt(alpha_prod_t_prev) * self.betas[t] / beta_prod_t
        current_sample_coeff = torch.sqrt(self.alphas[t]) * (1 - alpha_prod_t_prev) / beta_prod_t
        
        # 4. Compute predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        # 5. Add noise
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = torch.sqrt(self.posterior_variance[t]) * noise
        
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample


class PongLatentDataset(Dataset):
    """
    Dataset for training DiT
    Stores (current_latent, action, next_latent) tuples
    """
    def __init__(self, data):
        """
        Args:
            data: List of tuples (frame_t, action, frame_t+1)
        """
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        latent_t, action, latent_next = self.data[idx]
        return {
            'latent_t': latent_t,
            'action': action,
            'latent_next': latent_next
        }


class DiTTrainer:
    """
    Trainer for DiT model
    """
    def __init__(
        self,
        vae_encoder,
        dit_model,
        device='cuda',
        num_train_timesteps=1000
    ):
        self.vae_encoder = vae_encoder.to(device).eval()
        self.dit = dit_model.to(device)
        self.device = device
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, device=device)
        
        # Freeze VAE encoder
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
    
    def prepare_latents(self, frames, actions):
        """
        Convert frames to latents using VAE encoder
        Args:
            frames: List of numpy arrays (H, W, C)
            actions: List of action indices
        Returns:
            dataset: PongLatentDataset
        """
        print("🔄 Encoding frames to latents...")
        latents = []
        
        with torch.no_grad():
            for frame in tqdm(frames):
                # Convert frame to tensor
                if isinstance(frame, np.ndarray):
                    frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0)
                    if frame_tensor.max() > 1.0:
                        frame_tensor = frame_tensor / 255.0
                    frame_tensor = frame_tensor.to(self.device)
                else:
                    frame_tensor = frame
                
                # Encode
                z, _, _ = self.vae_encoder.encode(frame_tensor)
                latents.append(z.squeeze(0))  # (num_patches, latent_dim)
        
        # Create dataset
        data = []
        for i in range(len(latents) - 1):
            data.append((latents[i], actions[i], latents[i + 1]))
        
        print(f"✅ Created dataset with {len(data)} samples")
        return PongLatentDataset(data)
    
    def train_step(self, batch):
        """Single training step"""
        latent_t = batch['latent_t'].to(self.device)  # (B, num_patches, latent_dim)
        actions = batch['action'].to(self.device)  # (B,)
        latent_next = batch['latent_next'].to(self.device)  # (B, num_patches, latent_dim)
        
        # Sample random timesteps
        bsz = latent_t.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (bsz,), device=self.device
        ).long()
        
        # Add noise to target latent
        noise = torch.randn_like(latent_next)
        noisy_latent = self.scheduler.add_noise(latent_next, noise, timesteps)
        
        # Predict noise
        noise_pred = self.dit(noisy_latent, timesteps, actions, latent_t)
        
        # Compute loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        return loss
    
    def train(self, dataset, epochs=10, batch_size=16, lr=1e-4, save_dir='checkpoints'):
        """
        Train DiT model
        Args:
            dataset: PongLatentDataset
            epochs: number of epochs
            batch_size: batch size
            lr: learning rate
            save_dir: directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        optimizer = torch.optim.AdamW(self.dit.parameters(), lr=lr)
        
        print(f"\n🚀 Starting DiT training...")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {batch_size}")
        print(f"- Learning rate: {lr}")
        print(f"- Dataset size: {len(dataset)}")
        
        self.dit.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                optimizer.zero_grad()
                loss = self.train_step(batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f'dit_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.dit.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Save final model
        final_path = os.path.join(save_dir, 'dit_final.pth')
        torch.save(self.dit.state_dict(), final_path)
        print(f"💾 Saved final model: {final_path}")
        
        return train_losses
    
    def generate(self, initial_latent, actions, num_steps=50):
        """
        Generate sequence of latents given initial latent and actions
        Args:
            initial_latent: (1, num_patches, latent_dim) - starting latent
            actions: List of action indices
            num_steps: number of denoising steps
        Returns:
            latents: List of generated latents
        """
        self.dit.eval()
        latents = [initial_latent]
        
        with torch.no_grad():
            for action in actions:
                action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
                prev_latent = latents[-1]
                
                # Start from pure noise
                latent = torch.randn_like(prev_latent)
                
                # Denoise
                for t in reversed(range(0, self.scheduler.num_train_timesteps, self.scheduler.num_train_timesteps // num_steps)):
                    timestep = torch.tensor([t], device=self.device)
                    
                    # Predict noise
                    noise_pred = self.dit(latent, timestep, action_tensor, prev_latent)
                    
                    # Remove noise
                    latent = self.scheduler.step(noise_pred, t, latent)
                
                latents.append(latent)
        
        return latents[1:]  # Exclude initial latent


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("DiT Trainer Test")
    print("=" * 60)
    
    from vae_encoder import create_vae_encoder
    from dit_model import create_dit
    
    # Create models
    encoder = create_vae_encoder()
    dit = create_dit()
    
    # Create trainer
    trainer = DiTTrainer(encoder, dit, device='cpu')
    
    print("✅ Trainer created successfully")
    
    # Test with dummy data
    dummy_frames = [np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8) for _ in range(10)]
    dummy_actions = np.random.randint(0, 4, 9)
    
    dataset = trainer.prepare_latents(dummy_frames, dummy_actions)
    print(f"Dataset size: {len(dataset)}")
    
    # Test train step
    batch = {
        'latent_t': torch.randn(2, 21*16, 16),
        'action': torch.randint(0, 4, (2,)),
        'latent_next': torch.randn(2, 21*16, 16)
    }
    
    loss = trainer.train_step(batch)
    print(f"✅ Train step successful, loss: {loss.item():.4f}")
