"""
Training script for Pong Autoencoder and DiT
訓練 Pong 的 Autoencoder 和 Diffusion Transformer
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
from decoder import create_decoder, create_dit, create_autoencoder


# ============================================================
# Dataset
# ============================================================

class PongFrameDataset(Dataset):
    """Pong 遊戲畫面數據集"""
    def __init__(self, frames, actions=None):
        """
        Args:
            frames: numpy array (N, H, W, C) 或 list of frames
            actions: numpy array (N,) 對應的動作（用於 DiT 訓練）
        """
        self.frames = frames
        self.actions = actions
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        # 轉成 tensor 並正規化
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame).float()
        
        # 確保是 (C, H, W) 格式
        if frame.shape[0] != 3:
            frame = frame.permute(2, 0, 1)
        
        # 正規化到 [0, 1]
        if frame.max() > 1.0:
            frame = frame / 255.0
        
        if self.actions is not None:
            action = self.actions[idx]
            return frame, action
        
        return frame


# ============================================================
# Stage 1: 訓練 Autoencoder
# ============================================================

class AutoencoderTrainer:
    """Autoencoder 訓練器"""
    def __init__(self, encoder, decoder, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.autoencoder = create_autoencoder(encoder, decoder).to(device)
        self.optimizer = optim.AdamW(self.autoencoder.parameters(), lr=1e-4, weight_decay=0.01)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader):
        """訓練一個 epoch"""
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
    
    def validate(self, dataloader):
        """驗證"""
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
        """儲存模型"""
        torch.save({
            'encoder': self.autoencoder.encoder.state_dict(),
            'decoder': self.autoencoder.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        print(f"✅ Model saved to {path}")
    
    def load(self, path):
        """載入模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.autoencoder.encoder.load_state_dict(checkpoint['encoder'])
        self.autoencoder.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"✅ Model loaded from {path}")
    
    def plot_losses(self, save_path='training_curves.png'):
        """繪製訓練曲線"""
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
# Stage 2: 訓練 DiT
# ============================================================

class DiTTrainer:
    """DiT 訓練器"""
    def __init__(self, dit, encoder, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.dit = dit.to(device)
        self.encoder = encoder.to(device)
        self.encoder.eval()  # Encoder 已經訓練好，設為評估模式
        
        self.optimizer = optim.AdamW(self.dit.parameters(), lr=1e-4, weight_decay=0.01)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
    
    def add_noise(self, latent, timesteps):
        """添加噪聲（簡化版 diffusion）"""
        noise = torch.randn_like(latent)
        # 簡單的線性噪聲調度
        alpha = 1.0 - timesteps.float().unsqueeze(-1).unsqueeze(-1) / 1000.0
        noisy_latent = alpha * latent + (1 - alpha) * noise
        return noisy_latent, noise
    
    def train_epoch(self, dataloader):
        """訓練一個 epoch"""
        self.dit.train()
        total_loss = 0
        
        for frames, actions in tqdm(dataloader, desc="Training DiT"):
            frames = frames.to(self.device)
            actions = actions.to(self.device)
            
            # 1. 用 encoder 編碼畫面
            with torch.no_grad():
                latent = self.encoder(frames)
            
            # 2. 添加噪聲
            batch_size = frames.shape[0]
            timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)
            noisy_latent, noise = self.add_noise(latent, timesteps)
            
            # 3. DiT 預測噪聲
            pred_noise = self.dit(noisy_latent, timesteps, actions)
            
            # 4. 計算損失
            loss = self.criterion(pred_noise, noise)
            
            # 5. 更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def save(self, path):
        """儲存模型"""
        torch.save({
            'dit': self.dit.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_losses': self.train_losses
        }, path)
        print(f"✅ DiT is saved to {path}")
    
    def load(self, path):
        """載入模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.dit.load_state_dict(checkpoint['dit'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_losses = checkpoint.get('train_losses', [])
        print(f"✅ DiT is loaded from {path}")


# ============================================================
# 主要訓練函數
# ============================================================

def collect_pong_data(num_frames=1000, view=False):
    """收集 Pong 遊戲數據"""
    print(f"📊 Collecting {num_frames} frames of Pong data...")
    
    from pong import Pong
    import gymnasium
    import ale_py
    
    gymnasium.register_envs(ale_py)
    render_mode = "human" if view else "rgb_array"
    env = gymnasium.make("ALE/Pong-v5", render_mode=render_mode, frameskip=1)
    
    frames = []
    actions = []
    
    obs, info = env.reset()
    
    for i in tqdm(range(num_frames), desc="Collecting frames"):
        # 獲取畫面
        frame = env.unwrapped.get_wrapper_attr("ale").getScreenRGB()
        frames.append(frame)
        
        # 隨機動作
        action = env.action_space.sample()
        actions.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    
    frames = np.array(frames)
    actions = np.array(actions)
    
    print(f"✅ Data collection complete.")
    print(f"   - Frames shape: {frames.shape}")
    print(f"   - Actions shape: {actions.shape}")
    
    return frames, actions


def train_autoencoder(frames, epochs=50, batch_size=32, save_dir='checkpoints'):
    """訓練 Autoencoder"""
    print("\n" + "=" * 70)
    print("🚀 Start training Autoencoder")
    print("=" * 70)
    
    # 創建目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 分割訓練/驗證集
    split_idx = int(len(frames) * 0.9)
    train_frames = frames[:split_idx]
    val_frames = frames[split_idx:]
    
    # 創建數據集
    train_dataset = PongFrameDataset(train_frames)
    val_dataset = PongFrameDataset(val_frames)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 創建模型
    encoder = create_encoder()
    decoder = create_decoder()
    trainer = AutoencoderTrainer(encoder, decoder)
    
    print(f"\nModel parameters: {sum(p.numel() for p in trainer.autoencoder.parameters()):,}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    
    # 訓練
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n📍 Epoch {epoch+1}/{epochs}")
        
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Val Loss: {val_loss:.6f}")
        
        # 儲存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save(os.path.join(save_dir, 'best_autoencoder.pth'))
            print(f"   🌟 New best model!")
        
        # 定期儲存
        if (epoch + 1) % 10 == 0:
            trainer.save(os.path.join(save_dir, f'autoencoder_epoch_{epoch+1}.pth'))
    
    # 繪製訓練曲線
    trainer.plot_losses(os.path.join(save_dir, 'autoencoder_curves.png'))
    
    return trainer


def train_dit(frames, actions, encoder, epochs=30, batch_size=32, save_dir='checkpoints'):
    """訓練 DiT"""
    print("\n" + "=" * 70)
    print("🌟 Start training DiT")
    print("=" * 70)
    
    # 創建數據集（需要配對的 frame 和 action）
    dataset = PongFrameDataset(frames, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 創建模型
    dit = create_dit()
    trainer = DiTTrainer(dit, encoder)
    
    print(f"\nDiT parameters: {sum(p.numel() for p in dit.parameters()):,}")
    print(f"Number of training samples: {len(dataset)}")
    
    # 訓練
    for epoch in range(epochs):
        print(f"\n📍 Epoch {epoch+1}/{epochs}")
        
        train_loss = trainer.train_epoch(dataloader)
        print(f"   Train Loss: {train_loss:.6f}")
        
        # 定期儲存
        if (epoch + 1) % 5 == 0:
            trainer.save(os.path.join(save_dir, f'dit_epoch_{epoch+1}.pth'))
    
    # 儲存最終模型
    trainer.save(os.path.join(save_dir, 'dit_final.pth'))
    
    return trainer


# ============================================================
# 主程式
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🎮 Pong AI Training Pipeline")
    print("=" * 70)
    
    # 設定
    NUM_FRAMES = 5000  # 收集的畫面數量
    AUTOENCODER_EPOCHS = 20  # Autoencoder 訓練輪數
    DIT_EPOCHS = 15  # DiT 訓練輪數
    BATCH_SIZE = 16
    
    # Step 1: 收集數據
    print("\n📊 Step 1: Collecting Game Data")
    frames, actions = collect_pong_data(num_frames=NUM_FRAMES, view=False)
    
    # Step 2: 訓練 Autoencoder
    print("\n🔧 Step 2: Training Autoencoder")
    ae_trainer = train_autoencoder(
        frames, 
        epochs=AUTOENCODER_EPOCHS, 
        batch_size=BATCH_SIZE
    )
    
    # Step 3: 訓練 DiT
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
    print("\n下一步:")
    print("1. Check checkpoints/autoencoder_curves.png to review training performance")
    print("2. Use the trained model to generate new Pong frames")
    print("3. Try using DiT to generate a playable game!")
