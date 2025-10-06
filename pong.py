import gymnasium
import ale_py
import numpy as np
import sys
import keyboard
import torch

# 導入我們的 encoder
from encoder import create_encoder, encode_pong_observation

np.set_printoptions(threshold=sys.maxsize)


class Pong:
    """ Pong interface with ViT Encoder integration """
    def __init__(self, VIEW=True, PLAY=False, use_encoder=False):
        gymnasium.register_envs(ale_py)
        render_mode = "human" if (VIEW or PLAY) else "rgb_array"
        self.env = gymnasium.make("ALE/Pong-v5", render_mode=render_mode, frameskip=1)
        self.env.reset()
        self.PLAY = PLAY
        self.PREV = None
        
        # 初始化 encoder（如果需要）
        self.use_encoder = use_encoder
        if self.use_encoder:
            print("初始化 ViT Encoder...")
            self.encoder = create_encoder()
            self.encoder.eval()  # 設定為評估模式
            print("✅ Encoder 已就緒")
            
            # 用於儲存編碼後的觀察值
            self.encoded_observations = []

    def visualize(self, FRAMES=10):
        """Runs Pong at 30FPS for FRAMES amount of frames"""
        for i in range(FRAMES):
            # 獲取 RGB 畫面
            obs = self.env.unwrapped.get_wrapper_attr("ale").getScreenRGB()
            
            # 如果使用 encoder，編碼當前畫面
            if self.use_encoder:
                latent = encode_pong_observation(self.encoder, obs)
                self.encoded_observations.append(latent)
                
                # 每 100 幀顯示一次編碼資訊
                if i % 100 == 0:
                    print(f"Frame {i}: 潛在表示形狀 = {latent.shape}")
            
            # 獲取動作
            action = self.getPlay() if self.PLAY else self.getAction(obs)

            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                obs, info = self.env.reset()

        self.env.close()
        print(f"Finished running PONG for: {FRAMES} frames")
        
        if self.use_encoder:
            print(f"收集了 {len(self.encoded_observations)} 個編碼觀察值")
            self.save_encoded_observations()

    def save_encoded_observations(self, filename="encoded_observations.pt"):
        """儲存編碼後的觀察值供後續訓練使用"""
        if not self.encoded_observations:
            print("沒有編碼觀察值可以儲存")
            return
        
        # 將所有觀察值堆疊成一個 tensor
        encoded_tensor = torch.cat(self.encoded_observations, dim=0)
        torch.save(encoded_tensor, filename)
        print(f"✅ 已儲存編碼觀察值到 {filename}")
        print(f"   形狀: {encoded_tensor.shape}")

    def getAction(self, obs):
        """
        使用編碼後的觀察值來決定動作
        這裡可以整合您的 AI 模型
        """
        # 選項 1: 使用原始像素找球
        masked = (obs == 236).all(axis=2)
        
        # 選項 2: 如果使用 encoder，可以在潛在空間中分析
        if self.use_encoder and len(self.encoded_observations) > 0:
            # 獲取最新的潛在表示
            latent = self.encoded_observations[-1]
            
            # TODO: 在這裡可以加入您的 AI 決策邏輯
            # 例如：使用潛在表示預測最佳動作
            # action = your_policy_network(latent)
            pass
        
        # 目前還是隨機動作
        return self.env.action_space.sample()

    def getPlay(self):
        """If player controlled, check keyboard inputs for actions"""
        if keyboard.is_pressed('w') or keyboard.is_pressed('up'):
            return 2
        elif keyboard.is_pressed('s') or keyboard.is_pressed('down'):
            return 3
        return 0


# ============ 使用範例 ============
if __name__ == "__main__":
    print("=" * 60)
    print("Pong with ViT Encoder 測試")
    print("=" * 60)
    
    # 創建 Pong 實例並啟用 encoder
    game = Pong(VIEW=True, PLAY=False, use_encoder=True)
    
    # 運行 300 幀以收集編碼數據
    game.visualize(FRAMES=300)
    
    print("\n✅ 測試完成！")