
import gymnasium
import ale_py
import numpy as np
import sys
import keyboard
import torch

# 導入我們的 encoder
from encoder import create_encoder, encode_pong_observation

np.set_printoptions(threshold=sys.maxsize)

# Color masks to figure out the ball and paddle positions
BALL_COLOR = 236
PADDLE_COLOR = np.array([92, 186, 92])

#Borders to help us compute the locations of the ball and paddle
BORDER_ROWS = np.array([24,  25,  26,  27,  28,  29,  30,  31,  32,  33, 194,
                       195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209])
TOP_BORDER = 33

#Location of the paddle's center on the X axis
PADDLE_X = 141.5

#Velocity epsilon for determining whether or not we need to track the location of the ball
V_EPS = 0.001


class Pong:
    """ Pong interface for accessing and interacting the ALE Gymnasium Pong model with ViT Encoder integration """

    def __init__(self, VIEW=True, PLAY=False, EPS=0.01, use_encoder=False):
        gymnasium.register_envs(ale_py)
        render_mode = "human" if (VIEW or PLAY) else "rgb_array"
        self.env = gymnasium.make(
            "ALE/Pong-v5", render_mode=render_mode, frameskip=1, repeat_action_probability=0.0)
        self.env.reset()
        self.PLAY = PLAY
        self.PREV = None
        self.EPS = EPS

        # 初始化 encoder（如果需要）
        self.use_encoder = use_encoder
        if self.use_encoder:
            print("Initializing ViT Encoder...")
            self.encoder = create_encoder()
            self.encoder.eval()  # 設定為評估模式
            print("✅ Encoder is ready")

            # 用於儲存編碼後的觀察值
            self.encoded_observations = []

    def visualize(self, FRAMES=10):
        """Runs Pong at 30FPS for FRAMES amount of frames
        Arguments:
        FRAMES -- integer (default 10 frames)
        """
        # Frame rate is 30FPS, so simulation will run for FRAMES/30 seconds
        for i in range(FRAMES):
            # 獲取 RGB 畫面
            obs = self.env.unwrapped.get_wrapper_attr("ale").getScreenRGB()

            # 如果使用 encoder，編碼當前畫面
            if self.use_encoder:
                latent = encode_pong_observation(self.encoder, obs)
                self.encoded_observations.append(latent)

                # 每 100 幀顯示一次編碼資訊
                if i % 100 == 0:
                    print(
                        f"Frame {i}: Latent representation shape = {latent.shape}")

            # 獲取動作
            if (self.PLAY):
                action = self.getPlay()
            else:
                action = self.getAction(obs)
                if (np.random.rand() < self.EPS):
                    action = self.env.action_space.sample()

            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                obs, info = self.env.reset()

        self.env.close()
        print(f"Finished running PONG for: {FRAMES} frames")

        if self.use_encoder:
            print(
                f"{len(self.encoded_observations)} observations were collected and encoded.")
            self.save_encoded_observations()

    def save_encoded_observations(self, filename="encoded_observations.pt"):
        """儲存編碼後的觀察值供後續訓練使用"""
        if not self.encoded_observations:
            print("No observations were collected and encoded.")
            return

        # 將所有觀察值堆疊成一個 tensor
        encoded_tensor = torch.cat(self.encoded_observations, dim=0)
        torch.save(encoded_tensor, filename)
        print(f"✅ Saved encoded observations to {filename}")
        print(f"   Shape: {encoded_tensor.shape}")
    
    # -----------------------------  getACTION PHYSICS LOGIC  ---------------------------------- #
    #                                                                                            #
    #  Calculate where the BALL will intercept the PADDLE's X-COORD and move towards that point  #
    #  Math:   -We use this equation:                                                            #
    #              C_NEW = C_PREV + V_C * TIME (where C == Coordinate, V == Velocity)            #
    #          -Rearrange to solve for the TIME the BALL intercepts the PADDLE X-COORD (141.5):  #
    #              TIME = (X_NEW - X_PREV) / V_X                                                 #
    #          -Now solve for the BALL's Y-COORD at that TIME:                                   #
    #              Y_HIT = Y_PREV + V_Y * TIME                                                   #
    #  Note: PADDLE's X-COORD is always 141.5                                                    #
    # ------------------------------------------------------------------------------------------ #

    def getAction(self, obs):
        """
        Automatically get best action by calculating the position of the BALL when it intercepts the PADDLE and moving towards it
        使用編碼後的觀察值來決定動作
        這裡可以整合您的 AI 模型
        """

        # 選項 2: 如果使用 encoder，可以在潛在空間中分析
        if self.use_encoder and len(self.encoded_observations) > 0:
            # 獲取最新的潛在表示
            latent = self.encoded_observations[-1]

            # TODO: 在這裡可以加入您的 AI 決策邏輯
            # 例如：使用潛在表示預測最佳動作
            # action = your_policy_network(latent)
            pass

        # 目前還是隨機動作

        # Get boundary positions of each: [top left corner, bottom right corner]
        BALL = self.getBallLocation(obs)
        PADDLE = self.getPaddleLocation(obs)

        if (BALL is not None and PADDLE is not None):

            # Calculate the center of each object: [TOP_CORNER + BOTTOM_CORNER] / 2
            BALL_CENTER = np.add(BALL[0], BALL[1]) / 2
            PADDLE_CENTER = np.add(PADDLE[0], PADDLE[1]) / 2

            if (self.PREV is not None):
                V = np.subtract(BALL_CENTER, self.PREV)
                if (V[1] > V_EPS):
                    TIME = (PADDLE_X - BALL_CENTER[1]) / V[1]
                    Y_HIT = BALL_CENTER[0] + V[0]*TIME
                    if (Y_HIT > PADDLE_CENTER[0]):
                        return 3
                    elif (Y_HIT < PADDLE_CENTER[0]):
                        return 2
            self.PREV = BALL_CENTER
        return 0

    def getBallLocation(self, obs):
        """Given a current RGB stream of the pixels, identify the location of the ball. Note: The ball doesn't seem to generate until the 60th frame"""

        # Get a mask where the values match our color
        color_mask = (obs == BALL_COLOR).all(axis=2)

        # Apply the mask to only get the indices that are rgb(COLOR, COLOR, COLOR)
        indices = np.argwhere(color_mask)

        # The ball is coordinates not in the BORDER
        ball_mask = ~np.isin(indices[:, 0], BORDER_ROWS)
        BALL = indices[ball_mask]

        # Return the top left corner and bottom right corner of the BALL if it exists
        if (BALL.size > 0):
            return np.array([BALL[0, :], BALL[BALL.shape[0]-1, :]])
        return None

    def getPaddleLocation(self, obs):
        """Helper function to determine the position of the GREEN Paddle. Note: GREEN paddle seems to load on the 3rd frame"""

        color_mask = (obs == PADDLE_COLOR).all(axis=2)
        indices = np.argwhere(color_mask)

        paddle_mask = indices[:, 0] > TOP_BORDER
        PADDLE = indices[paddle_mask]

        if (PADDLE.size > 0):
            return np.array([PADDLE[0, :], PADDLE[PADDLE.shape[0]-1, :]])
        return None

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
    print("Pong with ViT Encoder Test")
    print("=" * 60)

    # 創建 Pong 實例並啟用 encoder
    game = Pong(VIEW=True, PLAY=False, use_encoder=True)

    # 運行 300 幀以收集編碼數據
    game.visualize(FRAMES=300)

    print("\n✅ Test completed!")



