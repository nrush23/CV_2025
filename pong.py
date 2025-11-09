import gymnasium
import ale_py
import numpy as np
import sys
import keyboard
import torch

# Import our encoder components
from encoder import create_encoder, encode_pong_observation

np.set_printoptions(threshold=sys.maxsize)

#--------------------- PONG LOGIC, CRITICAL ----------------------------#

# Color masks to figure out the ball and paddle positions
BALL_COLOR = 236
PADDLE_COLOR = np.array([92, 186, 92])

# Borders to help us compute the locations of the ball and paddle
BORDER_ROWS = np.array([24,  25,  26,  27,  28,  29,  30,  31,  32,  33, 194,
                       195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209])
TOP_BORDER = 33

# Location of the paddle's center on the X axis
PADDLE_X = 141.5

# Velocity epsilon for determining whether or not we need to track the location of the ball
V_EPS = 0.001

#--------------------------- DON'T DELETE ------------------------------#

class Pong:
    """ Pong interface with ViT Encoder integration """

    def __init__(self, VIEW=True, PLAY=False, EPS=0.01, use_encoder=False):
        gymnasium.register_envs(ale_py)
        render_mode = "human" if (VIEW or PLAY) else "rgb_array"
        #-------------------- CRITICAL, LEAVE THIS ENV -------------------#
        self.env = gymnasium.make("ALE/Pong-v5", render_mode=render_mode, frameskip=1, repeat_action_probability=0.0)
        self.env.reset()
        self.PLAY = PLAY
        self.PREV = None
        self.EPS = EPS

        # Initialize the encoder (if required)
        self.use_encoder = use_encoder
        if self.use_encoder:
            print("Initializing ViT Encoder...")
            self.encoder = create_encoder()
            self.encoder.eval()  # Set to evaluation mode
            print("✅ Encoder is ready")

            # Used to store encoded observations
            self.encoded_observations = []

    def simulate(self, FRAMES=1000, COLLECT=False, CLOSE=True):
        """
        Runs a simulation for FRAMES amount of frames. If collecting, returns lists of the frames and actions taken at that frame
        Args:
            FRAMES (int): Number of frames.
            COLLECT (bool): Flag to collect frames and actions.
            CLOSE (bool): End the simulation after.
        Returns:
            Tuple [np.ndarray, np.ndarray] or None:
            If `COLLECT` is True, returns:
                - FRAMES (np.ndarray): List of RGB arrays representing frames of Pong.
                - ACTIONS (np.ndarray): List of action codes representing the action taken at
                its respective frame.
        """
        if COLLECT:
            DATA = []

        for i in range(FRAMES):
            start = self.env.unwrapped.get_wrapper_attr("ale").getScreenRGB()
            action = self.getAction(start)

            outcome, reward, terminated, truncated, info = self.env.step(action)

            if COLLECT:
                DATA.append((start, action, outcome))

            if terminated or truncated:
                start, info = self.env.reset()
        if CLOSE:
            self.env.close()
        print(f"Finished running PONG for: {FRAMES} frames")

        if COLLECT:
            frames_t = np.array([s for (s, a, o) in DATA], dtype=np.uint8)   # (N, 210, 160, 3)
            actions  = np.array([a for (s, a, o) in DATA], dtype=np.int64)   # (N,)
            # frames_t1 = np.array([o for (s, a, o) in DATA], dtype=np.uint8)
            return frames_t, actions


    def visualize(self, FRAMES=10):
        """Runs Pong at 30FPS for FRAMES amount of frames"""
        for i in range(FRAMES):
            # Get the RGB frame
            obs = self.env.unwrapped.get_wrapper_attr("ale").getScreenRGB()
            
            # Get our next action
            action = self.getAction(obs)

            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                obs, info = self.env.reset()

        self.env.close()
        print(f"Finished running PONG for: {FRAMES} frames")


    def save_encoded_observations(self, filename="encoded_observations.pt"):
        """Saves the encoded observations for later training use"""
        if not self.encoded_observations:
            print("No observations were collected and encoded.")
            return

        # Stack all observations into a single tensor
        encoded_tensor = torch.cat(self.encoded_observations, dim=0)
        torch.save(encoded_tensor, filename)
        print(f"✅ Saved encoded observations to {filename}")
        print(f"    Shape: {encoded_tensor.shape}")

    # -----------------------------  getAction PHYSICS LOGIC  ---------------------------------- #
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
        """
        #If user is playing, return keyboard input
        if self.PLAY:
            return self.getPlay()

        #Otherwise, computer policy: Check for random threshold
        if (np.random.rand() < self.EPS):
            return self.env.action_space.sample()

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


# ============ Usage Example ============
if __name__ == "__main__":
    print("=" * 60)
    print("Pong Test")
    print("=" * 60)

    # Create Pong instance and enable the encoder
    game = Pong(VIEW=True, PLAY=False)

    # Run for 300 frames to collect encoded data
    game.simulate(FRAMES=600)

    print("\n✅ Test completed!")
