import keyboard
import gymnasium
import ale_py
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


class Pong:
    """ Pong interface for accessing and interacting the ALE Gymnasium Pong model """

    def __init__(self, VIEW=True, PLAY=False):
        gymnasium.register_envs(ale_py)
        render_mode = "human" if (VIEW or PLAY) else "rgb_array"
        self.env = gymnasium.make(
            "ALE/Pong-v5", render_mode=render_mode, frameskip=1, repeat_action_probability=0.0)
        self.env.reset()
        self.PLAY = PLAY
        self.PREV = np.empty((1, 1), dtype=int)

    def visualize(self, FRAMES=10):
        """Runs Pong at 30FPS for FRAMES amount of frames
        Arguments:
        FRAMES -- integer (default 10 frames)
        """
        # Frame rate is 30FPS, so simulation will run for FRAMES/30 seconds
        for _ in range(FRAMES):
            obs = self.env.unwrapped.get_wrapper_attr("ale").getScreenRGB()
            action = self.getPlay() if self.PLAY else self.getAction(obs)

            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                obs, info = self.env.reset()

        self.env.close()
        print("Finished running PONG for: %i frames" % FRAMES)

    # ---------------------------- TO DO -------------------------------#
    #     Implement getAction so that it finds the pong square and     #
    #     automatically calculates the best move to make from there    #
    # ------------------------------------------------------------------#

    def getAction(self, obs):
        """Automatically get best action by calculating the position of the ball and moving towards it """
        BALL = self.getBallDimension(obs)

        return self.env.action_space.sample()

    def getBallDimension(self, obs):
        """Given a current RGB stream of the pixels, identify the location of the ball. Note: The ball doesn't seem to generate until the 60th frame"""
        COLOR = 236  # The color of the ball is RGB(236, 236, 236)

        #I looked at how the pixels changed from frame to frame, these consistently stayed the same so they must be the white borders
        BORDER_X = np.array([24,  25,  26,  27,  28,  29,  30,  31,  32,  33, 194, 195, 196, 197, 198, 199, 200, 201,
                             202, 203, 204, 205, 206, 207, 208, 209])
        
        #Get a mask where the values match our color
        color_mask = (obs == COLOR).all(axis=2)

        #Apply the mask to only get the indices that are rgb(COLOR, COLOR, COLOR)
        indices = np.argwhere(color_mask)

        #The ball is coordinates not in the BORDER
        ball_mask = ~np.isin(indices[:,0], BORDER_X)
        BALL = indices[ball_mask]

        #Print the BALL when it exists
        if (BALL.size > 0 ):
            print(BALL)
            return np.array([indices[0,:], indices[8,:]])
        return None
    
    def getPaddleDimension(self, obs):
        """Helper function to determine the position of the GREEN Paddle"""
        return

    def getPlay(self):
        """If player controlled, check keyboard inputs for actions"""
        if keyboard.is_pressed('w') or keyboard.is_pressed('up'):
            return 2
        elif keyboard.is_pressed('s') or keyboard.is_pressed('down'):
            return 3
        return 0
