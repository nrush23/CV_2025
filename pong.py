import gymnasium
import ale_py
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import keyboard


class Pong:
    """ Pong interface for accessing and interacting the ALE Gymnasium Pong model """
    def __init__(self, VIEW=True, PLAY=False):
        gymnasium.register_envs(ale_py)
        render_mode = "human" if (VIEW or PLAY) else "rgb_array"
        self.env = gymnasium.make("ALE/Pong-v5", render_mode=render_mode, frameskip=1)
        self.env.reset()
        self.PLAY = PLAY
        self.PREV = None

    def visualize(self, FRAMES=10):
        """Runs Pong at 30FPS for FRAMES amount of frames
        Arguments:
        FRAMES -- integer (default 10 frames)
        """
        #Frame rate is 30FPS, so simulation will run for FRAMES/30 seconds
        for _ in range(FRAMES):
            obs = self.env.unwrapped.get_wrapper_attr("ale").getScreenRGB()
            action = self.getPlay() if self.PLAY else self.getAction(obs)

            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                obs, info = self.env.reset()

        self.env.close()
        print("Finished running PONG for: %i frames" % FRAMES)


    #---------------------------- TO DO -------------------------------#
    #     Implement getAction so that it finds the pong square and     #
    #     automatically calculates the best move to make from there    #
    #------------------------------------------------------------------#
    def getAction(self, obs):
        """Automatically get best action by calculating the position of the ball and moving towards it """
        masked = (obs == 236).all(axis=2)
                

        return self.env.action_space.sample()

    def getPlay(self):
        """If player controlled, check keyboard inputs for actions"""
        if keyboard.is_pressed('w') or keyboard.is_pressed('up'):
            return 2
        elif keyboard.is_pressed('s') or keyboard.is_pressed('down'):
            return 3
        return 0  