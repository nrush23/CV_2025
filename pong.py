import keyboard
import gymnasium
import ale_py
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


class Pong:
    """ Pong interface for accessing and interacting the ALE Gymnasium Pong model """

    def __init__(self, VIEW=True, PLAY=False, EPS=0.01):
        gymnasium.register_envs(ale_py)
        render_mode = "human" if (VIEW or PLAY) else "rgb_array"
        self.env = gymnasium.make(
            "ALE/Pong-v5", render_mode=render_mode, frameskip=1, repeat_action_probability=0.0)
        self.env.reset()
        self.PLAY = PLAY
        self.PREV = None
        self.EPS = EPS

    def visualize(self, FRAMES=10):
        """Runs Pong at 30FPS for FRAMES amount of frames
        Arguments:
        FRAMES -- integer (default 10 frames)
        """
        # Frame rate is 30FPS, so simulation will run for FRAMES/30 seconds
        for _ in range(FRAMES):
            obs = self.env.unwrapped.get_wrapper_attr("ale").getScreenRGB()
            # action = self.getPlay() if self.PLAY else self.getAction(obs)
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
        print("Finished running PONG for: %i frames" % FRAMES)

    def getAction(self, obs):
        """Automatically get best action by calculating the position of the BALL when it intercepts the PADDLE and moving towards it """

        # Get boundary positions of each: [top left corner, bottom right corner]
        BALL = self.getBallDimension(obs)
        PADDLE = self.getPaddleDimension(obs)

        if (BALL is not None and PADDLE is not None):

            # Calculate the center of each object: [TOP_CORNER + BOTTOM_CORNER] / 2
            BALL_CENTER = np.add(BALL[0], BALL[1]) / 2
            PADDLE_CENTER = np.add(PADDLE[0], PADDLE[1]) / 2

            # ------------------------------------  PHYSICS LOGIC  ------------------------------------- #
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

            if (self.PREV is not None):
                V = np.subtract(BALL_CENTER, self.PREV)
                if (V[1] > 0.001):
                    TIME = (141.5 - BALL_CENTER[1]) / V[1]
                    Y_HIT = BALL_CENTER[0] + V[0]*TIME
                    if (Y_HIT > PADDLE_CENTER[0]):
                        return 3
                    elif (Y_HIT < PADDLE_CENTER[0]):
                        return 2
            self.PREV = BALL_CENTER
        return 0

    def getBallDimension(self, obs):
        """Given a current RGB stream of the pixels, identify the location of the ball. Note: The ball doesn't seem to generate until the 60th frame"""
        COLOR = 236  # The color of the ball is RGB(236, 236, 236)

        # I looked at how the pixels changed from frame to frame, these consistently stayed the same so they must be the white borders
        BORDER_X = np.array([24,  25,  26,  27,  28,  29,  30,  31,  32,  33, 194, 195, 196, 197, 198, 199, 200, 201,
                             202, 203, 204, 205, 206, 207, 208, 209])

        # Get a mask where the values match our color
        color_mask = (obs == COLOR).all(axis=2)

        # Apply the mask to only get the indices that are rgb(COLOR, COLOR, COLOR)
        indices = np.argwhere(color_mask)

        # The ball is coordinates not in the BORDER
        ball_mask = ~np.isin(indices[:, 0], BORDER_X)
        BALL = indices[ball_mask]

        # Return the top left corner and bottom right corner of the BALL if it exists
        if (BALL.size > 0):
            return np.array([BALL[0, :], BALL[BALL.shape[0]-1, :]])
        return None

    def getPaddleDimension(self, obs):
        """Helper function to determine the position of the GREEN Paddle. Note: GREEN paddle seems to load on the 3rd frame"""
        # The color of the GREEN paddle is rgb(92, 186, 92) and nothing GREEN appears below the top border
        COLOR = np.array([92, 186, 92])
        BORDER_X = 33

        color_mask = (obs == COLOR).all(axis=2)
        indices = np.argwhere(color_mask)

        paddle_mask = indices[:, 0] > BORDER_X
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
