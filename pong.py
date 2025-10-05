import gymnasium
import ale_py


class Pong:
    """ Pong interface for accessing and interacting the ALE Gymnasium Pong model """
    def __init__(self, VIEW):
        gymnasium.register_envs(ale_py)
        render_mode = "human" if VIEW else "rgb_array"
        self.env = gymnasium.make("ALE/Pong-v5", render_mode=render_mode)
        self.env.reset()

    def visualize(self, FRAMES=10):
        """Runs Pong at 60FPS for FRAMES amount of frames
        Arguments:
        FRAMES -- integer (default 10 frames)
        """
        #Frame rate is 60FPS, so simulation will run for FRAMES/60 seconds
        for _ in range(FRAMES):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                obs, info = self.env.reset()

        self.env.close()
        print("Finished running PONG for: %i frames" % FRAMES)