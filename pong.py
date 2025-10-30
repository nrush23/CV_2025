import gymnasium
import ale_py
import numpy as np
import sys
import keyboard
import torch

# Import our encoder components
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
        
        # Initialize the encoder (if required)
        self.use_encoder = use_encoder
        if self.use_encoder:
            print("Initializing ViT Encoder...")
            self.encoder = create_encoder()
            self.encoder.eval()  # Set to evaluation mode
            print("✅ Encoder is ready")
            
            # Used to store encoded observations
            self.encoded_observations = []

    def visualize(self, FRAMES=10):
        """Runs Pong at 30FPS for FRAMES amount of frames"""
        for i in range(FRAMES):
            # Get the RGB frame
            obs = self.env.unwrapped.get_wrapper_attr("ale").getScreenRGB()
            
            # If using the encoder, encode the current frame
            if self.use_encoder:
                latent = encode_pong_observation(self.encoder, obs)
                self.encoded_observations.append(latent)
                
                # Display encoding information every 100 frames
                if i % 100 == 0:
                    print(f"Frame {i}: Latent representation shape = {latent.shape}")
            
            # Get the action
            action = self.getPlay() if self.PLAY else self.getAction(obs)

            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                obs, info = self.env.reset()

        self.env.close()
        print(f"Finished running PONG for: {FRAMES} frames")
        
        if self.use_encoder:
            print(f"{len(self.encoded_observations)} observations were collected and encoded.")
            self.save_encoded_observations()

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

    def getAction(self, obs):
        """
        Determines the action using the encoded observation
        Your AI model can be integrated here
        """
        # Option 1: Use raw pixels to find the ball (for basic logic)
        masked = (obs == 236).all(axis=2)
        
        # Option 2: If using the encoder, analysis can be done in the latent space
        if self.use_encoder and len(self.encoded_observations) > 0:
            # Get the latest latent representation
            latent = self.encoded_observations[-1]
            
            # TODO: Add your AI decision logic here
            # For example: action = your_policy_network(latent)
            pass
        
        # Currently defaults to a random action
        return self.env.action_space.sample()

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
    print("Pong with ViT Encoder Test")
    print("=" * 60)
    
    # Create Pong instance and enable the encoder
    game = Pong(VIEW=True, PLAY=False, use_encoder=True)
    
    # Run for 300 frames to collect encoded data
    game.visualize(FRAMES=300)
    
    print("\n✅ Test completed!")
