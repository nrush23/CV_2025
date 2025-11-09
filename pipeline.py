import torch
from train import PongFrameDataset, AutoencoderTrainer, DiTTrainer
from pong import Pong
import data_utils as utils

class Pipeline():
    """
    Pipeline class to put together our train components for training and inference
    Args:
        encoder (ViTEncoder): Encoder to be used in our model.
        decoder (ViTDecoder): Decoder to be used in our model.
        dit (DiT): DiT to be used in our model.
        view (bool): Display Pong simulation, defaults to False.
        play (bool): Use keyboard input for training, defaults to False.
        eps (float): Epsilon value to be used in the Pong interface for likelihood to choose a random action.
        device (string): Device to be used for torch.
    """
    def __init__(self, encoder=None, decoder=None, dit=None, view=False, play=False, eps=0.01, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.ae_trainer = AutoencoderTrainer(encoder=encoder, decoder=decoder, device=device)
        self.dit_trainer = DiTTrainer(encoder=self.ae_trainer.autoencoder.encoder, dit=dit, device=device)
        self.pong = Pong(VIEW=view, PLAY=play, EPS=eps)
    
    def collect_pong_data(self, num_frames, view=False):
        """Collects Pong game data
        Args:
            num_frames (int): Number of frames to collect from ALE.
            view (bool): Display the frames in real time, default false.
        Returns:
            Tuple (frames, actions):
            - frames (np.ndarray): Np array of the frames with shape (N, 210, 160, 3).
            - actions (np.ndarray): Np array of the actions associated at each frame with shape (N, ).  
        """
        print(f"📊 Collecting {num_frames} frames of Pong data...")

        # PONG = Pong(VIEW=view, PLAY=False, EPS=0.01)
        frames, actions = self.pong.simulate(num_frames, True)
        
        print(f"✅ Data collection complete.")
        print(f"    - Frames shape: {frames.shape}")
        print(f"    - Actions shape: {actions.shape}")
        
        return frames, actions

    def train(self, NUM_FRAMES=5000, AUTOENCODER_EPOCHS=20, DIT_EPOCHS=15, BATCH_SIZE=16, save_dir='checkpoints'):
        """
        Train our model on the specified amount of frames, epochs, and batch size.
        Args:
            NUM_FRAMES (int): Number of frames.
            AUTOENCODER_EPOCHS (int): Autoencoder Epoch size.
            DIT_EPOCHS (int): DIT Epoch size.
            BATCH_SIZE (int): Batch size.
        Returns:
            Tuple (Autoencoder, DiT):
                - Autoencoder (Encoder): Autoencoder used during training.
                - DiT (DiT): DiT used during training.
        """
        print("=" * 70)
        print("🎮 Pong AI Training Pipeline")
        print("=" * 70)
        
        
        #Collect NUM_FRAMES amount of frames
        print("\n📊 Step 1: Collecting Game Data")
        frames, actions = self.collect_pong_data(num_frames=NUM_FRAMES, view=False)

        #Split into train and validation sets
        train_set, val_set = utils.train_val_split(frames)

        #Convert to PongFrameDatasets
        train_set = PongFrameDataset(train_set)
        val_set = PongFrameDataset(val_set)

        #Step 2: Train Autoencoder
        print("\n🔧 Step 2: Training Autoencoder")

        self.ae_trainer.train(train_dataset=train_set, val_dataset=val_set, epochs=AUTOENCODER_EPOCHS, batch_size=BATCH_SIZE, save_dir=save_dir)

        #Create DiT dataset
        dit_dataset = PongFrameDataset(frames=frames, actions=actions)

        #Step 3: Train DiT
        print("\n✨ Step 3: Training DiT")
        self.dit_trainer.train(dataset=dit_dataset, epochs=DIT_EPOCHS, batch_size=BATCH_SIZE, save_dir=save_dir)
        
        print("\n" + "=" * 70)
        print("🎉 Training complete!")
        print("=" * 70)
        print(f"\nModel saved to the {save_dir}/ directory")

    #TODO: Implement the inference pass using the ae_trainer and dit_trainer
    def inference(self, start=None, action=None):
        """
        Inference function to begin generating a new frame from a start point and frame.
        Args:
            start (np.ndarray): RGB array of an initial Pong starting frame (will randomly generate if not given).
            action (int): Integer action key to use from the Pale interface (will follow Pong settings for action).
        
        Returns:
            frame (np.ndarray): Returns an RGB array of the next frame conditioned based on our inputs.
        """
        pass