import torch
from train import PongDataset, PongFrameDataset, AutoencoderTrainer, DiTTrainer
from pong import Pong
import data_utils as utils
import os
import numpy as np

from timeit import default_timer as timer

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
        trained (bool): Flag to keep track of whether or not model has trained weights.
    """
    def __init__(self, encoder=None, decoder=None, dit=None, view=False, play=False, eps=0.01, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.ae_trainer = AutoencoderTrainer(encoder=encoder, decoder=decoder, device=device)
        self.dit_trainer = DiTTrainer(encoder=self.ae_trainer.autoencoder.encoder, dit=dit, device=device)
        self.trained = False
        self.device = device
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
        if not self.trained:
            SEEDS = utils.load_seeds()
            index = -1
            count = 0
            frames = []
            actions = []
            while count < num_frames:
                amount = min(10000, num_frames-count)
                index = (index + 1) % len(SEEDS)
                print(f"Switching to seed: {SEEDS[index]}")
                self.pong.env.reset(seed=int(SEEDS[index])) #These two lines should probably be a wrapper function in pong
                self.pong.PREV = None

                frames_batch, action_batch = self.pong.simulate(amount, COLLECT=True, CLOSE=False)
                frames.append(frames_batch)
                actions.append(action_batch)
                count += amount
            frames = np.concatenate(frames)
            actions = np.concatenate(actions)
            self.pong.env.close()
        else:
            frames, actions = self.pong.simulate(num_frames, COLLECT=True)
        print(f"✅ Data collection complete.")
        print(f"    - Frames shape: {frames.shape}")
        print(f"    - Actions shape: {actions.shape}")

        data = [(frames[t], actions[t], frames[t+1]) for t in range(frames.shape[0] - 1)]
        
        return data
    
    def load_weights(self, ae_path=None, dit_path=None):
        """
        Loads the given path files as weights into the ae_trainer and dit_trainer.
        Args:
            ae_path (string): OS path to the directory with the ae_trainer weights.
            dit_path (string): OS path to the directory with the dit_trainer weights.
        Returns:
            None.
        """
        try:
            #Try to load the given weights and set trained flag to True
            print("="*70)
            if ae_path is not None:
                self.ae_trainer.load(ae_path)
            if dit_path is not None:
                self.dit_trainer.load(dit_path)
            self.trained = True
            print("="*70)
        except IOError:
            raise FileNotFoundError()

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
        
        os.makedirs(save_dir, exist_ok=True)
        
        #Collect NUM_FRAMES amount of frames
        print("\n📊 Step 1: Collecting Game Data")
        data = self.collect_pong_data(num_frames=NUM_FRAMES, view=False)

        #If AUTOENCODER_EPOCHS is None, it means load previous weights and only
        #train the DiT
        if AUTOENCODER_EPOCHS > 0:
            frames, _, _ = zip(*data)
            #Split into train and validation sets
            train_set, val_set = utils.train_val_split(frames)

            #Convert to PongFrameDatasets
            train_set = PongFrameDataset(train_set)
            val_set = PongFrameDataset(val_set)

            #Step 2: Train Autoencoder
            print("\n🔧 Step 2: Training Autoencoder")

            self.ae_trainer.train(train_dataset=train_set, val_dataset=val_set, epochs=AUTOENCODER_EPOCHS, batch_size=BATCH_SIZE, save_dir=save_dir)
        
        if DIT_EPOCHS > 0:
            #Create DiT dataset
            dit_dataset = PongDataset(data)

            #Step 3: Train DiT
            print("\n✨ Step 3: Training DiT")
            self.dit_trainer.train(dataset=dit_dataset, epochs=DIT_EPOCHS, batch_size=BATCH_SIZE, save_dir=save_dir)

        self.trained = self.trained or (DIT_EPOCHS > 0 or AUTOENCODER_EPOCHS > 0)

        print("\n" + "=" * 70)
        print("🎉 Training complete!")
        print("=" * 70)
        print(f"\nModel saved to the {save_dir}/ directory")

    #TODO: Implement the inference pass using the ae_trainer and dit_trainer
    def inference(self, num_frames=20):
        """
        Inference function to begin generating a new frame from a start point and frame.
        Args:
            start (np.ndarray): RGB array of an initial Pong starting frame (will randomly generate if not given).
            num_frames (int): Number of frames to generate
        
        Returns:
            frames (np.ndarray): List of RGB arrays of generated frames.
        """
        print("="*70)
        assert self.trained, "No model loaded or trained."
        print(f"Running inference on model for {num_frames} frames.")

        frames = []
        data = self.collect_pong_data(num_frames=2, view=False)
        frame_t, action_t, frame_next = data[0]

        latent_t = self.ae_trainer.autoencoder.encoder.encode_frame(frame_t)
        action_t = torch.tensor([action_t], dtype=torch.long, device=self.device)

        torch.set_grad_enabled(False)
        latent_t = self.ae_trainer.autoencoder.encoder.encode_frame(frame_t)

        #Latency testing setup

        torch.cuda.synchronize()

        latencies = []
        start = timer()
        for i in range(num_frames):
            # timestep = torch.tensor([0], dtype=torch.long, device=self.device)
            timestep = torch.randint(0, 1000, (1,), device=self.device)
            
            #True noisy image
            noisy_latent, noise = self.dit_trainer.add_noise(latent=latent_t, timesteps=timestep)
            #Predicted noise
            eps_hat = self.dit_trainer.dit(noisy_latent, timestep, action_t, latent_t)
            
            #Undo noise on prediction
            latent_hat = self.dit_trainer.remove_noise(noisy_latent=noisy_latent, timesteps=timestep, noise=eps_hat)
            
            #Decode
            pred = self.ae_trainer.autoencoder.decoder(latent_hat)

            #Convert back to standard RGB (210, 160, 3)
            pred = pred.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            end = timer()
            latencies.append(end - start)
            start = end
            frames.append(torch.from_numpy(pred))
            
            latent = latent_hat

        #Save video
        utils.save_animation(torch.stack(frames), name='anim', fps=10, format='gif')
        print("Saved animation.")

        print("="*70)

        min_latency = np.argmin(latencies)
        max_latency = np.argmax(latencies)
        print("Latency Statistics:")
        print(f"MIN({min_latency}): {latencies[min_latency]}")
        print(f"MAX({max_latency}): {latencies[max_latency]}")
        print(f"AVG: {np.mean(latencies)}")
        print(f"STD: {np.std(latencies)}")

        utils.make_plot(np.arange(num_frames), latencies, title='Latencies', data_label='Latency', x_label='Frame Index', y_label='Time (seconds)', name='latency', save_dir='generated')
        return frames

