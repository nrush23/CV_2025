"""
OASIS Main Entry Point (Compatible with original command format)
Combines game simulation, training, and inference for Pong using OASIS framework.
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train, collect_pong_data
from pong import Pong


def main():
    """
    OASIS Training Pipeline - Compatible with original command format
    
    Original command example:
        python main.py -f 10000 -t -ae 25 -de 25 -b 32
    
    New OASIS equivalent:
        python main.py -f 10000 -t -ae 25 -de 25 -b 32 --seq-len 4
    """
    
    parser = argparse.ArgumentParser(
        description='OASIS Pong Training - Compatible with original format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (small)
  python main.py -f 2000 -t -ae 10 -de 10 -b 8
  
  # Standard training (original command compatible)
  python main.py -f 10000 -t -ae 25 -de 25 -b 32
  
  # Full training with OASIS features
  python main.py -f 10000 -t -ae 50 -de 30 -b 32 --seq-len 4 --kl-weight 1e-6
  
  # Just play the game (view mode)
  python main.py -f 1000 -v
  
  # Inference mode (TODO: needs inference_oasis.py)
  python main.py -l -i 90
        """
    )
    
    # Original arguments (kept for compatibility)
    parser.add_argument("-f", "--Frames", type=int, default=5000,
                        help="Number of frames to collect (default: 5000)")
    parser.add_argument("-v", "--View", action='store_true',
                        help="Visualize the game simulation")
    parser.add_argument("-p", "--Play", action='store_true',
                        help="User plays the game")
    parser.add_argument("-e", "--Epsilon", type=float, default=0.01,
                        help="Epsilon for random moves (default: 0.01)")
    parser.add_argument("-t", "--Train", action='store_true',
                        help="Training mode")
    parser.add_argument("-ae", "--AutoEncod", type=int, default=50,
                        help="VAE/Autoencoder epochs (default: 50)")
    parser.add_argument("-de", "--DiT", type=int, default=30,
                        help="DiT epochs (default: 30)")
    parser.add_argument("-b", "--Batches", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("-l", "--Load", action="store_true",
                        help="Load trained models for inference")
    parser.add_argument("-i", "--Inferences", type=int, default=90,
                        help="Number of inference frames (default: 90)")
    
    # New OASIS-specific arguments
    parser.add_argument("--seq-len", "--sequence-length", type=int, default=4,
                        dest="seq_len",
                        help="Sequence length for temporal modeling (default: 4)")
    parser.add_argument("--kl-weight", type=float, default=1e-6,
                        help="KL divergence weight for VAE (default: 1e-6)")
    parser.add_argument("--save-dir", type=str, default='checkpoints',
                        help="Directory to save checkpoints (default: 'checkpoints')")
    parser.add_argument("--visualize-every", type=int, default=5,
                        help="Visualize every N epochs (default: 5)")
    
    args = parser.parse_args()
    
    # Parse arguments
    FRAMES = args.Frames
    VIEW = args.View
    PLAY = args.Play
    EPS = args.Epsilon
    TRAIN = args.Train
    AE_EPOCHS = args.AutoEncod
    DIT_EPOCHS = args.DiT
    BATCH_SIZE = args.Batches
    LOAD = args.Load
    INFERENCES = args.Inferences
    SEQ_LEN = args.seq_len
    KL_WEIGHT = args.kl_weight
    SAVE_DIR = args.save_dir
    VIZ_EVERY = args.visualize_every
    
    GAME = not (TRAIN or LOAD)
    
    # Print configuration
    print("=" * 70)
    print("🎮 OASIS Pong - Main Entry Point")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"  Mode: {'Game' if GAME else ('Training' if TRAIN else 'Inference')}")
    print(f"  Frames: {FRAMES}")
    print(f"  Batch Size: {BATCH_SIZE}")
    
    if TRAIN:
        print(f"  VAE Epochs: {AE_EPOCHS}")
        print(f"  DiT Epochs: {DIT_EPOCHS}")
        print(f"  Sequence Length: {SEQ_LEN}")
        print(f"  KL Weight: {KL_WEIGHT}")
        print(f"  Save Directory: {SAVE_DIR}")
    
    print()
    
    # Mode 1: Play the game (no training)
    if GAME:
        print("🎮 Starting Game Mode")
        print("-" * 70)
        game = Pong(VIEW=VIEW, PLAY=PLAY, EPS=EPS)
        game.simulate(FRAMES, COLLECT=False, CLOSE=True)
        print("\n✅ Game finished!")
    
    # Mode 2: Training mode
    elif TRAIN:
        print("🚀 Starting Training Mode")
        print("-" * 70)
        
        # Confirm before training
        print(f"\n⚠️  About to train with:")
        print(f"   • {FRAMES} frames")
        print(f"   • {AE_EPOCHS} VAE epochs")
        print(f"   • {DIT_EPOCHS} DiT epochs")
        print(f"   • Batch size: {BATCH_SIZE}")
        print(f"   • Sequence length: {SEQ_LEN}")
        
        response = input("\nContinue? [Y/n]: ").strip().lower()
        if response and response != 'y' and response != 'yes':
            print("Training cancelled.")
            return
        
        # Run training
        try:
            ae_trainer, dit_trainer = train(
                NUM_FRAMES=FRAMES,
                AUTOENCODER_EPOCHS=AE_EPOCHS,
                DIT_EPOCHS=DIT_EPOCHS,
                BATCH_SIZE=BATCH_SIZE,
                SEQUENCE_LENGTH=SEQ_LEN,
                VISUALIZE_EVERY=VIZ_EVERY,
                save_dir=SAVE_DIR
            )
            
            print("\n" + "=" * 70)
            print("🎉 Training Complete!")
            print("=" * 70)
            print(f"\n📁 Models saved to: {SAVE_DIR}/")
            print("\n📊 Files created:")
            print(f"   • {SAVE_DIR}/best_autoencoder.pth")
            print(f"   • {SAVE_DIR}/autoencoder_final.pth")
            print(f"   • {SAVE_DIR}/dit_final.pth")
            print(f"   • {SAVE_DIR}/vae_training_curves.png")
            print(f"   • {SAVE_DIR}/dit_training_curves.png")
            print(f"   • {SAVE_DIR}/visualizations/")
            
            print("\n💡 Next steps:")
            print("   1. Check training curves in the save directory")
            print("   2. Review visualizations")
            print("   3. Run inference with: python main.py -l -i 90")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            print(f"Partial checkpoints may be in: {SAVE_DIR}/")
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Mode 3: Inference mode
    elif LOAD:
        print("🔮 Starting Inference Mode")
        print("-" * 70)
        
        # Check if inference_oasis.py exists
        inference_path = os.path.join(os.path.dirname(__file__), 'inference_oasis.py')
        if not os.path.exists(inference_path):
            print("❌ inference_oasis.py not found!")
            print("\nTo use inference mode, you need inference_oasis.py")
            print("It should already be in /mnt/user-data/outputs/")
            return
        
        try:
            from inference_oasis import OASISInference
            
            # Default paths
            vae_path = os.path.join(SAVE_DIR, 'best_autoencoder.pth')
            dit_path = os.path.join(SAVE_DIR, 'dit_final.pth')
            
            # Check if files exist
            if not os.path.exists(vae_path):
                print(f"❌ VAE model not found: {vae_path}")
                print("Please train first with: python main.py -f 5000 -t -ae 50 -de 30")
                return
            
            if not os.path.exists(dit_path):
                print(f"❌ DiT model not found: {dit_path}")
                print("Please complete training first")
                return
            
            print(f"📦 Loading models...")
            print(f"   VAE: {vae_path}")
            print(f"   DiT: {dit_path}")
            
            inference = OASISInference(vae_path, dit_path)
            
            print(f"\n🎬 Generating {INFERENCES} frames...")
            
            # Collect initial frame from game
            game = Pong(VIEW=False, PLAY=False, EPS=0.01)
            frames, actions = game.simulate(1, COLLECT=True, CLOSE=False)
            start_frame = frames[0]
            
            # Generate sequence
            output_path = f'{SAVE_DIR}/inference_output.mp4'
            inference.generate_sequence(
                start_frame=start_frame,
                actions=actions[:INFERENCES],
                num_frames=INFERENCES,
                output_path=output_path
            )
            
            print(f"\n✅ Inference complete!")
            print(f"📹 Video saved to: {output_path}")
            
        except ImportError as e:
            print(f"❌ Could not import inference_oasis: {e}")
            print("Make sure inference_oasis.py is in the same directory")
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
