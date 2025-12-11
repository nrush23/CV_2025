"""
Main script for training and running OASIS-Pong
"""
import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from oasis_pipeline import create_oasis_pipeline
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def train_mode(args):
    """Training mode"""
    # Import Pong here to avoid dependency if only using inference
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from pong import Pong
    
    print("\n🎮 OASIS-Pong Training Mode")
    
    # Create Pong environment
    pong = Pong(VIEW=args.view, PLAY=False, EPS=args.epsilon)
    
    # Create pipeline
    vae_config = {
        'img_height': 210,
        'img_width': 160,
        'patch_size': 10,
        'embed_dim': 512 if not args.small else 256,
        'depth': 6 if not args.small else 4,
        'latent_dim': 16 if not args.small else 8,
    }
    
    dit_config = {
        'latent_dim': vae_config['latent_dim'],
        'hidden_size': 384 if not args.small else 256,
        'depth': 12 if not args.small else 8,
        'num_heads': 6 if not args.small else 4,
        'num_actions': 6,
    }
    
    pipeline = create_oasis_pipeline(vae_config=vae_config, dit_config=dit_config)
    
    # Load existing weights if specified
    if args.load_vae:
        pipeline.vae_trainer.load(args.load_vae)
    if args.load_dit:
        dit_path = args.load_dit
        state_dict = torch.load(dit_path, map_location=pipeline.device)
        if 'model_state_dict' in state_dict:
            pipeline.dit.load_state_dict(state_dict['model_state_dict'])
        else:
            pipeline.dit.load_state_dict(state_dict)
        print("✅ DiT loaded successfully")
    
    # Train
    pipeline.train(
        pong_env=pong,
        num_frames=args.frames,
        vae_epochs=args.vae_epochs,
        dit_epochs=args.dit_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )


def inference_mode(args):
    """Inference mode"""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    print("\n🎬 OASIS-Pong Inference Mode")
    
    # Import Pong to get initial frame
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from pong import Pong
    
    # Create pipeline
    vae_config = {
        'img_height': 210,
        'img_width': 160,
        'patch_size': 10,
        'embed_dim': 512 if not args.small else 256,
        'depth': 6 if not args.small else 4,
        'latent_dim': 16 if not args.small else 8,
    }
    
    dit_config = {
        'latent_dim': vae_config['latent_dim'],
        'hidden_size': 384 if not args.small else 256,
        'depth': 12 if not args.small else 8,
        'num_heads': 6 if not args.small else 4,
        'num_actions': 6,
    }
    
    pipeline = create_oasis_pipeline(vae_config=vae_config, dit_config=dit_config)
    
    # Load weights
    if not args.vae_path or not args.dit_path:
        raise ValueError("Must specify --vae-path and --dit-path for inference")
    
    pipeline.load_weights(vae_path=args.vae_path, dit_path=args.dit_path)
    
    # Get initial frame
    pong = Pong(VIEW=False, PLAY=False)
    pong.env.reset()
    initial_frame = pong.env.unwrapped.get_wrapper_attr("ale").getScreenRGB()
    
    # Generate random actions or use specified
    if args.actions:
        actions = [int(a) for a in args.actions.split(',')]
    else:
        actions = np.random.choice([0, 2, 3], size=args.num_frames)
    
    print(f"Generating {len(actions)} frames...")
    
    # Run inference
    generated_frames = pipeline.inference(
        initial_frame=initial_frame,
        actions=actions,
        num_denoising_steps=args.denoising_steps
    )
    
    # Save animation
    print(f"💾 Saving animation to {args.output}...")
    save_animation(generated_frames, args.output, fps=args.fps)
    print("✅ Done!")


def save_animation(frames, output_path, fps=10):
    """Save frames as animation"""
    frames = np.array(frames)
    
    if frames.max() > 1.0:
        frames = frames.astype(np.float32) / 255.0
    
    fig, ax = plt.subplots(figsize=(8, 10.5))
    ax.axis('off')
    
    im = ax.imshow(frames[0], animated=True)
    
    def update_frame(frame_number):
        im.set_array(frames[frame_number])
        return [im]
    
    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=len(frames),
        interval=1000/fps,
        blit=True,
        repeat=True
    )
    
    if output_path.endswith('.mp4'):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='OASIS-Pong'), bitrate=1800)
        anim.save(output_path, writer=writer)
    elif output_path.endswith('.gif'):
        anim.save(output_path, writer='pillow', fps=fps)
    else:
        raise ValueError("Output must be .mp4 or .gif")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='OASIS-Pong: Neural Game Engine')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True,
                        help='Mode: train or inference')
    
    # Training arguments
    parser.add_argument('--frames', type=int, default=5000,
                        help='Number of frames to collect for training')
    parser.add_argument('--vae-epochs', type=int, default=20,
                        help='Number of VAE training epochs')
    parser.add_argument('--dit-epochs', type=int, default=15,
                        help='Number of DiT training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--save-dir', type=str, default='oasis_checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--view', action='store_true',
                        help='View Pong during data collection')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='Epsilon for Pong AI')
    parser.add_argument('--load-vae', type=str, default=None,
                        help='Path to pre-trained VAE weights')
    parser.add_argument('--load-dit', type=str, default=None,
                        help='Path to pre-trained DiT weights')
    
    # Inference arguments
    parser.add_argument('--vae-path', type=str, default=None,
                        help='Path to trained VAE weights')
    parser.add_argument('--dit-path', type=str, default=None,
                        help='Path to trained DiT weights')
    parser.add_argument('--num-frames', type=int, default=30,
                        help='Number of frames to generate')
    parser.add_argument('--denoising-steps', type=int, default=50,
                        help='Number of denoising steps per frame')
    parser.add_argument('--actions', type=str, default=None,
                        help='Comma-separated list of actions (e.g., "0,2,3,2")')
    parser.add_argument('--output', type=str, default='generated_pong.gif',
                        help='Output animation file (.mp4 or .gif)')
    parser.add_argument('--fps', type=int, default=10,
                        help='FPS for output animation')
    
    # Model size
    parser.add_argument('--small', action='store_true',
                        help='Use smaller model (faster but lower quality)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'inference':
        inference_mode(args)


if __name__ == "__main__":
    main()
