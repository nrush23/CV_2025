import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os


def train_val_split(dataset, p=0.9):
    """Utility to split datasets into a training and validation set
    Args:
        dataset (np.ndarray): Original dataset with size N.
        p (float): Percent to keep in the training set, default 0.9.
    Returns:
        Tuple (train_set, val_set):
            - train_set (np.ndarray): Training set with data from [0, Math.floor(N*p)].
            - val_set (np.ndarray): Validation set with data from [Math.floor(N*p), N].
    """
    split_idx = int(len(dataset) * p)
    train_set = dataset[:split_idx]
    val_set = dataset[split_idx:]

    return train_set, val_set


def save_img(frame, name='test'):
    """
    Utility to save images using matplotlib
    Args:
        frame (np.ndarray): RGB image, expecting numpy array with shape (H, W, C).
        name (string): Name of the saved image, default is test"""
    assert frame.shape == (210, 160, 3)
    plt.imsave(f"generated/{name}.png", frame)

# TODO: Write code to create and save an animation to generated/ using matplotlib.animation


def save_animation(frames, name='test', format='mp4', fps=30):
    """
    Utility to make an animation from a list of frames.

    Args:
        frames (np.ndarray): Array of RGB frames, shape is (N, 210, 160, 3).
        name (string): Name of the animation (without extension), default is 'test'.
        fps (int): Frames per second for the animation, default is 30.
        format (string): Output format - 'mp4' or 'gif', default is 'mp4'.

    Returns:
        None. Saves animation to generated/{name}.{format}

    Example:
        frames = np.random.randint(0, 255, (100, 210, 160, 3), dtype=np.uint8)
        save_animation(frames, name='my_pong_game', fps=30, format='mp4')
    """
    # check inputs
    assert len(
        frames.shape) == 4, f"Expected 4D array (N, H, W, C), got shape {frames.shape}"
    assert frames.shape[1:] == (
        210, 160, 3), f"Expected frames with shape (N, 210, 160, 3), got {frames.shape}"

    # check generate directory
    os.makedirs('generated', exist_ok=True)

    # check frames range
    if frames.max() > 1.0:
        frames = frames.astype(np.float32) / 255.0

    print(f"🎬 Creating animation with {len(frames)} frames at {fps} FPS...")

    # create figure and axis
    fig, ax = plt.subplots(figsize=(8, 10.5))
    ax.axis('off')

    # initialize image
    im = ax.imshow(frames[0], animated=True)

    # update frame
    def update_frame(frame_number):
        im.set_array(frames[frame_number])
        return [im]

    # create animation
    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=len(frames),
        interval=1000/fps,
        blit=True,
        repeat=True
    )

    # save animation
    output_path = f"generated/{name}.{format}"

    try:
        if format == 'mp4':
            # save as MP4
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(
                artist='Pong AI'), bitrate=1800)
            anim.save(output_path, writer=writer)

        elif format == 'gif':
            # save as GIF
            anim.save(output_path, writer='pillow', fps=fps)

        else:
            raise ValueError(
                f"Unsupported format: {format}. Use 'mp4' or 'gif'.")

        print(f"✅ Animation saved to {output_path}")
        print(f"   Duration: {len(frames)/fps:.2f} seconds")
        print(f"   Resolution: 210×160")
        print(f"   Total frames: {len(frames)}")

    except Exception as e:
        print(f"❌ Error saving animation: {e}")

        if format == 'mp4':
            print("\n💡 Tip: MP4 format requires ffmpeg. Install it with:")
            print("   - Windows: Download from https://ffmpeg.org/download.html")
            print("   - Mac: brew install ffmpeg")
            print("   - Linux: sudo apt-get install ffmpeg")
            print("\n   Or try using format='gif' instead.")

        raise

    finally:
        plt.close(fig)


def generate_seeds(N=10):
    """
    Generates seeds to be put into seeds.txt.
    Args:
        N (int): How many seeds to create.

    Returns:
        None.
    """
    print('='*70)
    print(f"Generating {N} seeds...")
    generator = np.random.default_rng()
    MAX = 1000000

    seeds = generator.choice(a=MAX, size=N, replace=False)

    with open("seeds.txt", "w") as f:
        for s in seeds:
            f.write(f"{s}\n")
    print(f"Finished generating {N} seeds.")
    print('='*70)


def load_seeds():
    print('='*70)
    print("Loading seeds from seeds.txt...")
    seeds = np.loadtxt('seeds.txt', dtype=int)
    print(seeds)
    print("Finished loading seeds from seeds.txt")
    print('='*70)
    return seeds


def make_plot(x, y, title='X vs. Y', data_label='Data', x_label='X', y_label='Y', name='graph', save_dir='generated/'):
    """Plots the training curves"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}.png")
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=data_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Graph saved to {save_path}")
