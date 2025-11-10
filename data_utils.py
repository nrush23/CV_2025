import matplotlib.pyplot as plt

def train_val_split(dataset, p = 0.9):
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
    plt.imsave("generated/test.png", frame)

#TODO: Write code to create and save an animation to generated/ using matplotlib.animation
def save_animation(frames, name='test'):
    """
    Utility to make an animation from a list of frames.
    Args:
        frames (np.ndarray): Array of RGB frames, shape is (N, 210, 160, 3).
        name (string): Name of the animation, default is test.
    Returns:
        None."""
    pass