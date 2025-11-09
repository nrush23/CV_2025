import argparse, os
from pong import Pong
import train

def main():
    """ Load Pong and our model to begin training

        Command Line Arguments:
        -f || --Frames --> type int, how many frames we run ALE Pong for (Default 10)
        -v || --View --> bool flag, do you want to see the human view (Default true)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--Frames", type=int, default=300,help="Number of frames to run the training game simulation")
    parser.add_argument("-v", "--View", action='store_true', help="Visualize the training game simulation")
    parser.add_argument("-p", "--Play", action='store_true', help="Determines whether the user plays or not")
    parser.add_argument("-e", "--Epsilon", type=float, default=0.01, help="Epsilon value for Computer to randomly choose a different move")
    parser.add_argument("-t", "--Train", action='store_true', help="Training mode")
    parser.add_argument("-ae", "--AutoEncod", type=int, default=20,help="Autoencoder epoch amount, default=20")
    parser.add_argument("-de", "--DiT", type=int, default=15,help="DiT epoch amount, default=15")
    parser.add_argument("-b", "--Batches", type=int, default=16,help="Batch size amount, default=16")
    parser.add_argument("-l", "--Load", action="store_true", help="Loads a DiT from dit_final.pth")
    args = parser.parse_args()

    #Command line arguments
    FRAMES = args.Frames
    VIEW = args.View
    PLAY = args.Play
    EPS = args.Epsilon
    TRAIN = args.Train
    AE = args.AutoEncod
    DE = args.DiT
    BATCHES = args.Batches
    LOAD = args.Load

    #Train to get model weights
    if TRAIN:
        ae, dit = train.train(FRAMES, AE, DE, BATCHES)
    #Use preexisting weights to generate frames
    elif LOAD:
        PATH = "checkpoints/dit_final.pth"
        try:
            WEIGHTS = open(PATH, "r")
        except IOError:
            raise FileNotFoundError(f"{PATH!r} not found")
    else:
        #Run the pong for FRAMES amount of frames (defaults to 10)
        game = Pong(VIEW, PLAY, EPS)
        game.simulate(FRAMES)


if __name__ == "__main__":
    main()