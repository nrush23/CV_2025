import argparse
from pong import Pong

def main():
    """ Load Pong and our model to begin training

        Command Line Arguments:
        -f || --Frames --> type int, how many frames we run ALE Pong for (Default 10)
        -v || --View --> bool flag, do you want to see the human view (Default true)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--Frames", type=int, help="Number of frames to run the training game simulation")
    parser.add_argument("-v", "--View", action='store_false', help="Visualize the training game simulation")
    parser.add_argument("-p", "--Play", action='store_true', help="Determines whether the user plays or not")
    parser.add_argument("-e", "--Epsilon", type=float, help="Epsilon value for Computer to randomly choose a different move")
    args = parser.parse_args()

    #Command line arguments
    FRAMES = args.Frames if args.Frames else 10
    VIEW = args.View
    PLAY = args.Play if args.Play else False
    EPS = args.Epsilon if args.Epsilon else 0.01

    #Run the pong for FRAMES amount of frames (defaults to 10)
    game = Pong(VIEW, PLAY, EPS)
    game.simulate(FRAMES)

    # TODO: Add our simulation loop code here, this should initialize our encoders and pass Pong frames to them

if __name__ == "__main__":
    main()