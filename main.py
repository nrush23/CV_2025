import argparse, os
from pong import Pong
import train
from pipeline import Pipeline

def main():
    """ Load Pong and our model to begin training

        Command Line Arguments:
        -f || --Frames --> type int, how many frames we run ALE Pong for (Default 300)
        -v || --View --> bool flag, do you want to see the human view (Default false)
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
        #------ ORIGINAL SYNTAX -----#
        # ae, dit = train.train(FRAMES, AE, DE, BATCHES)
        #----------------------------#

        #---------- TESTING ---------#
        pipeline = Pipeline()
        pipeline.train(FRAMES, AE, DE, BATCHES, save_dir='testing')
        #----------------------------#
        
    #Use preexisting weights to generate frames
    elif LOAD:
        DIT_PATH = "checkpoints/dit_final.pth"
        AE_PATH = "checkpoints/best_autoencoder.pth"
        try:
            #Load weights for the autoencoder
            AE_TRAINER = train.AutoencoderTrainer()
            AE_TRAINER.load(AE_PATH)

            #Load weights for the DiT
            DIT_TRAINER = train.DiTTrainer(AE_TRAINER.autoencoder.encoder)
            DIT_TRAINER.load(DIT_PATH)

        except IOError:
            raise FileNotFoundError()
    else:
        #Run the pong for FRAMES amount of frames (defaults to 300)
        game = Pong(VIEW, PLAY, EPS)
        game.simulate(FRAMES)


if __name__ == "__main__":
    main()