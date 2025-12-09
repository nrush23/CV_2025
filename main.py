import argparse
from pong import Pong
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
    parser.add_argument("-i", "--Inferences", type=int, default=90,help="Number of frames to run inference.")
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
    INFERENCES = args.Inferences
    GAME = not (TRAIN or LOAD)

    #Play the game if not training or loading (for now)
    if GAME:
        game = Pong(VIEW, PLAY, EPS)
        game.simulate(FRAMES)
    else:
    #Make our pipeline and either train for weights or load from checkpoints
        pipeline = Pipeline()
        if LOAD:
            DIT_PATH = 'reloading/dit_final.pth'
            AE_PATH = 'testing/best_autoencoder.pth'
            pipeline.load_weights(AE_PATH, DIT_PATH)
        if TRAIN:
            pipeline.train(FRAMES, AE, DE, BATCHES, save_dir='seed_test')
        
        #Run inference on one frame
        if PLAY:
            pipeline.inference_interactive()
        else:
            pipeline.inference(INFERENCES)



if __name__ == "__main__":
    main()
