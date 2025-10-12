# CV Project 2025
Computer Vision Fall 2025 project for generating frames of Pong or Tetris based off of key board actions in real time.

## Setup
1. Clone a copy of this repository to your local machine
2. Open a terminal and navigate to the CV_2025 folder
3. Once inside the CV_2025 folder, make a new python environment (this creates a new environment named venv):
    - ``` python -m venv venv ```
4. Activate your environment:
    - Windows: ```.\venv\Scripts\activate```
    - Mac: ```source venv/bin/activate```
5. Once your environment is activated, you should see:
    - Windows: ```(venv) C:\...your folder path...>```
    - Mac: ```(venv) ... $```
6. Now install the required libraries:
    - ```pip install -r requirements.txt```
7. Run commands:
    - Main file: `python main.py [-f FRAMES] [-v] [-p] [-h]`
        - *f: Frames amount*
        - *v: View in window*
        - *p: Player keyboard input mode*
        - *h: Help*
    - Any file: `python file_name.py`
8. When finished, deactivate your environment:
    - ```deactivate```
    
Add any additional required libraries to the requirements.txt file

## To Do (Need to do ASAP)

- [ ] Set up actual training loop. This should happen in the main or be called in there through an external class like train.py
    - This loop should instantiate our actual model, determine batch size, send and receive inputs to the Pong interface, and train for a specified amount of epochs.
    - Should track loss over time and generate a matplotlib graph with optional dashboarding from Pytorch.
    - Should have a section for inference which does not use gradients
- [ ] Extract frames from the Pong interface:
    - Right now, we have a Pong class which allows games to be visualized from computer choices or player choices as a demo. We need to implement additional separation or a new function to generate and return the next frame based on deciding the current action from the current frame.
- [ ] Finish implementing the model 

### What we've already dones
- [ x ] Basic Pong setup, does the following:
    - Allows keyboard input for future human playing
    - Has a mode to visualize actions in real time (for our understanding/tests)
    - Uses computer policy to automatically make best moves with a probability to do something random
        - When training, this will allow us to generate games with different levels of "expertise"
    - Has scaffold for using actions from the encoder if implemented in the future
- [ x ] First level ViT Encoder creation:
- [ x ] Created a main file:
    - Inside main.py, we define different command line arguments to parse the following arguments:
        - *f: Frames amount*
        - *v: View in window*
        - *p: Player keyboard input mode*
        - *e: Epsilon probability to make any random move
        - *h: Help*
    - By default, no training is happening here. This code just calls the Pong interface to view a game with FRAMES length, Computer or Player control, and with an Epsilon probability for the Computer mode.

## Similar Models
- Oasis Model: https://oasis-model.github.io/
- MineWorld: https://github.com/microsoft/mineworld
- Atari World Modeling: https://arxiv.org/pdf/2405.12399


