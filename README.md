# CV Project 2025
Computer Vision Fall 2025 project for generating frames of Pong or Tetris based off of key board actions in real time.

## Setup
- Clone a copy of this repository to your local machine
- Open a terminal and navigate to the CV_2025 folder
- Once inside the CV_2025 folder, make a new python environment (this creates a new environment named venv):
    - python -m venv venv
- Activate your environment:
    - Windows: .\venv\Scripts\activate
    - Mac: source venv/bin/activate
- Once your environment is activated, you should see:
    - Windows: (venv) C:\...your folder path...
    - Mac: (venv) ... $
- Now install the required libraries:
    - pip install -r requirements.txt
    
Add any additional required libraries to the requirements.txt file

## To Do (Try to have basic setups for next weekend)

- [ ] Set up a Pong model to be used within our model. This should have an API to be used within our pipeline to move the paddle.
    - ALE Gymnaisum Pong API: https://ale.farama.org/environments/pong/
    - ALE models: https://ale.farama.org/environments/

- [ ] Pick our transformer encoder model and implement in using PyTorch
    - ViT

- [ ] Pick our transformer decoder model and implement using PyTorch.
    - DiT

- [ ] Get comfortable on HPC Greene and do a test train.

## Similar Models
- Oasis Model: https://oasis-model.github.io/
- MineWorld: https://github.com/microsoft/mineworld


