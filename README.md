# suikaGameAI
WIP

Training a DeepQ network to play suikagame https://suikagame.io / https://suika.gg. I used DeepQ Tetris solvers as references because of their similarity in game, input, and output states.

suikasite.py - Reverse engineered the game and its use of the Cocos2d-x game engine to create an API for extracting game state data and controlling the game in a Selenium browser instance.
model.py - DeepQ network parameters
train.py - Training and evaluation of model
