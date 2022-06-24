# Deep reinforcement learning - Homework 03
## by Tim Niklas Witte, Bhaskar Majumder, Moritz Lönker 

An agent (DQN) is trained on the [LunarLander enviroment of OpenAI Gym](
https://www.gymlibrary.ml/environments/box2d/lunar_lander/).

For each episode the current performance metrics of the model such as average reward per episode, 
Score (sum of rewards per episode) are logged (tensorboard).


Note that, code from Tim Niklas Witte's final project of the IANNWTF course was reused to solve this task:
[Play Flappy Bird by applying Dueling Double Deep Q Learning](
https://github.com/schadenfreude2030/iannwtf-project)
In other words, code for the training, agent, Q-network, ReplayMemory and the EpsilonGreedyStrategy are reused and slightly changed for solving this task.

# Network

```bash
Model: "dqn"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               multiple                  576       
                                                                 
 dense_1 (Dense)             multiple                  8320      
                                                                 
 dense_2 (Dense)             multiple                  516       
                                                                 
=================================================================
Total params: 9,412
Trainable params: 9,412
Non-trainable params: 0
_________________________________________________________________
```

# Usage
Run `Training.py` to start the training:

```bash
python3 Training.py
```

# Evaluation
## Training
<img src="./media/trainingPlot.png" width="1000" height="300">


## Development of the policy
### 0 Episodes

<img src="./media/GIFs/episode_0.gif" width="400" height="400">

### 100 Episodes

<img src="./media/GIFs/episode_100.gif" width="400" height="400">

### 200 Episodes

<img src="./media/GIFs/episode_200.gif" width="400" height="400">

### 300 Episodes

<img src="./media/GIFs/episode_300.gif" width="400" height="400">

### 400 Episodes

<img src="./media/GIFs/episode_400.gif" width="400" height="400">

### 500 Episodes

<img src="./media/GIFs/episode_500.gif" width="400" height="400">

### 600 Episodes

<img src="./media/GIFs/episode_600.gif" width="400" height="400">