# Deep reinforcement learning - Homework 03
## by Tim Niklas Witte, Bhaskar Majumder, Moritz Lönker 

An agent (DQN) is trained on the LunarLander enviroment of OpenAI Gym.
[LunarLander enviroment of OpenAI Gym](
https://www.gymlibrary.ml/environments/box2d/lunar_lander/).

For each episode the current performance metrics of the model such as average reward per episode, 
Score (sum of rewards per episode) are logged (tensorboard).


Note that, code from Tim Niklas Witte's final project of the IANNWTF course was taken to solve this task:
[Play Flappy Bird by applying Dueling Double Deep Q Learning](
https://github.com/schadenfreude2030/iannwtf-project)
In other words, code for the agent, Q-network, ReplayMemory and the EpsilonGreedyStrategy are reused.

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