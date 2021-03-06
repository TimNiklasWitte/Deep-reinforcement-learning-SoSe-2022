# Deep reinforcement learning - Homework 02
## by Moritz Lönker and Tim Niklas Witte 

# Usage

```bash
python3 Main.py --help
usage: Main.py [-h] [--n N] [--MC] [--plotInterval PLOTINTERVAL]
               [--episodes EPISODES] [--lastPlotName LASTPLOTNAME]
               [--alpha ALPHA] [--gamma GAMMA]

Present the MC or n-step SARSA algorithm. After the episodes are done the
Q-Table is plotted: Each Q-value is presented.

optional arguments:
  -h, --help            show this help message and exit
  --n N                 Set the n in n step SARSA (default = 1).
  --MC                  Use MC Control with Exploring Starts instead of SARSA.
  --plotInterval PLOTINTERVAL
                        Set X. Each X times the current Q-Table will be
                        plotted and stored in
                        ./Plots/{n_sarsa}-step_SARSA/episode_{num_episode}.
                        The directory will be created if it does not exists.
                        (default = 0 -> create no plots).
  --episodes EPISODES   Set the number of episodes (default = 100).
  --lastPlotName LASTPLOTNAME
                        File name of the last plot.
  --alpha ALPHA         Set the learning rate (default = 0.01).
  --gamma GAMMA         Set the discount factor (default = 0.99).
```

# Enviroment


```bash
1 0 0 0 0 
0 0 0 0 0 
0 0 - 0 0 
0 0 0 ☐ 0 
0 0 0 0 $ 

Legend:
1 = Current position of the agent
0 = Empty, reward = 0
$ = Terminal state, reward = 1 
- = State to avoid, reward = -1
☐ = Obstacle
---------------------------------
```

# Evaluation
## 1-step SARSA

<img src="./GIFs/1-step_SARSA.gif" width="400" height="400">
<img src="./Plots/1-step_SARSA/episode_999.png" width="400" height="400">

## 2-step SARSA

<img src="./GIFs/2-step_SARSA.gif" width="400" height="400">
<img src="./Plots/2-step_SARSA/episode_999.png" width="400" height="400">

## 3-step SARSA

<img src="./GIFs/3-step_SARSA.gif" width="400" height="400">
<img src="./Plots/3-step_SARSA/episode_999.png" width="400" height="400">

## 4-step SARSA

<img src="./GIFs/4-step_SARSA.gif" width="400" height="400">
<img src="./Plots/4-step_SARSA/episode_999.png" width="400" height="400">

## 5-step SARSA

<img src="./GIFs/5-step_SARSA.gif" width="400" height="400">
<img src="./Plots/5-step_SARSA/episode_999.png" width="400" height="400">

## MC Control with Exploring Starts

<img src="./GIFs/MC_Control.gif" width="400" height="400">
<img src="./Plots/MC_Control/episode_19999.png" width="400" height="400">