import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter


def main():
    
    # Reinforce
    reinforce_df_avg_reward = pd.read_csv('./Reinforce/TensorBoard_csv_files/run-.-tag-Average reward.csv', sep=',')
    reinforce_avg_reward = reinforce_df_avg_reward["Value"]

    reinforce_df_score = pd.read_csv('./Reinforce/TensorBoard_csv_files/run-.-tag-Score.csv', sep=',')
    reinforce_score = reinforce_df_score["Value"]

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax = plot_avgReward(ax, reinforce_avg_reward)
    ax = plot_score(ax, reinforce_score)
    plt.suptitle("Training based on Reinforce")
    plt.savefig("./media/TrainingPlot_Reinforce.png")

    # A2C
    a2c_df_avg_reward = pd.read_csv('./A2C multiple env/TensorBoard_csv_files/run-.-tag-Average reward.csv', sep=',')
    a2c_avg_reward = a2c_df_avg_reward["Value"]

    a2c_df_score = pd.read_csv('./A2C multiple env/TensorBoard_csv_files/run-.-tag-Score.csv', sep=',')
    a2c_score = a2c_df_score["Value"]

    a2c_df_loss = pd.read_csv('./A2C multiple env/TensorBoard_csv_files/run-.-tag-Loss.csv', sep=',')
    a2c_loss = a2c_df_loss["Value"]

    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax = plot_avgReward(ax, a2c_avg_reward)
    ax = plot_score(ax, a2c_score)
    ax = plot_loss(ax, a2c_loss)
    
    plt.suptitle("Training based on A2C")
    fig.set_size_inches(w=17.5, h=4)

    plt.savefig("./media/TrainingPlot_A2C.png")

    # PPO
    ppo_df_avg_reward = pd.read_csv('./PPO/TensorBoard_csv_files/run-.-tag-Average reward.csv', sep=',')
    ppo_avg_reward = ppo_df_avg_reward["Value"]

    ppo_df_score = pd.read_csv('./PPO/TensorBoard_csv_files/run-.-tag-Score.csv', sep=',')
    ppo_score = ppo_df_score["Value"]

    ppo_df_loss = pd.read_csv('./PPO/TensorBoard_csv_files/run-.-tag-Loss.csv', sep=',')
    ppo_loss = ppo_df_loss["Value"]

    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax = plot_avgReward(ax, ppo_avg_reward)
    ax = plot_score(ax, ppo_score)
    ax = plot_loss(ax, ppo_loss)
    
    plt.suptitle("Training based on PPO")
    fig.set_size_inches(w=17.5, h=4)

    plt.savefig("./media/TrainingPlot_PPO.png")
  
  

def plot_avgReward(ax, avg_reward):
    ax[0].plot(avg_reward, alpha=0.5, label="not smoothed")
    avg_reward_smoothed = savgol_filter(avg_reward, 41, 3)
    ax[0].plot(avg_reward_smoothed, label="smoothed")
    ax[0].legend(loc="lower right")

    ax[0].set_title("Average reward per episode")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Average reward")
    ax[0].grid(True)

    return ax

def plot_score(ax, score):
    ax[1].plot(score, alpha=0.5, label="not smoothed")
    score_smoothed = savgol_filter(score, 41, 3)
    ax[1].plot(score_smoothed, label="smoothed")

    ax[1].set_title("Score per episode")
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Score")
    ax[1].grid(True)
    ax[1].legend(loc="lower right")

    return ax

def plot_loss(ax, loss):
    ax[2].plot(loss)
    ax[2].set_title("Critic loss per episode")
    ax[2].set_xlabel("Episode")
    ax[2].set_ylabel("Critic loss")
    ax[2].grid(True)

    return ax
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")