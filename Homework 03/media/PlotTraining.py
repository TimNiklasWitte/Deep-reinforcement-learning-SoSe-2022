import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

def main():
    
    df_avg_reward = pd.read_csv('../TensorBoard_csv_files/run-test-tag-Average reward.csv', sep=',')
    avg_reward = df_avg_reward["Value"]
  
    df_epsilon = pd.read_csv('../TensorBoard_csv_files/run-test-tag-Epsilon (EpsilonGreedyStrategy).csv', sep=',')
    epsilon = df_epsilon["Value"]

    df_score = pd.read_csv('../TensorBoard_csv_files/run-test-tag-Score.csv', sep=',')
    score = df_score["Value"]

    df_stepsPerEpisode = pd.read_csv('../TensorBoard_csv_files/run-test-tag-Steps per episode.csv', sep=',')
    stepsPerEpisode = df_stepsPerEpisode["Value"]

    x = np.arange(len(epsilon))

    fig, ax = plt.subplots(nrows=1, ncols=4)
    ax[0].plot(x, avg_reward, alpha=0.5, label="not smoothed")
    avg_reward_smoothed = savgol_filter(avg_reward, 51, 3)
    ax[0].plot(x, avg_reward_smoothed, label="smoothed")
    ax[0].legend(loc="lower right")

    ax[0].set_title("Average reward per episode")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Average reward")
    ax[0].grid(True)

    ax[1].plot(x, epsilon)
    ax[1].set_title("Epsilon Greedy Strategy")
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Epsilon")
    ax[1].grid(True)

    ax[2].plot(x, score, alpha=0.5, label="not smoothed")
    score_smoothed = savgol_filter(score, 51, 3)
    ax[2].plot(x, score_smoothed, label="smoothed")

    ax[2].set_title("Score per episode")
    ax[2].set_xlabel("Episode")
    ax[2].set_ylabel("Score")
    ax[2].grid(True)
    ax[2].legend(loc="lower right")

    ax[3].plot(x, stepsPerEpisode, alpha=0.5, label="not smoothed")
    stepsPerEpisode_smoothed = savgol_filter(stepsPerEpisode, 51, 3)
    ax[3].plot(x, stepsPerEpisode_smoothed, label="smoothed")

    ax[3].set_title("Steps per episode")
    ax[3].set_xlabel("Episode")
    ax[3].set_ylabel("Steps")
    ax[3].grid(True)
    ax[3].legend(loc="lower right")

    #plt.tight_layout()
    
    fig.set_size_inches(w=23, h=4)
    plt.savefig("../media/trainingPlot.png", bbox_inches='tight')
    plt.show()

    



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")