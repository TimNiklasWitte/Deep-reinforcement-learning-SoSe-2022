from multiprocessing import Process, shared_memory
import gym
import numpy as np
from numpy.random import default_rng

from PolicyNet import *

import cv2

import matplotlib.pyplot as plt

def preprocess(state):

    state = np.cast['float32'](state)
    state = (state / 128.) - 1
    return state

def main():

    # Logging
    file_path = "test_logs/test"
    train_summary_writer = tf.summary.create_file_writer(file_path)

 
    env = gym.make("CarRacing-v0")
    env.render()

    env1 = gym.make("CarRacing-v0")
    env1.render()


    input()
    policy_net = PolicyNet()
    policy_net.build((1, *env.observation_space.shape))
    policy_net.summary()
    
    BUFF_SIZE = 200
    BATCH_SIZE = 32

    gamma = 0.99
    
    for num_iteration in range(100):

        states = np.zeros(shape=(BUFF_SIZE, *env.observation_space.shape), dtype=np.uint8)
        actions = np.zeros(shape=(BUFF_SIZE, *env.action_space.shape), dtype=np.float32)
        returns = np.zeros(shape=(BUFF_SIZE), dtype=np.float32)
        
        #
        # Generate samples
        #

        print("generate data")
        idx = 0
        while idx < BUFF_SIZE:
            done_float = False 

            rewards = []

            state = env.reset()
            env.render()
            while not done_float:

                state = preprocess(state)
                state = np.expand_dims(state, axis=0)
                action = policy_net(state)
                action = action[0].numpy() # remove batch dim and convert to numpy
                action[0] = 2*action[0] - 1
                next_state, reward, done_flag, _ = env.step(action)
                env.render()
                rewards.append(reward)

                random_number = np.random.random(1)
                if random_number < 0.1:
                    
                    # calc return
                    for j in range(len(rewards)):
                        g_t = 0
                        for i in range(len(rewards[j:])):
                            g_t += (gamma**i) * rewards[i]

                    #img = env.render(mode='rgb_array')

                    states[idx] = state #cv2.resize(img, (96,96))
                    actions[idx] = action
                    returns[idx] = g_t
                    
                    #print(next_state)
                    # plt.imshow(next_state)
                    # plt.show()
                    #env.render()
                    #print(idx)
                    idx += 1
                    

                state = next_state

                if BUFF_SIZE < idx or idx % 100 == 0:
                    break
        
        #
        # Evaluation
        #
        print("Evalation")
        done_float = False 

        rewards = []
        score = 0
        cnt_steps = 0

        env.render()
        state = env.reset()

        while not done_float:
            state = preprocess(state)
            state = np.expand_dims(state, axis=0)
            action = policy_net(state)
            action = action[0].numpy() # remove batch dim and convert to numpy
            action[0] = 2*action[0] - 1
            next_state, reward, done_flag, _ = env.step(action)

            env.render() # mode='rgb_array'

            rewards.append(reward)
            score += reward
            cnt_steps += 1
            state = next_state

            

            if 1000 < cnt_steps:
                break

        print(f"  Episode: {num_iteration}")
        print(f"    Score: {round(score, 2)}")
        print(f"Avg Score: {round(np.mean(rewards), 2)}")
        print(f"    Steps: {cnt_steps}")
        print("------------------------") 

        with train_summary_writer.as_default():
            tf.summary.scalar(f"Average reward", np.mean(rewards), step=num_iteration)
            tf.summary.scalar(f"Score", score, step=num_iteration)
            tf.summary.scalar(f"Steps per episode", cnt_steps, step=num_iteration)

        #
        # Generate samples
        #
        print("Training")
        rng = default_rng()
        batch_indices = rng.choice(BUFF_SIZE, size=BATCH_SIZE, replace=False)

        sampled_states = states[batch_indices, ...]
        sampled_states = preprocess(sampled_states)

        # print(sampled_states[0])
        # plt.imshow(sampled_states[0])
        # plt.show()

        sampled_returns = returns[batch_indices]

        policy_net.train_step(sampled_states, sampled_returns)
    
   


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
