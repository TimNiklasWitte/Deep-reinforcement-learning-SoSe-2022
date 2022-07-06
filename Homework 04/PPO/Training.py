import tensorflow as tf
import gym
import numpy as np

from EpisodeMemory import *
from PolicyNet import *
from ValueNet import *

def preprocess(state):

    state = np.cast['float32'](state)
    state = (state / 128.) - 1
    return state


def main():

    # Logging
    file_path = "test_logs/test"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    num_episodes = 1000
    episode_len = 2000
    gamma = 0.99
    target_kl = 0.01
    episodes_until_saveNet = 50

    env = gym.make("CarRacing-v1")
    state = env.reset()

    policy_net = PolicyNet()
    policy_net.build((1, *env.observation_space.shape))
    policy_net.summary()

    value_net = ValueNet()
    value_net.build((1, *env.observation_space.shape))
    value_net.summary()


    episodeMemory = EpisodeMemory(episode_len, env.observation_space.shape, env.action_space.shape[0])

    for num_episode in range(num_episodes):
        
        rewards = [] # evaluation (later...)
  
        state = env.reset()
        for i in range(episode_len):

            state = preprocess(state)
            state = np.expand_dims(state, axis=0)
            
            action, log_prob = policy_net(state)
            action = action.numpy()[0] # convert to numpy + remove batch dim
            log_prob = log_prob.numpy()[0] # convert to numpy + remove batch dim


            next_state, reward, done , _ = env.step(action)

            rewards.append(reward)

            next_state = preprocess(next_state)
            next_state = np.expand_dims(next_state, axis=0)
            value_next_state = value_net(next_state)

            value_state = value_net(state)

            # A(s,a) = Q(s,a) - V(s) 
            #        = r_t + gamma*V(s_(t+1)) - V(s_t)

            advantage = reward + gamma*value_next_state - value_state
            

            # V(s) = r_t + gamma*V(s_(t+1))
            values_target = reward + gamma*value_net(next_state)

            episodeMemory.store(state, values_target, log_prob, advantage)
            state = next_state


    
        #
        # Create dataset
        #
        states = episodeMemory.states
        values_target = episodeMemory.values_target
        log_probs = episodeMemory.log_probs
        advantages = episodeMemory.advantages 

        dataset = tf.data.Dataset.from_tensor_slices((states, values_target, log_probs, advantages))
        dataset = dataset.shuffle(episode_len)
        dataset = dataset.batch(64, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        #
        # Train policy net
        #
        for states, _, log_probs, advantages in dataset:
            kl = policy_net.train_step(states, log_probs, advantages)

            # Early stopping mechanism
            if kl > 1.5 * target_kl:
                break
    
        # 
        # Train value net
        #
        for state, values_target, _, _ in dataset.take(1):
            loss = value_net.train_step(state, values_target)


        episodeMemory.reset()

        #
        # Evalation
        #
        
        score = np.sum(rewards)
        avg_reward = np.mean(rewards)

        print(f"   Episode: {num_episode}")
        print(f"     Score: {score}")
        print(f"Avg reward: {avg_reward}")
        print(f"      Loss: {loss}")
        print("------------------------")

        #
        # Log
        #
        loss = loss.numpy()
        with train_summary_writer.as_default():
            tf.summary.scalar(f"Score", score, step=num_episode)
            tf.summary.scalar(f"Average reward", avg_reward, step=num_episode)
            tf.summary.scalar(f"Loss", loss, step=num_episode)

        if num_episode % episodes_until_saveNet == 0:
            policy_net.save_weights(f"./saved_models/policy_net/trained_weights_{num_episode}", save_format="tf")
            value_net.save_weights(f"./saved_models/value_net/trained_weights_{num_episode}", save_format="tf")
     


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")