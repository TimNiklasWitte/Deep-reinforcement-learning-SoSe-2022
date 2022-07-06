import gym
import numpy as np

from PolicyNet import *
from ValueNet import *

import tqdm

def preprocess(state):

    state = np.cast['float32'](state)
    state = (state / 128.) - 1
    return state


def main():

    train_steps = 10000
    steps_until_reset = 100
    steps_until_evalution = 100
    evaluation_steps = 100
    steps_until_modelSave = 1000

    # Logging
    file_path = "test_logs/test"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    batch_size = 32
    gamma = 0.99

    env = gym.vector.make("CarRacing-v1", num_envs=batch_size)

    policy_net = PolicyNet()
    policy_net.build(env.observation_space.shape)
    policy_net.summary()

    value_net = ValueNet()
    value_net.build(env.observation_space.shape)
    value_net.summary()

  
    states = env.reset()
    
    for step_num in range(train_steps):
        
        states = preprocess(states)
        actions, _ = policy_net(states)

        actions = actions.numpy()
        
        next_states, rewards, dones , _ = env.step(actions)

        #
        # Train policy net
        #

        # A(s,a) = Q(s,a) - V(s) 
        #        = r_t + gamma*V(s_(t+1)) - V(s_t)

        next_states = preprocess(next_states)
        rewards = np.expand_dims(rewards, axis=-1)
        advantages = rewards + gamma*value_net(next_states) - value_net(states)

        policy_net.train_step(states, advantages)

        # 
        # Train value net
        #
        values_target = rewards + gamma*value_net(next_states)
        loss = value_net.train_step(states, values_target)
     

        states = next_states
        if np.any(dones) or step_num % steps_until_reset == 0:
            states = env.reset()


        #
        # Log
        #

        # Evalation
        if step_num % steps_until_evalution == 0:
            env_test = gym.make("CarRacing-v1")
            state = env_test.reset()

            rewards = []
            score = 0
            for test_run_step in range(evaluation_steps):
                state = preprocess(state)
                state = np.expand_dims(state, axis=0)
                action, _ = policy_net(state)
                action = action.numpy()
                action = action[0]
                
                next_state, reward, done , _ = env_test.step(action)

                state = next_state
                rewards.append(reward)
                score += reward
                if done:
                    break 
            
            env_test.close()
            avg_reward = np.mean(rewards)

            
            print(f" Iteration: {step_num}")
            print(f"     Score: {score}")
            print(f"Avg reward: {avg_reward}")
            print(f"      Loss: {loss}")
            print("------------------------")

            loss = loss.numpy()
            with train_summary_writer.as_default():
                tf.summary.scalar(f"Score", score, step=step_num)
                tf.summary.scalar(f"Average reward", avg_reward, step=step_num)
                tf.summary.scalar(f"Loss", loss, step=step_num)
            

        if step_num % steps_until_modelSave == 0:
            policy_net.save_weights(f"./saved_models/policy_net/trained_weights_{step_num}", save_format="tf")
            value_net.save_weights(f"./saved_models/value_net/trained_weights_{step_num}", save_format="tf")
   


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")