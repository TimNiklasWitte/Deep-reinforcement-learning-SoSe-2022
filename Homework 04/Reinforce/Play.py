import gym

def main():
    env = gym.make("CarRacing-v1")
    env.reset()
    env.render()

    

    for i in range(1500):
        actions = env.action_space.sample()
        env.step(actions)

        env.render()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")