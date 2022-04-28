import imageio

def main():
    n_sarsa = 5
    file_names = [f"./Plots/{n_sarsa}-step_SARSA/episode_{num_episode}.png" for num_episode in range(0, 2000, 50) ]
    with imageio.get_writer(f"./GIFs/{n_sarsa}-step_SARSA.gif", mode='I') as writer:
        for filename in file_names:
            image = imageio.imread(filename)
            writer.append_data(image)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")