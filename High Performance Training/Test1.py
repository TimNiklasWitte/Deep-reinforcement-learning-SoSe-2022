import tensorflow as tf
from multiprocessing.pool import Pool

class Frog():
    def __init__(self, noise, num_legs=4):
        self.noise = noise
        self.num_legs = num_legs
        self.some_var = tf.Variable(42)

    def make_noise(self, num_times):
        print(self.noise * num_times)


def check_legs(noise, num_legs, num_times):
    frog = Frog(noise, num_legs)
    if frog.num_legs < 4:
        print("Help, I only got {} legs".format(frog.num_legs))
    
    frog.make_noise(num_times)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.initializers.global_variables())
        print(sess.run(frog.some_var))

if __name__ == "__main__":


    with tf.compat.v1.Session() as sess:
        pass

    with tf.compat.v1.Session() as sess:
        pass

    # num_legs_list = range(2, 6)
    # num_times_list = range(1, 5)
    # args = [(
    #     'ribbet', num_legs, num_times
    # ) for num_legs, num_times in zip(num_legs_list, num_times_list)]
    # with Pool() as p, tf.compat.v1.Session() as sess:
    #     p.starmap(check_legs, args)