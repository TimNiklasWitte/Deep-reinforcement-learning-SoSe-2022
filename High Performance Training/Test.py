from DQN import *
from multiprocessing import Process, Value, shared_memory

import numpy as np
import time

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class Test():

    def __init__(self):
        pass
        # self.net = DQN(5) 
        # self.net.build(input_shape=(1,5))


    def foo1(self):

        tf.keras.backend.clear_session()


        net = DQN(5) 
        net.build(input_shape=(1,5))

        while True:
            x = np.random.rand(1,5)
            y = net(x)
            print(y)
            time.sleep(1)

    # def foo2(self):

    #     net1 = DQN(5) 
    #     net1.build(input_shape=(1,5))

    #     while True:
    #         x = np.random.rand(1,5)
    #         y = net1.predict(x)
 

         

if __name__ == "__main__":

 

    test = Test()
    

    # print(test.list[0])

    procs = []
    for i in range(5):
        proc = Process(target=test.foo1)
        proc.start()
        procs.append(proc)

    


    proc.join()

    # print(test.list[0])
    # print("here")
    # print(test.x)
    #print(y.value)
    