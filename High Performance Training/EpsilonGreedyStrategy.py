from multiprocessing import Value

class EpsilonGreedyStrategy:
    def __init__(self, start: float, end: float, decay: float):
        """
        Init EpsilonGreedyStrategy. Epsilon first value is start. 
        Calling reduce_epsilon will reduce Epsilon (multiplicative decay).
        Its min value will be end.

        Keyword arguments:
            start -- start value of Epsilon
            end -- min value of Epsilon
            decay -- multiplicative decay term
        """

        self.epsilon = Value('d', start)
        self.end = end

        self.decay = decay

    def get_exploration_rate(self):
        """
        Return the current Epsilon value
        """
        return self.epsilon.value

    def reduce_epsilon(self):
        
        """
        Multiplicative decay of Epsilon by a factor definied in decay parameter (see constructor).
        Note after Epsilon reached the min value (see end parameter in constructor), there will 
        be no decay.
        """

        new_epsilon = self.epsilon.value * self.decay
        if self.end < new_epsilon:
            self.epsilon.value = new_epsilon

       
