import tensorflow as tf

class DQN(tf.keras.Model):
    def __init__(self, num_actions: int):
        """Init the DQN. 

        Keyword arguments:
        num_actions -- Number of possible actions which can be taken in the gym.
        """
        super(DQN, self).__init__()

        self.layer_list = [
            tf.keras.layers.Dense(32, activation='tanh'),
            tf.keras.layers.Dense(num_actions, activation=None),
        ]

 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.MeanSquaredError()

    @tf.function
    def call(self, x: tf.Tensor):
        """Forward pass through the network. 

        Keyword arguments:
        x -- network input

        Return:
        network output
        """

        for layer in self.layer_list:
            x = layer(x)
        
        return x
            

    @tf.function
    def train_step(self, x: tf.Tensor, target: tf.Tensor):
        """Train the network based on input and target,

        Keyword arguments:
        x -- network input
        target -- target
        """
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.loss_function(target, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
