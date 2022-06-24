import tensorflow as tf

class PolicyNet(tf.keras.Model):
    def __init__(self, num_actions: int):

        super(PolicyNet, self).__init__()

        self.layer_list = [
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(num_actions, activation="softmax"),
        ]

 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.MeanSquaredError()

    @tf.function
    def call(self, x: tf.Tensor):

        for layer in self.layer_list:
            x = layer(x)
        
        return x
            

    @tf.function
    def train_step(self, state, g_t):
        

        with tf.GradientTape() as tape:
            actions = self(state, training=True)
            update =  -g_t * tf.math.log (actions)
      

        gradients = tape.gradient(update, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))




