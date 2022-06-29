import tensorflow as tf

class PolicyNet(tf.keras.Model):
    def __init__(self):

        super(PolicyNet, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(3,3), padding="same", activation="tanh"),
            #tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(3,3), padding="same", activation="tanh"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3, activation='sigmoid')
        ]
 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
   
    @tf.function
    def call(self, x: tf.Tensor):

        for layer in self.layer_list:
            x = layer(x)

        return x
            

    @tf.function
    def train_step(self, state, g_t):
        

        with tf.GradientTape() as tape:
            actions = self(state)
            update = tf.math.multiply(-g_t[:, tf.newaxis], tf.math.log(actions))
        
        gradients = tape.gradient(update, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))




