from tkinter import Y
import tensorflow as tf

class PolicyNet(tf.keras.Model):
    def __init__(self):

        super(PolicyNet, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(8, kernel_size=(3, 3), strides=(3,3), padding="same", activation="tanh"),
            tf.keras.layers.Conv2D(10, kernel_size=(3, 3), strides=(3,3), padding="same", activation="tanh"),
            tf.keras.layers.Flatten()
        ]

        self.mu_0 = tf.keras.layers.Dense(1, activation='tanh')
        self.mu_1_2 = tf.keras.layers.Dense(2, activation='sigmoid')

        self.sigma_0 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.sigma_1_2 = tf.keras.layers.Dense(2, activation='sigmoid')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
   
    @tf.function
    def call(self, x: tf.Tensor):

        for layer in self.layer_list:
            x = layer(x)

        mu_0 = self.mu_0(x)
        mu_1_2 = self.mu_1_2(x)
 
        sigma_0 = self.sigma_0(x)
        sigma_1_2 = self.sigma_1_2(x)

        # reparameterization trick
        batch_size = x.shape[0]
        epsilon_0 = tf.random.normal(shape=(batch_size, 1))
        y_0 = mu_0 + epsilon_0 * sigma_0
        
        epsilon_2_3 = tf.random.normal(shape=(batch_size, 1))
        y_2_3 = mu_1_2 + epsilon_2_3 * sigma_1_2

        # clipping
        y_0 = tf.clip_by_value(y_0, -1, 1)
        y_2_3 = tf.clip_by_value(y_2_3, 0, 1)

        actions = tf.concat([y_0, y_2_3], axis=-1)

        # scale y_0 from [-1, 1] to [0, 1]
        y_0 = (y_0 + 1 + 0.000001) / 2 # avoid div by 0
        probs_actions = tf.concat([y_0, y_2_3], axis=-1)
        log_probs_actions = tf.math.log(probs_actions)

        return actions, log_probs_actions
            

    @tf.function
    def train_step(self, state, g_t):
        

        with tf.GradientTape() as tape:
            _, log_probs_actions = self(state)
            update = -tf.math.multiply(g_t[:, tf.newaxis], log_probs_actions)
        
        gradients = tape.gradient(update, self.trainable_variables)
 
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))




