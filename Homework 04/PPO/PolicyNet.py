import tensorflow as tf

class PolicyNet(tf.keras.Model):
    def __init__(self):

        super(PolicyNet, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(4, kernel_size=(3, 3), strides=(3,3), padding="same", activation="tanh"),
            tf.keras.layers.Conv2D(8, kernel_size=(3, 3), strides=(3,3), padding="same", activation="tanh"),
            tf.keras.layers.Flatten()
        ]

        # steering [-1, 1]
        # gas [0, 1]
        # breaking [0, 1]
        self.mu_steering = tf.keras.layers.Dense(1, activation='tanh')
        self.mu_gas_breaking = tf.keras.layers.Dense(2, activation='sigmoid')

        self.sigma_steering = tf.keras.layers.Dense(1, activation='sigmoid')
        self.sigma_gas_breaking = tf.keras.layers.Dense(2, activation='sigmoid')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
   
    @tf.function
    def call(self, x: tf.Tensor):

        for layer in self.layer_list:
            x = layer(x)

        
        mu_steering = self.mu_steering(x)
        mu_gas_breaking = self.mu_gas_breaking(x)
 
        sigma_steering = self.sigma_steering(x)
        sigma_gas_breaking = self.sigma_gas_breaking(x)

        # reparameterization trick
        batch_size = x.shape[0]
        # epsilon = tf.random.normal(shape=(batch_size, 1))
        # action_steering = mu_steering + epsilon * sigma_steering
        action_steering = tf.random.normal(shape=(batch_size, 1), mean=mu_steering, stddev=sigma_steering)
        # epsilon = tf.random.normal(shape=(batch_size, 2))
        # action_gas_breaking = mu_gas_breaking + epsilon * sigma_gas_breaking
        action_gas_breaking = tf.random.normal(shape=(batch_size, 2), mean=mu_gas_breaking, stddev=sigma_gas_breaking)
        # clipping
        action_steering = tf.clip_by_value(action_steering, -1, 1)
        action_gas_breaking = tf.clip_by_value(action_gas_breaking, 0, 1)

        actions = tf.concat([action_steering, action_gas_breaking], axis=-1)

        # scale y_0 from [-1, 1] to [0, 1]
        action_steering = (action_steering + 1) / 2
        probs_actions = tf.concat([action_steering + 0.0001, action_gas_breaking + 0.0001], axis=-1)
        log_probs_actions = tf.math.log(probs_actions)

        return actions, log_probs_actions
            

    #@tf.function
    def train_step(self, states, log_probs, advantages):
        
        epsilon = 0.2 
       

        with tf.GradientTape() as tape:
            
            _ , currentPolicy_log_prob = self(states)
            oldPolicy_log_prob = log_probs

            ratio = tf.exp( currentPolicy_log_prob - oldPolicy_log_prob)
            clipped_ratio = tf.clip_by_value (ratio, 1 - epsilon, 1 + epsilon)

            # Three probs -> one prob by using mean
            ratio = tf.reduce_mean(ratio, axis=-1)
            clipped_ratio = tf.reduce_mean(clipped_ratio, axis=-1)

            update = -tf.reduce_mean(
                tf.minimum(clipped_ratio*advantages, ratio*advantages)
            )

        gradients = tape.gradient(update, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        kl = tf.reduce_mean(oldPolicy_log_prob - currentPolicy_log_prob)
        kl = tf.reduce_sum(kl)
        return kl