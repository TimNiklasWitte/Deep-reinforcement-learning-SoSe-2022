import tensorflow as tf

class ValueNet(tf.keras.Model):
    def __init__(self):

        super(ValueNet, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(4, kernel_size=(3, 3), strides=(3,3), padding="same", activation="tanh"),
            tf.keras.layers.Conv2D(8, kernel_size=(3, 3), strides=(3,3), padding="same", activation="tanh"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation=None)
        ]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss_function = tf.keras.losses.MeanSquaredError()
    

        self.metric = tf.keras.metrics.Mean(name="loss")

    @tf.function
    def call(self, x: tf.Tensor):

        for layer in self.layer_list:
            x = layer(x)

        return x
            

    @tf.function
    def train_step(self, states, values_target):
        
        with tf.GradientTape() as tape:
            values_prediction = self(states)
            loss = self.loss_function(values_prediction, values_target)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric.update_state(loss)
        return self.metric.result()
        