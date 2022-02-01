import tensorflow as tf
import tensorflow.keras.layers as L

# Person identification model (Simple CNN)
class PersonIdentificationModel(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()

        self.conv1 = L.Conv2D(16, 3, activation='relu', input_shape=input_shape)
        self.conv2 = L.Conv2D(16, 3, activation='relu')
        self.conv3 = L.Conv2D(32, 3, activation='relu')

        self.mp = L.MaxPool2D((2, 2), padding='same')

        self.fit = L.Flatten()

        self.dense = L.Dense(1024, activation='relu')
        self.op_dense = L.Dense(1, activation='softmax')

    def call(self, x):
        h1 = self.mp(self.conv1(x))
        h2 = self.mp(self.conv2(h1))
        h3 = self.mp(self.conv3(h2))

        h4 = self.dense(self.fit(h3))
        return self.op_dense(h4)