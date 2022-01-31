import tensorflow as tf
import tensorflow.keras.layers as L

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from person_identification_model import PersonIdentificationModel

class personIden():
    def __init__(self, input_shape):
        self.person_iden = PersonIdentificationModel(input_shape)
        input_shape = (None, input_shape[0], input_shape[1], input_shape[2])
        self.person_iden.build(input_shape=input_shape)
        self.person_iden.compile(optimizer=tf.keras.optimizers.Adam(),
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=['accuracy'])

    def run(self, dataA, dataB):
        self.person_iden.fit(dataA, 0, batch_size=16, epochs=10)

        self.person_iden(dataB)