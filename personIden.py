import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from person_identification_model import PersonIdentificationModel

import numpy as np

import matplotlib.pyplot as plt

class personIden():
    def __init__(self, input_shape):
        self.person_iden = PersonIdentificationModel(input_shape)
        self.person_iden.build(input_shape=input_shape)
        self.person_iden.compile(optimizer=tf.keras.optimizers.Adam(),
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=['accuracy'])

    def run(self, dataA, dataB, dataFalse):
        batch_size = 10

        #! 処理が激遅いからなんとかしたい
        dataA = tf.convert_to_tensor(dataA, dtype=np.float32)
        dataB = tf.convert_to_tensor(dataB, dtype=np.float32)
        dataFalse = tf.convert_to_tensor(dataFalse, dtype=np.float32)

        for i in range(25):
            F_idx = np.random.randint(0, dataFalse.shape[0], batch_size)
            A_idx = np.random.randint(0, dataA.shape[0], batch_size)

            true_loss = self.person_iden.train_on_batch(dataA.numpy()[A_idx], np.ones((batch_size, 1)))
            false_loss = self.person_iden.train_on_batch(dataFalse.numpy()[F_idx], np.zeros((batch_size, 1)))
            print("{}: true_loss {}, false_loss {}".format(i, true_loss, false_loss))

        r_list = self.person_iden.predict(dataB)
        r = [np.argmax(r) for r in r_list]
        print("result:", r)