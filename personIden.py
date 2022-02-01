import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from person_identification_model import PersonIdentificationModel

import numpy as np

class personIden():
    def __init__(self, input_shape):
        self.person_iden = PersonIdentificationModel(input_shape)
        self.person_iden.build(input_shape=input_shape)
        self.person_iden.compile(optimizer=tf.keras.optimizers.Adam(),
                                loss=tf.keras.losses.BinaryCrossentropy(),
                                metrics=['accuracy'])

    def run(self, dataA, dataB):
   
        dataA = tf.convert_to_tensor(dataA, dtype=tf.float32)
#!============================================= TEST =============================================

#!============================================= TEST =============================================
        dataB = tf.convert_to_tensor(dataB, dtype=tf.float32)
        label = tf.convert_to_tensor([0]*len(dataA)) 

        for i in range(10):
            loss = self.person_iden.train_on_batch(dataA, label)
            print("{}: loss {}".format(i, loss))

        print(self.person_iden.metrics_names)

        r = self.person_iden.predict(dataB)
        print("result:", r)