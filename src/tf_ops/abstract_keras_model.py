import tensorflow as tf
from abc import ABCMeta, abstractclassmethod

class BaseKerasModel(tf.keras.Model, metaclass=ABCMeta):
    
    @abstractclassmethod
    def call(self):
        pass

    @abstractclassmethod
    def serve(self):
        pass