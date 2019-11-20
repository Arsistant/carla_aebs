import sys
import os
import cv2
import carla
from carla import Image
import numpy as np
from keras.models import model_from_json
import tensorflow as tf


sess = tf.compat.v1.InteractiveSession()

class DistanceCalculation():
    def __init__(self, ego_vehicle, leading_vehicle, perception=None):
        self.ego_vehicle = ego_vehicle
        self.leading_vehicle = leading_vehicle
        self.perception = perception
        if perception is not None:
            with open(os.path.join(perception, "perception_architecture.json"), 'r') as f:
                self.model = model_from_json(f.read())
            self.model.load_weights(os.path.join(perception, 'perception_weights.h5'))


    def getTrueDistance(self):
        distance = self.leading_vehicle.get_location().y - self.ego_vehicle.get_location().y \
                - self.ego_vehicle.bounding_box.extent.x - self.leading_vehicle.bounding_box.extent.x
        return distance 
    
    def getRegressionDistance(self, image):
        if self.perception is not None:
            img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            img = np.reshape(img, (image.height, image.width, 4))
            img = cv2.resize(img, (224,224))
            img = img[:, :, :3]/255.
            #img = img[:, :, ::-1]/255.
            img = np.expand_dims(img,axis=0)
            distance = self.model.predict(img)*120.0
            return float(distance[0][0])
        return None
        