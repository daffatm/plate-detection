import tensorflow as tf
import cv2
import numpy as np
import argparse

from rtsr.data import visualize_samples
from rtsr.models import Generator


class SuperResolution():
    def __init__(self):
        self.generator = Generator()
        self.generator.load_weights('./rtsr/weights/GeneratorVG4(1520).h5')
        print("Super Resolution model loaded.")
        
    
    def enhance_image(self, image=None, size=(20, 16)):
        cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lr_image = tf.convert_to_tensor(cv_rgb, dtype=tf.float32)

        sr_image = self.generator(tf.expand_dims(lr_image, 0), training = False)[0]
        sr_image = tf.clip_by_value(sr_image, 0, 255)
        sr_image = tf.round(sr_image)
        sr_image = tf.cast(sr_image, tf.uint8)

        # Convert TensorFlow tensor to opencv
        np_image = sr_image.numpy()
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        np_image = np_image.astype(np.uint8)

        return np_image
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, 
                        default=None, 
                        help ='Path to the low resolution image.')
    args = parser.parse_args()
    
    enhancer = SuperResolution()
    lr_img = cv2.imread(args.source)
    hr_img = enhancer.enhance_image(lr_img)
    cv2.imshow("Result", hr_img)
    