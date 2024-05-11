import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
from scipy.ndimage import rotate
import time
import torch
import numpy as np


class FPS:
    def __init__(self, avg=10) -> None:
        self.accum_time = 0
        self.counts = 0
        self.avg = avg

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self):
        self.synchronize()
        self.prev_time = time.time()

    def stop(self, debug=True):
        self.synchronize()
        self.accum_time += time.time() - self.prev_time
        self.counts += 1
        if self.counts == self.avg:
            self.fps = round(self.counts / self.accum_time)
            if debug: print(f"FPS: {self.fps}")
            self.counts = 0
            self.accum_time = 0


def correct_skew(image, delta=0.5, limit=5):
    def determine_score(arr, angle):
        data = rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0,0),  sigmaX=33, sigmaY=33)
    divide = cv2.divide(gray, blur, scale=255)
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    skewness_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)
    skewness_thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)

    return skewness_img, skewness_thresh

def resize_img(image, target=300):
    target_width = target
    target_height = target 

    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height

    if target_width / aspect_ratio <= target_height:
        target_height = int(target_width / aspect_ratio)
    else:
        target_width = int(target_height * aspect_ratio)
    
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    return resized_image