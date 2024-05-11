import cv2
import numpy as np
import argparse
import onnxruntime as ort
from collections import OrderedDict, namedtuple

from utils import correct_skew, resize_img
import easyocr

class PlateRecognition():
    def __init__(self, model_path, enhancer, cuda=False):
        self.model_path = model_path
        self.cuda = cuda
        
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        print("Onnx runtime running with plate detector model...")
        
        self.reader = easyocr.Reader(['en'], gpu=cuda, quantize=True)
        self.enhancer = enhancer
    
    
    def extract_text(self, img):
        text = self.reader.readtext(img, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        return text
    
    
    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)
    
    
    def plate_detector(self, img):
        names = ['license']
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255
        im.shape

        outname = [i.name for i in self.session.get_outputs()]
        outname

        inname = [i.name for i in self.session.get_inputs()]
        inname

        inp = {inname[0]:im}

        outputs = self.session.run(outname, inp)[0]
        return outputs, ratio, dwdh
    
    
    def plat_recognition(self, img, box):
        x, y, w, h = box[0], box[1], (box[2] - box[0]), (box[3] - box[1])

        crop_img = img[y:y + h, x:x + w]
        hr_img = self.enhancer.enhance_image(crop_img)

        if hr_img.shape[0] > 400 or hr_img.shape[1] > 400:
            hr_img = resize_img(hr_img)
        skewness, thresh_skew = correct_skew(hr_img)

        inv = 255 - thresh_skew
        kernel = np.ones((2, 2), np.uint8)
        dilate = cv2.dilate(inv, kernel)

        # try:
        result = ""
        text = self.extract_text(dilate)
        if text:
            # print(text)
            if text[0][2] > 0.3:
                for s in text:
                    result += s[1]
                return result
        
        text = self.extract_text(skewness)
        if text:
            # print(text)
            for s in text:
                result += s[1]
            return result

        result = "not detected"
        return result
    
    
    def anpr(self, img, threshold):
        # plate detector
        outputs, ratio, dwdh = self.plate_detector(img)

        result = [img.copy()]
        license_num = ""

        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
            image = result[int(batch_id)]
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()

            if score >= threshold:
                # plate recognition
                license_num = self.plat_recognition(image, box)

                color = [0, 255, 0]
                if license_num == "not detected":
                    color = [0, 0, 255]

                license_num = license_num.strip()

                cv2.rectangle(image, box[:2], box[2:], color, 2)
                cv2.putText(image, license_num, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness=2)

        # result = cv2.cvtColor(result[0], cv2.COLOR_BGR2RGB)
        return result[0], license_num