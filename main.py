import argparse
import cv2
import numpy as np
import warnings

from plate_recognition import PlateRecognition
from super_resolution import SuperResolution
from utils import FPS

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, 
                        default='webcam', 
                        help ='Path to image/webcam.')
    parser.add_argument('--model', type=str, 
                        default='./model/best.onnx', 
                        help ='Path to model.')
    parser.add_argument('--threshold', type=float,
                        default=0.55, 
                        help ='Threshold for plate detection model.')
    parser.add_argument('--cuda', 
                        action='store_true', 
                        help ='Use GPU or not.')
    args = parser.parse_args()
    
    enhancer = SuperResolution()
    anpr = PlateRecognition(args.model, enhancer, args.cuda)
    fps = FPS()
    
    if args.source == "webcam":
        print("Webcam")
        # Get a reference to webcam #0 (the default one)
        process_this_frame = True
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        if not video_capture.isOpened():
            print("Failed to open webcam.")
            exit()

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Only process every other frame of video to save time
            if process_this_frame:
                # Resize frame of video to 1/4 size for faster recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                # rgb_small_frame = small_frame[:, :, ::-1]
                # Start count fps
                fps.start()

            process_this_frame = not process_this_frame

            # Preprocess results
            frame, license_num = anpr.anpr(frame, args.threshold)
            print(f"license: {license_num}")
            # Stop count fps
            fps.stop()
            # Display the resulting image
            cv2.imshow('Real-time License Plate Recognition', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()
    else:
        img = cv2.imread(args.source)
        output, license_num = anpr.anpr(img, args.threshold)
        print(f"license: {license_num}")
        
        cv2.imshow("Result", output)
        cv2.waitKey(0) & 0xFF == ord('q')
        
    

