from facial_emotion_recognition import EmotionRecognition
from cv2 import cv2 as cv

er = EmotionRecognition(device='cpu')

img = cv.imread('3.jpg')

img = er.recognise_emotion(img,return_type='BGR')

cv.imshow("Image",img)

while True:
    key = cv.waitKey(10)
    if key & 0xff == 27:
        break
