import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2

loadMD = load_model("mobileV2-maskcheckx.h5")

IMGsize = 128
faceNet = cv2.dnn.readNet('deploy.txt','res10_300x300_ssd_iter_140000.caffemodel')
cap = cv2.VideoCapture(0)

#############################
while True:
    ret, img = cap.read()
    img = cv2.resize(img , (800,450))
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:


            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            startX = startX-30
            startY = startY -30
            endX = endX+30
            endY = endY+30

            y2 = startY - 60

            roi = img[startY:endY, startX:endX]
            imgPixel = cv2.resize(roi, (IMGsize, IMGsize))
            imr = np.array(imgPixel)
            imr = imr.astype('float32')
            imr /= 255
            imr = np.reshape(imr, (1, 128, 128, 3))
            r = loadMD.predict(imr)
            ir = np.argmax(r)
            if ir == 1:
                result = 'mask'
                c = (0,255,0)
            elif ir == 0:
                result = 'no-mask'
                c = (0,0,255)
            cv2.rectangle(img, (startX, startY), (endX, endY), c, 3)
            cv2.rectangle(img, (startX, startY - 40), (endX, startY), c, -1)
            cv2.putText(img, str(result), (startX + 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255,), 1)


    cv2.imshow("image", img)

    k = cv2.waitKey(10)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()