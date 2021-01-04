# -*- coding: utf-8 -*-
# @Time : 2020/9/10 下午4:38 
# @Author : midaskong 
# @File : face_align.py 
# @Description:

import cv2
import os
import numpy as np

REFERENCE_FACIAL_POINTS = [        # default reference facial points for crop_size = (112, 112); should adjust REFERENCE_FACIAL_POINTS accordingly for other crop_size
    [30.29459953+8,  51.69630051],
    [65.53179932+8,  51.50139999],
    [48.02519989+8,  71.73660278],
    [33.54930115+8,  92.3655014],
    [62.72990036+8,  92.20410156]
]

#bg = np.zeros([112,112,3])
bg = cv2.imread("/home/hwits/Documents/FaceRec/face_rec/facebase/crop/YaoMing.jpg")
for b in REFERENCE_FACIAL_POINTS:
    cv2.circle(bg, (int(b[0]), int(b[1])), 2, (255, 255, 255), 4)

cv2.imshow("bg",bg)
cv2.waitKey(0)
