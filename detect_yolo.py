from sys import stderr, stdout
import os 
import cv2 
import numpy as np
from subprocess import *
import re
from tracker import x_to_bbox, bbox_to_z
def detect(frame):
    #don't use this
    if not os.getcwd().endswith('darknet'):
        os.chdir('darknet')
    img_path = frame
    out = Popen(f'./darknet detector test ../obj.data ../yolov4-obj.cfg ../yolo-weights/yolov4-obj_best.weights -ext_output {img_path} -dont_show -thresh 0.3', shell =True, stdout=PIPE, stderr=PIPE)
    stdout, stderr=(out.communicate())
    stdout = stdout.decode('utf-8')
    bboxes = []
    for line in stdout.split('\n'):
        if line.startswith('player'):
            conf, bbox = line.split('\t')
            score = re.search("[0-9]+",conf)
            score=int(score.group())
            matches = re.findall("[0-9]+", bbox)
            coords = (list(map(int,matches)))
            bbox = [coords[0], coords[1], coords[0]+coords[2], coords[1]+coords[3], score/100]
            bboxes.append(bbox)
    return bboxes
if __name__=='__main__':
    bboxes = (detect('../0186.jpg'))
    img = cv2.imread('../DataSet_001/0186.jpg')
    for d in bboxes:
        d = bbox_to_z(d)
        d = list(map(int,x_to_bbox(d)[0]))
        cv2.rectangle(img, (d[0],d[1]),(d[2],d[3]), (0,255,0), 1)
    cv2.imshow('img', img)
    cv2.waitKey(0)