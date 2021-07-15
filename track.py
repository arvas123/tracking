from tracker import *
import cv2
import os
import matplotlib.pyplot as plt

def with_detections():
    '''
    dont use this - it needs to reload the detector everytime which is a huge timewaster 
    i have attached the tensorflow weights file, but havent figured out how to create a session so i can run it on a video stream
    '''
    from detect_tf import gen_bbox
    total_time = 0.0
    num_frames = 0
    dir = 'DataSet_001'
    frame_list = sorted(os.listdir(dir))[0:100]
    mot_tracker = Sort(max_age=10)
    fr=0
    for frame in frame_list:
        if frame.endswith('txt'): continue
        fr+=1
        path = f'{dir}/{frame}'
        detections = np.array(gen_bbox(path))
        num_frames+=1
        img = cv2.imread(f'{dir}/{frame}')
        trackers = mot_tracker.update(img, dets=detections)
        for d in trackers:
            d=d.astype(np.int32)
            cv2.rectangle(img, (d[0],d[1]),(d[2],d[3]), (0,255,0), 1)
            cv2.putText(img, f'{d[4]}',
                        (d[0], d[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,0), 2)
        cv2.imshow('img', img)
        ch = cv2.waitKey(10)
        if chr(ch&256)=='c': break
def without_detection():
    '''
    with detections preloaded
    the test set i have is not continous. the first 300 or so frames are
    it quits when a frame is missing
    use ./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output -dont_show -out result.json < data/train.txt
    in colab with the filenames in train.txt
    that json file will directly run here
    I've attached my detections file
    '''
    import json
    detections = json.load(open('detection.json'))
    mot_tracker = Sort(max_age=10)
    frame=0
    for i in range(len(detections)):
        
        frame = detections[i]
        fname = frame['filename']
        img = cv2.imread(fname)
        try:
            h,w,_ = img.shape
        except Exception:
            print(fname)
            break
        dets  = frame['objects']
        l_dets = []
        for det in dets:
            vals = det['relative_coordinates']
            c_x, c_y, o_h, o_w = vals['center_x'], vals['center_y'], vals['height'], vals['width']
            c_x*=w
            o_w*=w
            c_y*=h
            o_h*=h
            x1,x2,y1,y2 = c_x - o_w/2, c_x + o_w/2, c_y - o_h/2, c_y + o_h/2
            conf = det['confidence']
            if conf>0.0:
                l_dets.append([x1,y1,x2,y2])
        trackers = mot_tracker.update(img, dets=np.array(l_dets))
        for d in trackers:
            d=d.astype(np.int32)
            cv2.rectangle(img, (d[0],d[1]),(d[2],d[3]), (0,255,0), 1)
            cv2.putText(img, f'{d[4]}',
                    (d[0], d[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,0), 2)
        frame+=1
        cv2.imshow(f'frame {frame}', img)
        ch = cv2.waitKey(1)
        if chr(ch&256)=='c':
            break
if __name__=='__main__':
    (without_detection())
    
        





