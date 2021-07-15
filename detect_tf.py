import tensorflow as tf
from tensorflow.python import saved_model
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
def gen_bbox(image_path, input_size=416, weight_path='./checkpoints', iou=0.45, score=0.45):
    #load model
    #Note: session is not created
    # to create session this will have to be moved to track.py
    model = tf.saved_model.load(weight_path, tags=[tag_constants.SERVING])
    
    #load image
    original_image = cv2.imread(image_path)
    h,w,_ = original_image.shape
    image = cv2.resize(original_image, (input_size,input_size))
    image = image/255
    image_data = np.asarray([image]).astype(np.float32)
    #run yolov4 on image
    infer = model.signatures['serving_default']
    batch_data = tf.constant(image_data)
    bbox = infer(batch_data)
    for key, val in bbox.items():
        boxes = val[:,:,0:4]
        pred_conf = val[:, :, 4:]
    #filter out some bboxes
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores = tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
        ),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score
    )
    valid_detections = valid_detections.numpy()[0]
    boxes = boxes.numpy()[0][:valid_detections]
    scores = scores.numpy()[0][:valid_detections]
    detections = []
    for i, box in enumerate(boxes):
        detections.append([int(box[1]*w), int(box[0]*h), int(box[3]*w), int(box[2]*h), scores[i]])
    return detections

    
if __name__=='__main__':
    gen_bbox('0186.jpg')






