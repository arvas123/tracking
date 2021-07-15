# tracking
An implementation of tracking by detection for football videos
* Runs on CPU only. GPU will require different configs and imports (not included in repo)
* tf session not created yet
* use detection.json file for detections
* checkpoints contains tf weights file (saved_model.pb)
* yolo-weights contain yolo weights. saved_model is derived from yolov4-obj_best.weights
link to obj.zip https://drive.google.com/file/d/1DMh7mXasSpF_4tHDKvwJ-N1xovSBEnIr/view?usp=sharing  
link to test.zip https://drive.google.com/file/d/1rLuqmrMGsCUms8yTfdH68N-jHDAJwlab/view?usp=sharing  
link to yolov4.conv.137 https://drive.google.com/file/d/1a0bH26FdWLZLW7LW5gct1ETZYpGu4m-o/view?usp=sharing  
To run detect_yolo you will need to get darknet from https://github.com/AlexeyAB/darknet  