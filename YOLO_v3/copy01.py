import cv2
import matplotlib.pyplot as plt
from utils import *
from darknet import Darknet

cfg_file = 'cfg\yolov3.cfg'
weight_file = 'weights\yolov3.weights'
namesfile = 'data\coco.names'
m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)
m.print_network()

nms_thresh = 0.6
iou_thresh = 0.4

clicked=False
def onMouse(event,x,y,flags,param):
    global clicked
    if event==cv2.EVENT_FLAG_LBUTTON:
        clicked=True
video_capture = cv2.VideoCapture(0)
cv2.namedWindow("MYwindow")
cv2.setMouseCallback("MYwindow",onMouse)
success,frame=video_capture.read()
while success and cv2.waitKey(1)==-1 and not clicked :
    frame = cv2.resize(frame, (m.width, m.height))
    iou_thresh = 0.4
    nms_thresh = 0.6
    boxes = detect_objects(m, frame, iou_thresh, nms_thresh)
    print_objects(boxes, class_names)
    plot_boxes(frame, boxes, class_names, plot_labels = True,winname="MYwindow")
    success,frame=video_capture.read()
cv2.destroyAllWindows()
video_capture.release()
