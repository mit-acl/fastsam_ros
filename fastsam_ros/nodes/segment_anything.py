#!/usr/bin/env python3

import sys
import pathlib

import rospy
import numpy as np
import cv2 as cv
import cv_bridge

import sensor_msgs.msg as sensor_msgs
import vision_msgs.msg as vision_msgs

from fastsam_ros.wrapper import FastSAM
# from yolov7_ros.visualization import Visualizer
# from yolov7_ros.utils import create_detection_msg

class Segmenter:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        
        # ROS Parameters
        self._weights = pathlib.Path(rospy.get_param("~weights", 'weights/FastSAM-x.pt'))
        if not self._weights.is_absolute():
            self._weights = pathlib.Path(__file__).absolute().parent.parent / self._weights
        if not self._weights.exists():
            rospy.logerr(f"Weights '{self._weights.as_posix()}' do not exist!")
            rospy.signal_shutdown(f"Weights '{self._weights.as_posix()}' do not exist!")
            sys.exit(1)

        self._imgsz = rospy.get_param("~image_size", 1024)
        self._conf_thr = rospy.get_param("~confidence_threshold", 0.4)
        self._iou_thr = rospy.get_param("~iou_threshold", 0.9)
        self._device = rospy.get_param("~device", "cuda")

        # Load Yolo model
        self.segmenter = FastSAM(self._weights.as_posix(),
                conf_thresh=self._conf_thr, iou_thresh=self._iou_thr,
                img_size=self._imgsz, device=self._device)
        # self.viz = Visualizer(self.segmenter)

        # # Setup class database
        # for cls, name in enumerate(self.segmenter.model.names):
        #     rospy.set_param(f"~classes/{cls}", name)

        # ROS communication
        self.sub_img_in = rospy.Subscriber('image_raw', sensor_msgs.Image, self.img_cb)
        self.pub_img_out = rospy.Publisher('image_dets', sensor_msgs.Image, queue_size=10)
        self.pub_dets = rospy.Publisher('detections', vision_msgs.Detection2DArray, queue_size=10)

    def img_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        detections = self.segmenter.detect(frame)

        # detmsg = create_detection_msg(msg.header, detections)
        # detmsg.header = msg.header
        # self.pub_dets.publish(detmsg)

        # if self.pub_img_out.get_num_connections() > 0:
        #     self.viz.draw_2d_bboxes(frame, detections)
        #     out = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        #     out.header = msg.header
        #     self.pub_img_out.publish(out)


if __name__ == "__main__":
    rospy.init_node("FastSAM")

    segmenter = Segmenter()
    rospy.spin()
