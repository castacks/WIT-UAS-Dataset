#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology
# Credit, extended from: W. Nicholas Greene https://gist.github.com/wngreene/835cda68ddd9c5416defce876a4d7dd9

"""Extract images from a rosbag.
"""

import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()

    print(f"Extract images from {args.bag_file} on topic {args.image_topic} into {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    start_time = bag.get_start_time()

    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        elapsed_time = t.to_sec() - start_time
        filename = f"time_{elapsed_time:07.2f}.png"

        cv2.imwrite(os.path.join(args.output_dir, filename), cv_img)
        print(f"Wrote {filename}")

        count += 1

    bag.close()

    return

if __name__ == '__main__':
    main()

