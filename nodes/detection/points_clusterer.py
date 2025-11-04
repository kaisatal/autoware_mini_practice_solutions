#!/usr/bin/env python3

import rospy

from sensor_msgs.msg import PointCloud2
from ros_numpy import numpify, msgify
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured

class PointsClusterer:
    def __init__(self):
        
        # Parameters

        # Publishers

        # Subscribers
        rospy.Subscriber('points_filtered', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        pass

    def points_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        print(f"points shape: {points.shape}")
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('points_clusterer')
    node = PointsClusterer()
    node.run()