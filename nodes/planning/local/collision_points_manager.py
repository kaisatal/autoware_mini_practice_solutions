#!/usr/bin/env python3

import rospy
import shapely
import math
import numpy as np
import threading
from ros_numpy import msgify
from autoware_mini.msg import Path, DetectedObjectArray
from sensor_msgs.msg import PointCloud2

from shapely.geometry import LineString, Polygon
from shapely import prepare

DTYPE = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('vx', np.float32),
    ('vy', np.float32),
    ('vz', np.float32),
    ('distance_to_stop', np.float32),
    ('deceleration_limit', np.float32),
    ('category', np.int32)
])

class CollisionPointsManager:

    def __init__(self):

        # parameters
        self.safety_box_width = rospy.get_param("safety_box_width")
        self.stopped_speed_limit = rospy.get_param("stopped_speed_limit")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")

        # variables
        self.detected_objects = None

        # Lock for thread safety
        self.lock = threading.Lock()

        # publishers
        self.local_path_collision_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Path, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def path_callback(self, msg):
        with self.lock:
            detected_objects = self.detected_objects
        collision_points = np.array([], dtype=DTYPE)

        if len(msg.waypoints) == 0 or len(detected_objects) == 0:
            local_path_collision = PointCloud2()
            local_path_collision.header = msg.header
            self.local_path_collision_pub.publish(local_path_collision)
            return

        path_linestring = LineString([(w.position.x, w.position.y) for w in msg.waypoints])
        # Adding width to the local path
        buf_linestring = path_linestring.buffer(self.safety_box_width / 2, cap_style="flat")
        prepare(buf_linestring)
        
        for obj in detected_objects:
            coords = np.array(obj.convex_hull).reshape(-1, 3) # -1 means that the length is not known
            convex_hull = [(x, y) for x, y, _ in coords] # Polygon expects tuples
            obj_polygon = Polygon(convex_hull)

            if buf_linestring.intersects(obj_polygon): # Only objects on the path are important
                intersection_polygon = obj_polygon.intersection(buf_linestring)
                intersection_points = shapely.get_coordinates(intersection_polygon)

                object_speed = np.hypot(obj.velocity.x, obj.velocity.y)
                for x, y in intersection_points:
                    collision_points = np.append(collision_points, np.array([(x, y, obj.centroid.z, obj.velocity.x, obj.velocity.y, obj.velocity.z,
                                                                          self.braking_safety_distance_obstacle, np.inf, 3 if object_speed < self.stopped_speed_limit else 4)], dtype=DTYPE))
        
        print(f"collision_points: {collision_points}")
        local_path_collision = msgify(PointCloud2, collision_points)
        local_path_collision.header = msg.header
        self.local_path_collision_pub.publish(local_path_collision)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('collision_points_manager')
    node = CollisionPointsManager()
    node.run()