#!/usr/bin/env python3

import rospy
import shapely
import math
import numpy as np
import threading
from ros_numpy import msgify
from autoware_mini.msg import Path, DetectedObjectArray, Waypoint
from sensor_msgs.msg import PointCloud2

from shapely.geometry import LineString, Polygon
from shapely import prepare

from lanelet2.projection import UtmProjector
from lanelet2.io import Origin, load
from autoware_mini.msg import TrafficLightResult, TrafficLightResultArray

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
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")
        self.braking_safety_distance_stopline = rospy.get_param("~braking_safety_distance_stopline")
        # Parameters related to lanelet2 map loading
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        lanelet2_map_path = rospy.get_param("~lanelet2_map_path")

        # variables
        self.detected_objects = None
        self.goal_waypoint = None
        self.traffic_lights = []

        # Lock for thread safety
        self.lock = threading.Lock()

        # Load the map using Lanelet2
        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise RuntimeError('Only "utm" is supported for lanelet2 map loading')
        lanelet2_map = load(lanelet2_map_path, projector)

        # Extract all stop lines and signals from the lanelet2 map
        all_stoplines = get_stoplines(lanelet2_map)
        self.trafficlights = get_stoplines_trafficlights(lanelet2_map)
        # If stopline_id is not in self.signals then it has no signals (traffic lights)
        self.tfl_stoplines = {k: v for k, v in all_stoplines.items() if k in self.trafficlights}

        # publishers
        self.local_path_collision_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Path, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('global_path', Path, self.extract_goal_waypoint, tcp_nodelay=True)

        rospy.Subscriber('/detection/traffic_light_status', TrafficLightResultArray, self.traffic_light_status_callback, queue_size=1, tcp_nodelay=True)

    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def path_callback(self, msg):
        with self.lock:
            detected_objects = self.detected_objects
        collision_points = np.array([], dtype=DTYPE)

        # If either there is no path or no obstacles
        if len(msg.waypoints) == 0 or msg.waypoints is None or (len(detected_objects) == 0 and self.goal_waypoint is None and len(self.traffic_lights) == 0):
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
        # Processing of goal waypoint as another collision point
        if self.goal_waypoint is not None:
            with self.lock:
                x = self.goal_waypoint.position.x
                y = self.goal_waypoint.position.y
                z = self.goal_waypoint.position.z
            
            # Goal point slightly buffered
            goal_polygon = Polygon([(x - 0.1, y - 0.1), (x, y - 0.1), (x, y), (x - 0.1, y)])

            if buf_linestring.intersects(goal_polygon):
                intersection_polygon = goal_polygon.intersection(buf_linestring)
                intersection_points = shapely.get_coordinates(intersection_polygon)

                for point_x, point_y in intersection_points:
                    collision_points = np.append(collision_points, np.array([(point_x, point_y, z, 0, 0, 0, self.braking_safety_distance_goal, np.inf, 1)], dtype=DTYPE))

        # Processing traffic lights as collision points
        if len(self.traffic_lights) > 0:
            with self.lock:
                traffic_lights = self.traffic_lights
            
            for traffic_light in traffic_lights:
                if traffic_light.recognition_result_str == "red":
                    if buf_linestring.intersects(traffic_light):
                        intersection_polygon = traffic_light.intersection(buf_linestring)
                        intersection_points = shapely.get_coordinates(intersection_polygon)

                        for x, y in intersection_points:
                            collision_points = np.append(collision_points, np.array([(x, y, 0, 0, 0, 0, self.braking_safety_distance_stopline, np.inf, 2)], dtype=DTYPE))

        #print(f"collision_points: {collision_points}")
        local_path_collision = msgify(PointCloud2, collision_points)
        local_path_collision.header = msg.header
        self.local_path_collision_pub.publish(local_path_collision)
    
    def extract_goal_waypoint(self, msg):
        if len(msg.waypoints) == 0:
            self.goal_waypoint = None
        else:
            with self.lock:
                goal = msg.waypoints[-1] # Last point on global path is the goal
                self.goal_waypoint = Waypoint()
                self.goal_waypoint.position.x = goal.position.x
                self.goal_waypoint.position.y = goal.position.y
                self.goal_waypoint.position.z = goal.position.z
    
    def traffic_light_status_callback(self, msg):
        if len(msg.results) == 0:
            self.traffic_lights = []
        else:
            with self.lock:
                for tfl_result in msg.results:
                    self.traffic_lights.append(self.tfl_stoplines[tfl_result.stopline_id])

    def run(self):
        rospy.spin()


def get_stoplines(lanelet2_map):
    """
    Add all stop lines to a dictionary with stop_line id as key and stop_line as value
    :param lanelet2_map: lanelet2 map
    :return: {stop_line_id: stopline, ...}
    """

    stoplines = {}
    for line in lanelet2_map.lineStringLayer:
        if line.attributes:
            if line.attributes["type"] == "stop_line":
                # add stoline to dictionary and convert it to shapely LineString
                stoplines[line.id] = LineString([(p.x, p.y) for p in line])

    return stoplines


def get_stoplines_trafficlights(lanelet2_map):
    """
    Iterate over all regulatory_elements with subtype traffic light and extract the stoplines and sinals.
    Organize the data into dictionary indexed by stopline id that contains a traffic_light id and the four coners of the traffic light.
    :param lanelet2_map: lanelet2 map
    :return: {stopline_id: {traffic_light_id: {'top_left': [x, y, z], 'top_right': [...], 'bottom_left': [...], 'bottom_right': [...]}, ...}, ...}
    """

    signals = {}

    for reg_el in lanelet2_map.regulatoryElementLayer:
        if reg_el.attributes["subtype"] == "traffic_light":
            # ref_line is the stop line and there is only 1 stopline per traffic light reg_el
            linkId = reg_el.parameters["ref_line"][0].id

            for tfl in reg_el.parameters["refers"]:
                tfl_height = float(tfl.attributes["height"])
                # plId represents the traffic light (pole), one stop line can be associated with multiple traffic lights
                plId = tfl.id

                traffic_light_data = {'top_left': [tfl[0].x, tfl[0].y, tfl[0].z + tfl_height],
                                      'top_right': [tfl[1].x, tfl[1].y, tfl[1].z + tfl_height],
                                      'bottom_left': [tfl[0].x, tfl[0].y, tfl[0].z],
                                      'bottom_right': [tfl[1].x, tfl[1].y, tfl[1].z]}

                # signals is a dictionary indexed by stopline id and contains dictionary of traffic lights indexed by pole id
                # which in turn contains a dictionary of traffic light corners
                signals.setdefault(linkId, {}).setdefault(plId, traffic_light_data)

    return signals

if __name__ == '__main__':
    rospy.init_node('collision_points_manager')
    node = CollisionPointsManager()
    node.run()