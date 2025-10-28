#!/usr/bin/env python3

import rospy

# All these imports from lanelet2 library should be sufficient
import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest

from geometry_msgs.msg import PoseStamped

from autoware_mini.msg import Path
from autoware_mini.msg import Waypoint

def load_lanelet2_map(lanelet2_map_path):
    """
    Load a lanelet2 map from a file and return it
    :param lanelet2_map_path: name of the lanelet2 map file
    :param coordinate_transformer: coordinate transformer
    :param use_custom_origin: use custom origin
    :param utm_origin_lat: utm origin latitude
    :param utm_origin_lon: utm origin longitude
    :return: lanelet2 map
    """

    # get parameters
    coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
    use_custom_origin = rospy.get_param("/localization/use_custom_origin")
    utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
    utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

    # Load the map using Lanelet2
    if coordinate_transformer == "utm":
        projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
    else:
        raise ValueError('Unknown coordinate_transformer for loading the Lanelet2 map ("utm" should be used): ' + coordinate_transformer)

    lanelet2_map = load(lanelet2_map_path, projector)

    return lanelet2_map


class GlobalPlanner:
    def __init__(self):

        # Parameters
        self.lanelet2_map = load_lanelet2_map(rospy.get_param("~lanelet2_map_path"))
        # traffic rules
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, 
                                                           lanelet2.traffic_rules.Participants.VehicleTaxi)
        # routing graph
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)
        self.current_location = None
        self.goal_point = None
        self.speed_limit = rospy.get_param("~speed_limit")
        self.output_frame = rospy.get_param("~output_frame")

        rospy.loginfo(f"Speed limit: {self.speed_limit} km/h")

        # Publishers
        self.waypoints_pub = rospy.Publisher('planning/global_path', Path, queue_size=10, latch=True)

        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def current_pose_callback(self, msg):
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        # get start and end lanelets
        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]

        if self.goal_point is None: # Not added yet
            return
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.goal_point, 1)[0][1]

        # find routing graph
        route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, True)

        if route is None:
            rospy.logwarn("No route found for goal point!")
            return None

        # find shortest path
        path = route.shortestPath()
        # This returns LaneletSequence to a point where a lane change would be necessary to continue
        path_no_lane_change = path.getRemainingLane(start_lanelet)
        print(f"path_no_lane_change: {path_no_lane_change}")

    def goal_callback(self, msg):
        # loginfo message about receiving the goal point
        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(), 
                      msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 
                      msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, 
                      msg.pose.orientation.w, msg.header.frame_id)
        
        self.goal_point = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
    

    def convert_lanelet_sequence_to_waypoints(self, lanelet_sequence):
        for lanelet in lanelet_sequence:
            # code to check if lanelet has attribute speed_ref
            if 'speed_ref' in lanelet.attributes:
                speed_km_h = float(lanelet.attributes['speed_ref'])
                speed_m_s = speed_km_h / 3.6
            else:
                speed_m_s = self.speed_limit / 3.6
            
            waypoints = []
            
            for point in lanelet.centerline:
                # create Waypoint (from autoware_mini.msgs import Waypoint) and get the coordinates from lanelet.centerline points
                waypoint = Waypoint()
                waypoint.position.x = point.x
                waypoint.position.y = point.y
                waypoint.position.z = point.z
                waypoint.speed = speed_m_s

                waypoints.append(waypoint)

            path = Path()        
            path.header.frame_id = self.output_frame
            path.header.stamp = rospy.Time.now()
            path.waypoints = waypoints
            self.waypoints_pub.publish(path)
        
    
    def run(self):
        rospy.spin()
        
if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner')
    node = GlobalPlanner()
    node.run()