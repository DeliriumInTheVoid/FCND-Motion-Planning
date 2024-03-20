import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from planning_utils import a_star, heuristic, create_grid, read_local_position, prune_path, prune_path_bresenham, \
    create_skeleton, find_start_goal, heuristic_sqr, create_voronoi_grid_and_edges, closest_node_graph, a_star_graph, \
    heuristic_graph
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local

from skimage.util import invert
import matplotlib.pyplot as plt

import networkx as nx
import numpy.linalg as lin_alg

from global_path_planner import GlobalPathPlanner


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.80 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            distance = np.linalg.norm(self.target_position[0:2] - self.local_position[0:2])
            if distance < 2.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2],
                          self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING

        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        lat0, lon0 = read_local_position('colliders.csv')

        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 3.0)

        #  #################################
        # self.send_waypoints()
        # return
        #  #################################

        # TODO: retrieve current global position
        global_position = [self._longitude, self._latitude, 3.0]  # self._altitude

        # TODO: convert to current local position using global_to_local()
        current_local_position = global_to_local(global_position, self.global_home)

        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))

        # Define starting point on the grid (this is just grid center)
        # grid_start = (-north_offset, -east_offset)
        # TODO: convert start position to current position rather than map center
        grid_start = (int(current_local_position[0]) - north_offset, int(current_local_position[1]) - east_offset)

        # Set goal as some arbitrary position on the grid
        grid_goal = (-north_offset + 30, -east_offset - 30)
        # TODO: adapt to set goal as latitude / longitude position and convert
        # goal_global_position = [-122.398108, 37.793526, 3.0]
        goal_global_position = [-122.396082, 37.794156, TARGET_ALTITUDE]
        local_goal = global_to_local(goal_global_position, self.global_home)
        # grid_goal = (int(local_goal[0]) - north_offset, int(local_goal[1]) - east_offset)

        # Run A* to find a path from start to goal
        # path = self.grid_planing(grid, grid_start, grid_goal)
        # path = self.graph_medial_axis_planing(grid, grid_start, grid_goal)
        path = self.graph_voronoi_planing(data, TARGET_ALTITUDE, SAFETY_DISTANCE, grid_start, grid_goal)
        print('Path was found. Length: ', path)

        # TODO: prune path to minimize number of waypoints
        # path = prune_path(path)
        path, traced_paths = prune_path_bresenham(path, grid)
        print('Path was pruned. Length: ', path)

        # TODO (if you're feeling ambitious): Try a different approach altogether!

        # Convert path to waypoints
        waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0] for p in path]
        print('Waypoints were created: ', waypoints)
        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()


    def grid_planing(self, grid, grid_start, grid_goal):
        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        print('Local Start and Goal: ', grid_start, grid_goal)
        path, _ = a_star(grid, heuristic, grid_start, grid_goal)

        return path

    def graph_medial_axis_planing(self, grid, grid_start, grid_goal):
        skel, skel_distance = create_skeleton(grid)

        skel_start, skel_goal = find_start_goal(skel, grid_start, grid_goal)

        print(grid_start, grid_goal)
        print(skel_start, skel_goal)

        path, cost = a_star(invert(skel).astype(np.int), heuristic_sqr, tuple(skel_start), tuple(skel_goal))

        # plt.rcParams['figure.figsize'] = 12, 12
        # plt.imshow(grid, origin='lower')
        # plt.imshow(skel, cmap='Greys', origin='lower', alpha=0.7)
        #
        # plt.plot(grid_start[1], grid_start[0], 'gx')
        # plt.plot(grid_goal[1], grid_goal[0], 'gx')
        #
        # plt.plot(skel_start[1], skel_start[0], 'rx')
        # plt.plot(skel_goal[1], skel_goal[0], 'rx')
        #
        # pp2 = np.array(path)
        # plt.plot(pp2[:, 1], pp2[:, 0], 'r')
        #
        # plt.xlabel('EAST')
        # plt.ylabel('NORTH')
        # plt.show()

        return path

    def graph_voronoi_planing(self, grid_data, drone_altitude, safety_distance, grid_start, grid_goal):
        grid, edges, north_min, east_min = create_voronoi_grid_and_edges(grid_data, drone_altitude, safety_distance)
        print('Voronoi north offset = {0}, east offset = {1}'.format(north_min, east_min))
        print('Found %5d edges' % len(edges))

        graph = nx.Graph()
        for e in edges:
            p1 = e[0]
            p2 = e[1]
            dist = lin_alg.norm(np.array(p2) - np.array(p1))
            graph.add_edge(p1, p2, weight=dist)

        start_ne_g = closest_node_graph(graph, grid_start)
        goal_ne_g = closest_node_graph(graph, grid_goal)
        print(start_ne_g)
        print(goal_ne_g)

        path, cost = a_star_graph(graph, heuristic_graph, start_ne_g, goal_ne_g)
        print(len(path))

        # plt.rcParams['figure.figsize'] = 12, 12
        # plt.imshow(grid, origin='lower', cmap='Greys')
        #
        # for e in edges:
        #     p1 = e[0]
        #     p2 = e[1]
        #     plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-')
        #
        # plt.plot([grid_start[1], start_ne_g[1]], [grid_start[0], start_ne_g[0]], 'r-')
        # for i in range(len(path) - 1):
        #     p1 = path[i]
        #     p2 = path[i + 1]
        #     plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r-')
        # plt.plot([grid_goal[1], goal_ne_g[1]], [grid_goal[0], goal_ne_g[0]], 'r-')
        #
        # plt.plot(grid_start[1], grid_start[0], 'gx')
        # plt.plot(grid_goal[1], grid_goal[0], 'gx')
        #
        # plt.xlabel('EAST', fontsize=20)
        # plt.ylabel('NORTH', fontsize=20)
        # plt.show()

        return path


    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    path_planner = GlobalPathPlanner('colliders.csv', altitude=5, safety_distance=5)

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=600)
    drone = MotionPlanning(conn)

    drone.waypoints = path_planner.global_path

    time.sleep(1)
    drone.start()
