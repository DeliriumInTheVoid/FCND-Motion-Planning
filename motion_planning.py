import argparse
import time
from enum import Enum

import numpy as np

import msgpack

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection, WebSocketConnection  # noqa: F401
from udacidrone.messaging import MsgID

from global_path_planner import GlobalPathPlanner
from horizon_path_planner import HorizonPathPlanner

from shapely.geometry import Point
from planning_utils import read_local_position

from planning_constants import PlanningConst as PC


class Phases(Enum):
    EMPTY = -1
    MANUAL = 0
    ARMING = 1
    TAKEOFF = 2
    PLANNING = 3
    WAYPOINT = 4
    LANDING = 5
    DISARMING = 6


class Phase:
    def __init__(self, phase: Phases, flying_drone):
        self._phase = phase
        self._next_phase: Phase = self._get_initial_next_phase()
        self._drone = flying_drone

    def next_phase(self):
        return self._next_phase

    def start_phase(self):
        pass

    def update_local_position(self) -> bool:
        return False

    def update_velocity(self) -> bool:
        return False

    def update_state(self) -> bool:
        return False

    def _get_initial_next_phase(self):
        return EmptyPhase()


class EmptyPhase(Phase):
    def __init__(self):
        super().__init__(Phases.EMPTY, None)

    def _get_initial_next_phase(self):
        return None


class ManualPhase(Phase):
    def __init__(self, flying_drone: Drone):
        super().__init__(Phases.MANUAL, flying_drone)

    def update_state(self) -> bool:
        print("arming transition")
        # if not self._drone.armed:
        #     self._drone.arm()
        #     self._drone.take_control()
        #     self._drone.set_home_position(-122.396082, 37.794156, 3.0)
        # return False
        self._drone.arm()
        self._drone.take_control()

        self._next_phase = PathPlanningPhase(self._drone)
        return True


class ArmingPhase(Phase):
    def __init__(self, flying_drone: Drone):
        super().__init__(Phases.ARMING, flying_drone)

    def update_state(self) -> bool:
        if self._drone.armed:
            print("takeoff transition")
            target_altitude = 5.0
            self._drone.target_position[2] = target_altitude
            self._drone.takeoff(target_altitude)
            self._next_phase = TakeoffPhase(self._drone)
            return True
        return False


class TakeoffPhase(Phase):
    def __init__(self, flying_drone: Drone):
        super().__init__(Phases.TAKEOFF, flying_drone)

    def update_local_position(self) -> bool:
        altitude = -1.0 * self._drone.local_position[2]
        if altitude > 0.95 * self._drone.target_position[2]:
            self._next_phase = WaypointPhase(self._drone)
            return True
        return False


class DisarmingPhase(Phase):
    def __init__(self, flying_drone: Drone):
        super().__init__(Phases.DISARMING, flying_drone)

    def update_state(self) -> bool:
        if not self._drone.armed:
            print("manual transition")
            self._drone.release_control()
            self._drone.stop()
            self._drone.in_mission = False
            self._next_phase = ManualPhase(self._drone)
            return True
        return False


class LandingPhase(Phase):
    def __init__(self, flying_drone: Drone):
        super().__init__(Phases.LANDING, flying_drone)

    def update_local_position(self) -> bool:
        if ((self._drone.global_position[2] - self._drone.global_home[2] < 0.1) and
                abs(self._drone.local_position[2]) < 0.01):
            self._disarm()
            return True
        return False

    def _disarm(self):
        print("disarm transition")
        self._drone.disarm()
        self._next_phase = DisarmingPhase(self._drone)


class PathPlanningPhase(Phase):
    def __init__(self, flying_drone: Drone):
        super().__init__(Phases.PLANNING, flying_drone)

    def update_state(self):
        if not self._drone.armed:
            return False

        self._drone.plan_global_path()
        self._drone.send_waypoints()

        # self._drone.set_home_position(self._drone.global_position[0],
        #                               self._drone.global_position[1],
        #                               self._drone.global_position[2])

        self._next_phase = ArmingPhase(self._drone)
        return True


class WaypointPhase(Phase):
    def __init__(self, flying_drone: Drone):
        super().__init__(Phases.WAYPOINT, flying_drone)
        self._pre_planed_local_paths = []
        self._global_waypoint = None
        self._global_waypoints = self._drone.global_path_planner.global_path
        self._local_waypoints = []
        self._local_waypoint = None
        self._horizon_poly = None

        self._horizon_planner_thread = None

    def start_phase(self):
        if self._global_waypoints:
            self._global_waypoint = self._global_waypoints.pop(0)

            if not PC.PRE_PLANING:
                if PC.USE_HORIZON_PLANNER:
                    local_position = self._drone.local_position.copy()
                    local_position[2] *= -1
                    self._local_waypoint = [local_position[0], local_position[1], local_position[2], 0]

                    self._horizon_planner_thread = (
                        self._drone.local_path_planner.create_local_path_async(local_position, self._global_waypoint))
                else:
                    self._send_to_pt(self._global_waypoint)
            elif PC.USE_HORIZON_PLANNER:
                self._pre_planed_local_paths = self._drone.local_path_planner.pre_planed_local_paths
                self._local_waypoints = self._pre_planed_local_paths.pop(0)

                self._local_waypoint = self._local_waypoints.pop(0)
                self._send_to_pt(self._local_waypoint)

    def update_local_position(self) -> bool:
        if self._horizon_planner_thread is not None:
            if self._horizon_planner_thread.is_alive():
                return False
            else:
                self._horizon_planner_thread.join()
                self._local_waypoints, self._horizon_poly = self._horizon_planner_thread.result_queue.get()
                self._horizon_planner_thread = None

                self._local_waypoint = self._local_waypoints.pop(0)
                self._send_to_pt(self._local_waypoint)

        velocity = self._drone.local_velocity.copy()
        speed = np.linalg.norm(velocity)

        local_pos = self._drone.local_position.copy()
        local_pos[2] *= -1

        waypoint = self._local_waypoint if PC.USE_HORIZON_PLANNER else self._global_waypoint

        dist = np.linalg.norm(local_pos[0:2] - waypoint[0:2])  # self._local_waypoint[0:2]

        dist_threshold = 0.2 if len(self._global_waypoints) == 0 and len(self._local_waypoints) == 0 else 2

        if dist < dist_threshold:  # and speed < 0.1
            if PC.USE_HORIZON_PLANNER and self._local_waypoints:
                self._local_waypoint = self._local_waypoints.pop(0)
                self._send_to_pt(self._local_waypoint)
                return False
            elif PC.PRE_PLANING and PC.USE_HORIZON_PLANNER:
                if self._pre_planed_local_paths:
                    self._local_waypoints = self._pre_planed_local_paths.pop(0)
                else:
                    return self._lend()
            elif self._global_waypoints:
                if PC.USE_HORIZON_PLANNER:
                    if self._horizon_poly.contains(Point(self._global_waypoint[:3])):
                        self._global_waypoint = self._global_waypoints.pop(0)

                    self._horizon_planner_thread = self._drone.local_path_planner.create_local_path_async(local_pos,
                                                                                                          self._global_waypoint)

                    # self._local_waypoints, self._horizon_poly = self._drone.local_path_planner.create_local_path(local_pos, self._global_waypoint)

                    # if len(self._local_waypoints) == 0 and len(self._global_waypoints) > 0:
                    #     print("Error! Can't reach goal!")
                    #     # TODO: REPLANING
                else:
                    self._global_waypoint = self._global_waypoints.pop(0)
                    self._drone.cmd_position(self._global_waypoint[0], self._global_waypoint[1],
                                             self._global_waypoint[2], 0)

                return False
            else:
                return self._lend()

        return False

    def _lend(self):
        self._drone.land()
        self._next_phase = LandingPhase(self._drone)
        return True

    def _send_to_pt(self, pt):
        self._drone.cmd_position(pt[0], pt[1], pt[2], 0)


class BackyardFlyer(Drone):

    def __init__(self, connection, global_path_planner: GlobalPathPlanner, local_path_planner: HorizonPathPlanner):
        super().__init__(connection)
        self._target_position = np.array([0.0, 0.0, 0.0])
        self._local_waypoints = []
        self._in_mission = True
        self.check_state = {}

        self._global_path_planner = global_path_planner
        self._local_path_planner = local_path_planner

        # initial state
        self.current_phase: Phase = EmptyPhase()

        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    @property
    def global_path_planner(self):
        return self._global_path_planner

    @property
    def local_path_planner(self):
        return self._local_path_planner

    @property
    def target_position(self) -> np.ndarray:
        return self._target_position

    @property
    def in_mission(self) -> bool:
        return self._in_mission

    @in_mission.setter
    def in_mission(self, value: bool):
        self._in_mission = value

    def plan_global_path(self):
        if PC.PRE_PLANING:
            return

        lat, lon = read_local_position(PC.COLLIDERS_FILE)
        start_position = [lon, lat, PC.ALTITUDE]
        self.set_home_position(lon, lat, PC.ALTITUDE)
        self._global_path_planner.create_global_path(start_position, PC.GOAL_GLOBAL_POSITION, self.global_home,
                                                     PC.ALTITUDE, PC.SAFETY_DISTANCE)

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self._global_path_planner.global_path)
        self.connection._master.write(data)

    def local_position_callback(self):
        """
        This triggers when `MsgID.LOCAL_POSITION` is received and self.local_position contains new data
        """
        if self.current_phase.update_local_position():
            self._start_next_phase()

    def velocity_callback(self):
        """
        This triggers when `MsgID.LOCAL_VELOCITY` is received and self.local_velocity contains new data
        """
        if self.current_phase.update_velocity():
            self._start_next_phase()

    def state_callback(self):
        """
        This triggers when `MsgID.STATE` is received and self.armed and self.guided contain new data
        """
        if not self._in_mission:
            return

        if self.current_phase.update_state():
            self._start_next_phase()

    def start(self):
        """This method is provided
        
        1. Open a log file
        2. Start the drone connection
        3. Close the log file
        """
        self.current_phase = ManualPhase(self)

        print("Creating log file")
        self.start_log("Logs", "NavLog.txt")
        print("starting connection")
        self.connection.start()
        print("Closing log file")
        self.stop_log()

    def _start_next_phase(self):
        self.current_phase = self.current_phase.next_phase()
        self.current_phase.start_phase()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    lat0, lon0 = read_local_position(PC.COLLIDERS_FILE)
    # lat0, lon0, alt0 = 37.797760, -122.394280, 10.0

    global_path_planner = GlobalPathPlanner(PC.COLLIDERS_FILE)

    if PC.PRE_PLANING:
        # unfortunately there is no option to set initial position for the drone in the simulator, so it's just
        # hardcoded with value which drone has on start of the simulation
        global_home = [-122.39745, 37.79248, 0.]
        start_global_position = [lon0, lat0, PC.ALTITUDE]
        global_path_planner.create_global_path(start_global_position, PC.GOAL_GLOBAL_POSITION, global_home,
                                               PC.ALTITUDE, PC.SAFETY_DISTANCE)

    horizon_path_planner = HorizonPathPlanner(global_path_planner.global_path, global_path_planner.grid_data,
                                              PC.ALTITUDE, PC.SAFETY_DISTANCE)

    if PC.PRE_PLANING and PC.USE_HORIZON_PLANNER:
        global_waypoints = global_path_planner.global_path.copy()
        local_position = global_waypoints.pop(0)
        global_waypoint = global_waypoints.pop(0)

        local_waypoints, horizon_poly = horizon_path_planner.create_local_path(
            local_position, global_waypoint, PC.HORIZON_SIZE, PC.HORIZON_RANDOM_SAMPLES,
            PC.HORIZON_HEIGHT, PC.CONNECT_NEAREST_SAMPLES)

        local_waypoint = None

        dist = 1

        while dist > 0.2:
            if local_waypoints:
                local_waypoint = local_waypoints.pop(0)
            else:
                while global_waypoints and (horizon_poly.contains(Point(global_waypoint[:3]))
                                            or local_waypoint[:2] == global_waypoint[:2]):
                    global_waypoint = global_waypoints.pop(0)

                local_waypoints, horizon_poly = horizon_path_planner.create_local_path(
                    local_waypoint, global_waypoint, PC.HORIZON_SIZE, PC.HORIZON_RANDOM_SAMPLES,
                    PC.HORIZON_HEIGHT, PC.CONNECT_NEAREST_SAMPLES)

            dist = np.linalg.norm(np.array([global_path_planner.local_pt_end[:2]]) -
                                  np.array([
                                      local_waypoint[0] - global_path_planner.north_min,
                                      local_waypoint[1] - global_path_planner.east_min
                                  ]))

    if PC.RUN_DRONE:
        conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), threaded=False, PX4=False, timeout=600)
        # conn = WebSocketConnection('ws://{0}:{1}'.format(args.host, args.port))
        drone = BackyardFlyer(conn, global_path_planner, horizon_path_planner)

        time.sleep(2)
        drone.start()
