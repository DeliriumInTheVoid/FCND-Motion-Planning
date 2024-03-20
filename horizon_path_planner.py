import numpy as np
import matplotlib.pyplot as plt

from matplotlib.transforms import Affine2D
from shapely.geometry import Polygon, Point
from planning_utils import closest_node_graph, a_star_graph, heuristic_graph, create_grid
from matplotlib.patches import Polygon as MatplotlibPolygon
from probabilistic import create_graph, prune_path_3d
from sampling import filter_samples

import multiprocessing
import threading

from planning_constants import PlanningConst as PC


class HorizonPathPlanner:
    def __init__(self, global_path, grid_data, altitude, safety_distance):
        self._global_path = global_path  # just for visualization
        self._grid_data = grid_data
        self._grid, self._north_offset, self._east_offset = create_grid(grid_data, altitude, safety_distance)

        self._polygons = self.extract_polygons(grid_data)

        self._result_queue = multiprocessing.Queue()
        self._local_path_thread = None

        self._pre_planed_local_paths = []

    @property
    def pre_planed_local_paths(self):
        return self._pre_planed_local_paths.copy()

    def create_local_path_async(self, start, goal, horizon_size, horizon_random_samples,
                                horizon_height, connect_nearest_samples):

        self._result_queue = multiprocessing.Queue()
        self._local_path_thread = threading.Thread(
            target=self.create_local_path,
            args=(start, goal, horizon_size, horizon_random_samples, horizon_height, connect_nearest_samples)
        )
        self._local_path_thread.start()

        return self._local_path_thread

    def create_local_path(self, start, goal, horizon_size=200, horizon_random_samples=100,
                          horizon_height=40, connect_nearest_samples=10):
        start_pt = start[:2]
        goat_pt = goal[:2]
        horizon_frame, samples = self.create_horizon_frame_with_samples(start_pt, goat_pt, horizon_size,
                                                                        horizon_height=horizon_height,
                                                                        samples_num=horizon_random_samples)

        horizon_base_poly = Polygon(horizon_frame)

        print(f"Polygons length: {len(self._polygons)}")

        # Filter polygons that are in the horizon
        horizon_polygons = []
        for p, h in self._polygons:
            if p.overlaps(horizon_base_poly) or horizon_base_poly.contains(p):
                horizon_polygons.append((p, h))

        print(f"Horizon polygons length: {len(horizon_polygons)}")
        print(f"Horizon polygons:{horizon_polygons[:10]}")

        # Filter samples that can be reached
        samples = filter_samples(samples, horizon_polygons)
        print(f"Samples length: {len(samples)}")

        graph = create_graph(samples, connect_nearest_samples, horizon_polygons)

        nearest_start = closest_node_graph(graph, start[0:3])
        nearest_goal = closest_node_graph(graph, goal[0:3])

        full_path, cost = a_star_graph(graph, heuristic_graph, nearest_start, nearest_goal)
        # if start_pt != full_path[0]:
        #     full_path.insert(0, start[0:3])
        if goat_pt != full_path[-1] and horizon_base_poly.contains(Point(goal[:3])):
            full_path.append(tuple(goal[:3]))

        path = prune_path_3d(full_path, horizon_polygons)

        if PC.DRAW_HORIZON_STEPS:
            plt.rcParams['figure.figsize'] = 12, 12
            fig, ax = plt.subplots()
            ax.imshow(self._grid, cmap='Greys', origin='lower')

            if self._global_path:
                for i in range(len(self._global_path) - 1):
                    p1 = self._global_path[i]
                    p2 = self._global_path[i + 1]
                    ax.plot([p1[1] - self._east_offset, p2[1] - self._east_offset],
                            [p1[0] - self._north_offset, p2[0] - self._north_offset], 'r-', linewidth=4, alpha=0.6)

            for poly, h in self._polygons:
                x, y = poly.exterior.xy
                x -= np.array([self._north_offset])
                y -= np.array([self._east_offset])
                ax.add_patch(MatplotlibPolygon(xy=list(zip(y, x)), edgecolor='green', facecolor='none'))

            for poly, h in horizon_polygons:
                x, y = poly.exterior.xy
                x -= np.array([self._north_offset])
                y -= np.array([self._east_offset])
                ax.add_patch(MatplotlibPolygon(xy=list(zip(y, x)), edgecolor='red', facecolor='none'))

            hp_x, hp_y = horizon_base_poly.exterior.xy
            hp_x -= np.array([self._north_offset])
            hp_y -= np.array([self._east_offset])
            ax.add_patch(MatplotlibPolygon(xy=list(zip(hp_y, hp_x)), edgecolor='red', facecolor='none'))

            samples_pts = np.array(samples)
            north_vals = samples_pts[:, 0]
            east_vals = samples_pts[:, 1]
            ax.scatter(east_vals - self._east_offset, north_vals - self._north_offset, c='blue', s=10)

            # ########################## DRAW GRAPH ###########################
            # draw edges
            for (n1, n2) in graph.edges:
                plt.plot([n1[1] - self._east_offset, n2[1] - self._east_offset],
                         [n1[0] - self._north_offset, n2[0] - self._north_offset], 'black',
                         alpha=0.4)
            # draw all nodes
            # for n1 in nodes:
            #     plt.scatter(n1[1] - emin, n1[0] - nmin, c='blue')

            # draw connected nodes
            for n1 in graph.nodes:
                plt.scatter(n1[1] - self._east_offset, n1[0] - self._north_offset, c='red')

            # draw full graph path
            path_pairs = zip(full_path[:-1], full_path[1:])
            for (n1, n2) in path_pairs:
                plt.plot([n1[1] - self._east_offset, n2[1] - self._east_offset],
                         [n1[0] - self._north_offset, n2[0] - self._north_offset], 'green',
                         linewidth=3)

            # draw pruned path
            path_pairs = zip(path[:-1], path[1:])
            for (n1, n2) in path_pairs:
                plt.plot([n1[1] - self._east_offset, n2[1] - self._east_offset],
                         [n1[0] - self._north_offset, n2[0] - self._north_offset], 'blue',
                         linewidth=3)
            # #################################################################

            ax.set_xlim(0, 1000)
            ax.set_ylim(0, 1000)

            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')

            plt.show()

        if PC.PRE_PLANING:
            self._pre_planed_local_paths.append(path.copy())

        return path, horizon_base_poly

    def create_horizon_frame_with_samples(self, start_pt, end_pt, frame_size, horizon_height=40, samples_num=100):
        vector = np.array(end_pt) - np.array(start_pt)
        normalized_vector = vector / np.linalg.norm(vector)
        half_size = frame_size / 2

        frame_points = np.array([
            [start_pt[0] - half_size, start_pt[1] - half_size],
            [start_pt[0] + half_size, start_pt[1] - half_size],
            [start_pt[0] + half_size, start_pt[1] + half_size],
            [start_pt[0] - half_size, start_pt[1] + half_size]
        ])

        rotation_angle = np.arccos(normalized_vector[1])
        angle_deg = np.degrees(rotation_angle)

        min_x, min_y = np.min(frame_points, axis=0)
        max_x, max_y = np.max(frame_points, axis=0)

        samples = np.random.uniform(low=[min_x, min_y], high=[max_x, max_y], size=(samples_num, 2))

        transform = Affine2D().rotate_deg_around(start_pt[0], start_pt[1], np.degrees(rotation_angle))

        frame_points = transform.transform(frame_points)

        samples = transform.transform(samples)
        samples = [(pt_x, pt_y, np.random.uniform(0, horizon_height)) for pt_x, pt_y in samples]

        return frame_points, samples

    def extract_polygons(self, data):
        polygons = []
        for i in range(data.shape[0]):
            north, east, alt, d_north, d_east, d_alt = data[i, :]

            obstacle = [north - d_north, north + d_north, east - d_east, east + d_east]
            corners = [(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]),
                       (obstacle[1], obstacle[2])]

            height = alt + d_alt

            p = Polygon(corners)
            polygons.append((p, height))

        return polygons
