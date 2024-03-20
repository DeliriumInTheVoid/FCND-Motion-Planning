import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import numpy.linalg as lin_alg
from udacidrone import global_to_local

from planning_utils import create_voronoi_grid_and_edges, closest_node_graph, a_star_graph, heuristic_graph,\
    prune_path_bresenham

from planning_constants import PlanningConst as PC


class GlobalPathPlanner:
    def __init__(self, colliders_file):
        self._local_pt_start = [0.0, 0.0, 0.0]
        self._local_pt_end = [0.0, 0.0, 0.0]
        self._east_min = 0
        self._north_min = 0
        self._edges = []
        self._grid = None
        self._altitude = 5
        self._safety_distance = 5
        self._global_path = []
        self._grid_data = np.loadtxt(colliders_file, delimiter=',', dtype='Float64', skiprows=2)

    @property
    def altitude(self):
        return self._altitude

    @property
    def global_path(self):
        return self._global_path

    @property
    def local_pt_start(self):
        return self._local_pt_start

    @property
    def local_pt_end(self):
        return self._local_pt_end

    @property
    def north_min(self):
        return self._north_min

    @property
    def east_min(self):
        return self._east_min

    def create_global_path(self, global_start_pt, global_end_pt, global_home, altitude=5, safety_distance=5):
        self._safety_distance = safety_distance
        self._altitude = altitude

        self._grid, self._edges, self._north_min, self._east_min = (
            create_voronoi_grid_and_edges(self._grid_data, self._altitude, self._safety_distance))

        current_local_position = global_to_local(global_start_pt, global_home)
        self._local_pt_start = (int(current_local_position[0]) - self._north_min, int(current_local_position[1]) - self._east_min)

        local_goal = global_to_local(global_end_pt, global_home)

        self._local_pt_end = (int(local_goal[0]) - self._north_min, int(local_goal[1]) - self._east_min)

        self._graph_voronoi_planing(self._local_pt_start, self._local_pt_end)

    def _graph_voronoi_planing(self, grid_start, grid_goal):
        print('Found %5d edges' % len(self._edges))

        # Create graph from edges
        graph = nx.Graph()
        for e in self._edges:
            p1 = e[0]
            p2 = e[1]
            dist = lin_alg.norm(np.array(p2) - np.array(p1))
            graph.add_edge(p1, p2, weight=dist)

        # Find nearest nodes to start and goal points
        start_ne_g = closest_node_graph(graph, grid_start)
        goal_ne_g = closest_node_graph(graph, grid_goal)
        print(start_ne_g)
        print(goal_ne_g)

        # Use `a_star_graph` function to find the shortest path between start and goal nodes
        full_path, cost = a_star_graph(graph, heuristic_graph, start_ne_g, goal_ne_g)
        print(f"Found path length: {len(full_path)}")

        # Prune path using `prune_path_bresenham` function which uses conservative bresenham algorithm to remove
        # unnecessary waypoints.
        path, traced_paths = prune_path_bresenham(full_path, self._grid, True)
        path, _ = prune_path_bresenham(path, self._grid)
        print('Path was pruned. Length: ', len(path))

        self._global_path = [[int(p[0] + self._north_min), int(p[1] + self._east_min), self._altitude, 0] for p in path]

        if grid_goal != path[-1]:
            self._global_path.append([grid_goal[0] + self._north_min, grid_goal[1] + self._east_min, self._altitude, 0])

        print(f"Global path: {self._global_path}")

        if PC.DRAW_PATH_PRUNE_TRACING:
            self.draw_prune_path_tracing(full_path, traced_paths, grid_start, grid_goal)
        if PC.DRAW_GLOBAL_PATH:
            self.draw_global_path(full_path, grid_start, grid_goal, start_ne_g)

    def draw_prune_path_tracing(self, full_path, traced_paths, grid_start, grid_goal):
        plt.rcParams['figure.figsize'] = 12, 12

        for path in traced_paths:
            plt.imshow(self._grid, origin='lower', cmap='Greys')
            for e in self._edges:
                p1 = e[0]
                p2 = e[1]
                plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'g-')

            for i in range(len(full_path) - 1):
                p1 = full_path[i]
                p2 = full_path[i + 1]
                plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-')

            for pt in path:
                plt.scatter(pt[1], pt[0], c='red')

            plt.plot(grid_start[1], grid_start[0], 'gx')
            plt.plot(grid_goal[1], grid_goal[0], 'gx')

            plt.xlabel('EAST', fontsize=20)
            plt.ylabel('NORTH', fontsize=20)
            plt.show()

    def draw_global_path(self, full_path, grid_start, grid_goal, start_ne_g):

        plt.rcParams['figure.figsize'] = 12, 12
        plt.imshow(self._grid, origin='lower', cmap='Greys')

        for e in self._edges:
            p1 = e[0]
            p2 = e[1]
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'g-')

        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-')

        plt.plot([grid_start[1], start_ne_g[1]], [grid_start[0], start_ne_g[0]], 'r-')
        for i in range(len(self._global_path) - 1):
            p1 = self._global_path[i]
            p2 = self._global_path[i + 1]
            plt.plot([p1[1] - self._east_min, p2[1] - self._east_min], [p1[0] - self._north_min, p2[0] - self._north_min], 'r-', linewidth=2)

        plt.plot(grid_start[1], grid_start[0], 'gx')
        plt.plot(grid_goal[1], grid_goal[0], 'gx')

        plt.xlabel('EAST', fontsize=20)
        plt.ylabel('NORTH', fontsize=20)
        plt.show()

    @property
    def grid_data(self):
        return self._grid_data
