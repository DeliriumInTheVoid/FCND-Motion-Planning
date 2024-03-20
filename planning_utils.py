from enum import Enum
from queue import PriorityQueue
import numpy as np

from typing import Tuple

from bresenham import bresenham

from skimage.morphology import medial_axis
from skimage.util import invert

import numpy.linalg as lin_alg


def closest_node_graph(graph, current_point):
    """
    Compute the closest point in the `graph`
    to the `current_point`.
    """
    closest_point = None
    dist = 100000
    for p in graph.nodes:
        d = lin_alg.norm(np.array(p) - np.array(current_point))
        if d < dist:
            closest_point = p
            dist = d
    return closest_point


def heuristic_graph(n1, n2):
    return lin_alg.norm(np.array(n2) - np.array(n1))


def a_star_graph(graph, h, start, goal):
    """Modified A* to work with NetworkX graphs."""
    path = []
    path_cost = 0
    if start == goal:
        path.append(start)
        return path, path_cost
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                branch_cost = current_cost + cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost


def create_voronoi_grid_and_edges(data, drone_altitude, safety_distance):
    from scipy.spatial import Voronoi
    """
    Returns a grid representation of a 2D configuration space
    along with Voronoi graph edges given obstacle data and the
    drone's altitude.
    """
    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))
    # Initialize an empty list for Voronoi points
    points = []
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size - 1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size - 1)),
            ]
            grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1] = 1
            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])

    # TODO: create a voronoi graph based on
    # location of obstacle centres
    graph = Voronoi(points)

    # TODO: check each edge from graph.ridge_vertices for collision
    edges = []
    for v in graph.ridge_vertices:
        p1 = graph.vertices[v[0]]
        p2 = graph.vertices[v[1]]

        # pt1_int = (int(p1[0]), int(p1[1]))
        # pt2_int = (int(p2[0]), int(p2[1]))
        # print(f"bresenham for pt1: {pt1_int}, pt2: {pt2_int}")
        cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))

        hit = False

        for c in cells:
            # First check if we're off the map
            if np.amin(c) < 0 or c[0] >= grid.shape[0] or c[1] >= grid.shape[1]:
                hit = True
                break
            # Next check if we're in collision
            if grid[c[0], c[1]] == 1:
                hit = True
                break

        # If the edge does not hit on obstacle
        # add it to the list
        if not hit:
            # array to tuple for future graph creation step)
            p1 = (p1[0], p1[1])
            p2 = (p2[0], p2[1])
            edges.append((p1, p2))

    return grid, edges, int(north_min), int(east_min)


def create_skeleton(grid):
    """
    Create a skeleton of the given grid. The skeleton is a representation of the grid with a width of 1. The skeleton is
    created by using the medial axis transform of the grid.

    :param grid: The grid to create the skeleton from.
    :return: The skeleton of the grid.
    """
    # Create the medial axis of the grid.
    skel, distance = medial_axis(invert(grid), return_distance=True)

    # Distance to the background for pixels of the skeleton
    skel_distance = distance * skel

    return skel, skel_distance


def find_nearest_pt(skel, pt):
    # skel_cells = np.transpose(skel.nonzero())
    min_dist = np.linalg.norm(np.array(pt) - np.array(skel), axis=1).argmin()
    near_pt = skel[min_dist]
    return near_pt


def find_start_goal(skel, start, goal):
    skel_cells = np.transpose(skel.nonzero())
    near_start = find_nearest_pt(skel_cells, start)
    near_goal = find_nearest_pt(skel_cells, goal)
    return near_start, near_goal


def read_local_position(file_path) -> Tuple[float, float]:
    """
    Read local position from the given file path. The first line of the file is expected to be in the following format:
    lat0 37.792480, lon0 -122.397450

    :param file_path: The path to the file to read the local position from.
    :return: A tuple containing the local position in the format (lat0, lon0).
    """
    data = np.genfromtxt(file_path, delimiter=',', dtype='str', max_rows=1)
    print(data)

    lat0 = 0.0
    lon0 = 0.0
    for line in data:
        line_data = line.split()
        if len(line_data) < 2:
            continue
        if line_data[0] == 'lat0':
            lat0 = float(line_data[1])
        elif line_data[0] == 'lon0':
            lon0 = float(line_data[1])

    return lat0, lon0


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size - 1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size - 1)),
            ]
            grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1] = 1

    return grid, int(north_min), int(east_min)


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    NORTH_WEST = (-1, -1, np.sqrt(2))
    NORTH_EAST = (-1, 1, np.sqrt(2))
    SOUTH_WEST = (1, -1, np.sqrt(2))
    SOUTH_EAST = (1, 1, np.sqrt(2))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return self.value[0], self.value[1]


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    if (x - 1 < 0 or y - 1 < 0) or grid[x - 1, y - 1] == 1:
        valid_actions.remove(Action.NORTH_WEST)
    if (x - 1 < 0 or y + 1 > m) or grid[x - 1, y + 1] == 1:
        valid_actions.remove(Action.NORTH_EAST)
    if (x + 1 > n or y - 1 < 0) or grid[x + 1, y - 1] == 1:
        valid_actions.remove(Action.SOUTH_WEST)
    if (x + 1 > n or y + 1 > m) or grid[x + 1, y + 1] == 1:
        valid_actions.remove(Action.SOUTH_EAST)

    return valid_actions


def a_star(grid, h, start, goal):
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost


def heuristic_sqr(position, goal_position):
    return np.sqrt((position[0] - goal_position[0]) ** 2 + (position[1] - goal_position[1]) ** 2)


def heuristic(position, goal_position):
    # return np.abs(position[0] - goal_position[0]) + np.abs(position[1] - goal_position[1])
    return np.linalg.norm(np.array(position) - np.array(goal_position))
    # return np.sqrt((position[0] - goal_position[0])**2 + (position[1] - goal_position[1])**2)


import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MatplotlibPolygon

from shapely.geometry import Polygon


def prune_path_bresenham(path, grid, conservative=False):
    traced_paths = []
    if path is not None:
        pruned_path = path[:]  # Create a copy of the original path

        i = 0
        while i < len(pruned_path) - 2:
            p1 = pruned_path[i]
            p2 = pruned_path[i + 1]
            p3 = pruned_path[i + 2]

            cells = bresenham_conservative((int(p1[0]), int(p1[1])), (int(p3[0]), int(p3[1])), conservative)
            traced_paths.append(cells)
            # delta = 10
            # polygons = []
            # for cell in cells:
            #     pt_x, pt_y = cell
            #     obstacle = [pt_x - delta, pt_x + delta, pt_y - delta, pt_y + delta]
            #     polygons.append(
            #         Polygon([(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]),
            #                  (obstacle[1], obstacle[2])]))
            #
            # plt.rcParams['figure.figsize'] = 12, 12
            # fig, ax = plt.subplots()
            # ax.imshow(grid, cmap='Greys', origin='lower')
            #
            # delta = 6
            # for cell in cells:
            #     pt_x, pt_y = cell
            #     # pt_x += north_min
            #     # pt_y += east_min
            #     x = np.array([pt_x - delta, pt_x + delta, pt_x + delta, pt_x - delta, pt_x - delta]) + 12
            #     y = np.array([pt_y - delta, pt_y - delta, pt_y + delta, pt_y + delta, pt_y - delta]) + 12
            #     ax.add_patch(MatplotlibPolygon(xy=list(zip(y, x)), edgecolor='red', facecolor='none'))
            #
            # for p in polygons:
            #     x, y = p.exterior.xy
            #     ax.add_patch(MatplotlibPolygon(xy=list(zip(y, x)), edgecolor='blue', facecolor='none'))
            #
            # Set axis limits
            # ax.set_xlim(0, 1000)
            # ax.set_ylim(0, 1000)
            #
            # # Set axis labels (optional)
            # ax.set_xlabel('X-axis')
            # ax.set_ylabel('Y-axis')
            #
            # plt.show()

            if path_has_obstacle(grid, cells):
                i += 1
            else:
                pruned_path.remove(p2)
    else:
        pruned_path = path

    return pruned_path, traced_paths


def path_has_obstacle(grid, path):
    for p in path:
        if grid[p[0], p[1]] == 1:
            return True
    return False


def bresenham_conservative(p1, p2, conservative=False):
    x0, y0 = p1
    x1, y1 = p2
    cells = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    # err = dx - dy

    d = 0
    while x0 != x1 or y0 != y1:
        cells = cells + [[x0, y0]]
        if d < dx - dy:
            d += dy
            x0 += sx
        elif d == dx - dy:
            if conservative:
                cells.append([x0 + sx, y0])
                cells.append([x0, y0 + sy])
            d += dy
            x0 += sx
            d -= dx
            y0 += sy
        else:
            d -= dx
            y0 += sy

    return np.array(cells)


# based on the solution of the collinearity lesson
def prune_path(path):
    if path is not None:
        pruned_path = path[:]  # Create a copy of the original path

        i = 0
        while i < len(pruned_path) - 2:
            p1 = point(pruned_path[i])
            p2 = point(pruned_path[i + 1])
            p3 = point(pruned_path[i + 2])

            if collinearity_check(p1, p2, p3):
                pruned_path.remove(pruned_path[i + 1])
            else:
                i += 1
    else:
        pruned_path = path

    return pruned_path


def point(p):
    return np.array([p[0], p[1], 1.]).reshape(1, -1)


# should be faster than collinearity_check()
def collinearity_int(p1, p2, p3):
    collinear = False
    # TODO: Calculate the determinant of the matrix using integer arithmetic
    # TODO: Set collinear to True if the determinant is equal to zero
    det = p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
    det = np.abs(det)
    print(f"det 2d: ${det}")
    # collineal = True if det < 0.002 else False
    epsilon = 1e-2
    # Compare the absolute value of the determinant with epsilon
    if np.abs(det) < epsilon:
        collinear = True
    return collinear


def collinearity_check(p1, p2, p3, epsilon=1e-6):
    m = np.concatenate((p1, p2, p3), axis=0)
    det = np.linalg.det(m)
    return np.isclose(det, 0, atol=epsilon)
