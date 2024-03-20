import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from shapely.geometry import Polygon, Point
import time

# from planning_utils import create_grid


def create_voxmap(data, voxel_size=5):
    """
    Returns a grid representation of a 3D configuration space
    based on given obstacle data.

    The `voxel_size` argument sets the resolution of the voxel map.
    """
    # minimum and maximum north coordinates
    north_min = np.floor(np.amin(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.amax(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.amin(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.amax(data[:, 1] + data[:, 4]))

    # maximum altitude
    alt_max = np.ceil(np.amax(data[:, 2] + data[:, 5]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min)) // voxel_size
    east_size = int(np.ceil(east_max - east_min)) // voxel_size
    alt_size = int(alt_max) // voxel_size

    # Create an empty grid
    voxmap = np.zeros((north_size, east_size, alt_size), dtype=np.bool)

    for i in range(data.shape[0]):
        # TODO: fill in the voxels that are part of an obstacle with `True`
        #
        # i.e. grid[0:5, 20:26, 2:7] = True
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        obstacle = [
            int(north - d_north - north_min) // voxel_size,
            int(north + d_north - north_min) // voxel_size,
            int(east - d_east - east_min) // voxel_size,
            int(east + d_east - east_min) // voxel_size,
        ]

        height = int(alt + d_alt) // voxel_size
        voxmap[obstacle[0]:obstacle[1], obstacle[2]:obstacle[3], 0:height] = True

    return voxmap


def draw_voxel_map(voxmap):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxmap, edgecolor='k')
    ax.set_xlim(voxmap.shape[0], 0)
    ax.set_ylim(0, voxmap.shape[1])
    # add a bit to z-axis height for visualization
    ax.set_zlim(0, voxmap.shape[2] + 20)

    plt.xlabel('North')
    plt.ylabel('East')

    plt.show()


def extract_polygons(data):
    polygons = []
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        # TODO: Extract the 4 corners of each obstacle
        #
        # NOTE: The order of the points needs to be counterclockwise
        # in order to work with the simple angle test
        # Also, `shapely` draws sequentially from point to point.
        #
        # If the area of the polygon in shapely is 0
        # you've likely got a weird order.
        obstacle = [north - d_north, north + d_north, east - d_east, east + d_east]
        corners = [(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]),
                   (obstacle[1], obstacle[2])]

        # TODO: Compute the height of the polygon
        height = alt + d_alt

        p = Polygon(corners)
        polygons.append((p, height))

    return polygons


def sample_points(data, zmin=0, zmax=10, num_samples=100):
    xmin = np.min(data[:, 0] - data[:, 3])
    xmax = np.max(data[:, 0] + data[:, 3])

    ymin = np.min(data[:, 1] - data[:, 4])
    ymax = np.max(data[:, 1] + data[:, 4])

    # print("X")
    # print("min = {0}, max = {1}\n".format(xmin, xmax))
    #
    # print("Y")
    # print("min = {0}, max = {1}\n".format(ymin, ymax))
    #
    # print("Z")
    # print("min = {0}, max = {1}".format(zmin, zmax))

    xvals = np.random.uniform(xmin, xmax, num_samples)
    yvals = np.random.uniform(ymin, ymax, num_samples)
    zvals = np.random.uniform(zmin, zmax, num_samples)

    samples = np.array(list(zip(xvals, yvals, zvals)))

    return samples


def collides(polygons, point):
    # TODO: Determine whether the point collides
    # with any obstacles.
    for (p, height) in polygons:
        if p.contains(Point(point)) and height >= point[2]:
            return True
    return False


def get_reachable_point(samples, polygons):
    t0 = time.time()
    reachable_points = []
    for point in samples:
        if not collides(polygons, point):
            reachable_points.append(point)
    time_taken = time.time() - t0
    print("Time taken {0} seconds ...".format(time_taken))

    return reachable_points


# def draw_points(data, zmax, points):
#     grid = create_grid(data, zmax, 1)
#
#     fig = plt.figure()
#
#     plt.imshow(grid, cmap='Greys', origin='lower')
#
#     nmin = np.min(data[:, 0])
#     emin = np.min(data[:, 1])
#
#     # draw points
#     all_pts = np.array(points)
#     north_vals = all_pts[:, 0]
#     east_vals = all_pts[:, 1]
#     plt.scatter(east_vals - emin, north_vals - nmin, c='red')
#
#     plt.ylabel('NORTH')
#     plt.xlabel('EAST')
#
#     plt.show()


# def draw_graph_with_points(data, graph, points, zmax):
#     grid = create_grid(data, zmax, 1)
#
#     fig = plt.figure()
#
#     plt.imshow(grid, cmap='Greys', origin='lower')
#
#     nmin = np.min(data[:, 0])
#     emin = np.min(data[:, 1])
#
#     # draw edges
#     for (n1, n2) in graph.edges:
#         plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'black', alpha=0.5)
#
#     # draw all nodes
#     for n1 in points:
#         plt.scatter(n1[1] - emin, n1[0] - nmin, c='blue')
#
#     # draw connected nodes
#     for n1 in graph.nodes:
#         plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')
#
#     plt.xlabel('NORTH')
#     plt.ylabel('EAST')
#
#     plt.show()