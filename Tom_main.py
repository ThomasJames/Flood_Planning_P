import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj import Transformer
import shapely
from rtree.index import Index
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from pyproj import Geod
import geopandas as gpd
import rasterio
from rasterio import plot
import rtree
from rtree import index
import networkx as nx
import rasterio
from rasterio import mask
from rasterio.windows import Window
import pyproj
import numpy as np
import geopandas as gpd
import json
from rasterio.mask import mask
from shapely.wkt import loads
import scipy
from scipy import sparse
from numpy import asarray
from numpy import savetxt
from affine import Affine

# Function to take polygon, and coordinate
# Returns True if the point lies within the polygon
from sympy.utilities import pytest


def on_tile(c, b):
    try:
        if b.contains(c):
            return True
        else:
            return False
    except IOError:
        print("Unable to perform this operation")


# Function to join lists into coordinates
def generate_coordinates(p_x, p_y):
    return list(map(lambda x, y: (x, y), p_x, p_y))


if __name__ == "__main__":
    # Ask the user for their location
    print("Please input your location:")
    north, east = float(input("east: ")), float(input("north: "))
    print(north, east)

    # Create a buffer zone of 5km
    location = Point(east, north)

    # create a to spec bounding box "tile"
    tile = Polygon([(430000, 80000), (430000, 95000), (465000, 95000), (465000, 80000)])

    # Create an intersect polygon with the tile
    intersection_shape = location.buffer(5000).intersection(tile)

    # Assign the bounding box of the intersect shape to window
    window = intersection_shape.bounds
    print("window dimensions are: ", window)

    elevation = rasterio.open('elevation/SZ.asc')

    # Create a numpy array subset of the elevation data
    with rasterio.open('elevation/SZ.asc') as raster:
        # Upper Left pixel coordinate
        ul = raster.index(*window[0:2])
        # Lower right pixel coordinate
        lr = raster.index(*window[2:4])
        # Create the usable window dimensions
        window_pixel = ((lr[0], ul[0] + 1), (ul[1], lr[1] + 1))
        # Read the window to a np subset array
        elevation_window = raster.read(1, window=window_pixel)

        # Extract the x and y coordinates of the exterior
        intersection_pixel_x, intersection_pixel_y = raster.index(*intersection_shape.exterior.xy)
        # Generate the (x, y) coordinate format
        intersection_pixel_coords = generate_coordinates(intersection_pixel_x, intersection_pixel_y)
        # Generate a 'shapley' polygon
        intersection_pixel_polygon = Polygon(intersection_pixel_coords)

    # Rasterize the geometry
    mask = rasterio.features.rasterize(
        [(intersection_pixel_polygon, 0)],
        out_shape=elevation_window.shape,
        all_touched=True,
    )

    #  Create a numpy array of the buffer zone
    masked_elevation_data = np.ma.array(data=elevation_window, mask=mask.astype(bool))

    # Extract the coordinates of the highest point
    x, y = zip(np.where(masked_elevation_data == np.amax(masked_elevation_data)))
    x = (x[0])
    y = (y[0])
    x = x * 5
    y = y * 5
    x = x + window[1]
    y = y + window[0]

    print(x)
    print(y)

    print(np.amax(masked_elevation_data))
    print("elevation window shape is ", elevation_window.shape)
    print("The elevation window shape is: ", elevation_window.shape)
    print("Highest point on the window is", np.amax(elevation_window))
    print("Highest point in the buffer zone", np.amax(masked_elevation_data))
    print("The window bounds are: ", window)

    # # Some test coordinates
    # # (459619, 85800)
    # # (439619, 85800)
    # # (450000, 90000)

    plt.scatter(east, north, color="blue")
    plt.scatter(y, x, color="red")
    rasterio.plot.show(elevation, alpha=0.5)
    plt.show()
