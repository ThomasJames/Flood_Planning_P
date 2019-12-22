import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj import Transformer
import shapely
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
import pyproj
import numpy as np
import geopandas as gpd
import json


# Function to take polygon, and coordinate
# Returns True if the point lies within the polygon

def mbr(coordinate, bound):
    try:
        if bound.contains(coordinate):
            return True
        else:
            return False
    except IOError:
        print("Unable to perform this operation")


""""" Creating a GUI for the user  """
import tkinter
from tkinter import *
# Aborted due to computer crashing: https://www.geeksforgeeks.org/python-gui-tkinter/


import matplotlib.pyplot as plt
from shapely.geometry import LineString

# All modules that can be used have been imported.

if __name__ == "__main__":
    # Create variables containing the relevant data.
    # elevation = rasterio.open("../material/elevation/SZ.asc")
    # background = rasterio.open("../material/background/raster-50k_2724246.tif")

    """""
    Task1: User Input
    The application should ask the user to input their current location as a British National Grid coordinate
    (easting and northing). Then, it should test whether the user is within a box (430000, 80000) and (465000, 95000).
    If the input coor- dinate is outside this box, inform the user and quit the application.
    This is done because the elevation raster provided to you extends only from (425000, 75000) to (470000, 100000)
    and the input point must be at least 5km from the edge of this raster.
    """""

    # Request coordinates from the user.
    print("Plese input the coordinate of the point you want to test:")
    north = float(input("x: "))
    east = float(input("y: "))

    # Print the coordinates for reference
    print("Coordinates are ", east, " east and ", north, "north")

    # Assign the coordinates to a shapely point
    coordinate = Point(east, north)
    print(coordinate)

    # create a minimum bounding box polygon with the specified coordinates
    tile = Polygon([(430000, 80000), (430000, 95000), (465000, 95000), (465000, 80000)])
    # Create the coordinates for the exterior
    x, y = tile.exterior.xy
    # Plot the bounding box
    plt.fill(x, y)
    plt.show()

    # Call the mbr method from the Bounding class
    if mbr(coordinate, tile) == True:
        print("Point is on tile")
    else:
        print("Please quit the application")

    """""
    Task 2: Highest Point Identification
    Identify the highest point within a 5km radius from the user location.
    To successfully complete this task you could (1) use the window function in rasterio to limit the size of your
    elevation array. If you do not use this window you may experience memory issues; or, (2) use a rasterised 5km buffer
    to clip an elevation array. Other solutions are also accepted. Moreover, if you are not capable to solve this task 
    you can select a random point within 5km of the user.
    """""
    # import and view the elevation data
    elevation = rasterio.open("elevation/SZ.asc")
    # rasterio.plot.show(elevation)

    # import and view the background data
    background = rasterio.open("background/raster-50k_2724246.tif")
    # rasterio.plot.show(background)

    # Import isle of wight data
    # isle_of_wight = gpd.read_file('shape/isle_of_wight.shp')
    # isle_of_wight.plot()

    # access the value of this raster as a NumPy array
    print(elevation.read(1))
    print(elevation.height)

    # To find the point of intersection
    # coordinate.buffer(1).intersection(elevation)

    # To find if a point lies inside a polygon
    # mbr.contains(coordinate)

    # Work out how much is 5km in coordinates the buffer around the coordinate     
    # coordinate_5km_bound = coordinate.buffer(5000)

    # argmax / argmin - for rasterio return the min/max coordinates
    # numy.amax function will find the maximum value.
    # numpy.amax(a, axis=None, out=None, keepdims=<no value>, initial=<no value>)
    # Aguments - a is the numpy array to find the maximum value,

    # Create a line between the point of highest elevation and the
    # path = LineString([(coordinate_5km_bound), (1, 1)])

    # Find the instance of the line between the point and the highest point of elevation
    # path.length

    # Links from Jake:
    # Basic Rasterio
    # https://rasterio.readthedocs.io/en/stable/quickstart.html
    # To get min and max values
    # https://thispointer.com/find-max-value-its-index-in-numpy-array-numpy-amax/ (page 11)

    """""  
    Task 3: Nearest Integrated Transport Network
    Identify the nearest Integrated Transport Network (ITN) node to the user and the nearest ITN node to the highest
     point identified in the previous step. To successfully complete this task you could use r-trees.     
    ITN is:
    OS built transport network built to store data about Road Network (road geometry), Road Routing Information
    (routing information for drivers concerning mandatory and banned turns and other restrictions) and Urban Paths
    (man-made path geometry in urban areas).
    """""

    """""    INDEXING THE VALUE      """""

    # Import the index module from rtree
    from rtree import index

    # Build the index module using its default constructor
    idx = index.Index()

    # Assign a boundary to insert into the bounding box
    index_boundary = (430000, 80000), (430000, 95000), (465000, 95000), (465000, 80000)

    # We can now insert an entry into the index:
    idx.insert(0, index_boundary, "A")

    # Create a for loop to iterate over the possible options of routes

    # To find the nearest point
    # If multiple points are of equal distance, they will all be returned.
    for i in idx.nearest((20000, 30000), 1):
        print(i)

    """""  
    Task 4: Shortest Path
    Identify the shortest route using Naismith’s rule from the ITN node nearest to the user and the ITN node nearest to
    the highest point. Naismith’s rule states that a reasonably fit person is capable of waking at Page 2 of 5
    5km/hr and that an additional minute is added for every 10 meters of climb (i.e., ascent not descent).
    To successfully complete this task you could calculate the weight iterating through each link segment. Moreover, if
    you are not capable to solve this task you could (1) approximate this algorithm by calculating the weight using
    only the start and end node elevation; (2) identify the shortest distance from the node nearest the user to the
    node nearest the highest point using only links in the ITN.
    To test the Naismith’s rule, you can use (439619, 85800) as a starting point.
    """""

    """""  
    Task 5: Map Plotting
    Plot a background map 10km x 10km of the surrounding area. You are free to use either a 1:50k Ordnance Survey raster
    (with internal color-map). Overlay a transparent elevation raster with a suitable color-map. Add the user’s starting
    point with a suitable marker, the highest point within a 5km buffer with a suitable marker, and the shortest route
    calculated with a suitable line. Also, you should add to your map, a color-bar showing the elevation range, a north
    arrow, a scale bar, and a legend.
    """""
    # See the network analysis of intergrated traffic network section

    """""  
    Task 6: Extend the Region
    The position of the user is restricted to a region in where the user must be more than 5km from the edge of the
    elevation raster. Write additional code to overcome this limitation.
    """""
