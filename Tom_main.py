
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


import numpy as np


def zoom(a, factor):
    a = np.asarray(a)
    slices = [slice(0, old, 1 / factor) for old in a.shape]
    idxs = (np.mgrid[slices]).astype('i')
    return a[tuple(idxs)]


""""" Creating a GUI for the user  """
import tkinter
from tkinter import *
# Aborted due to computer crashing: https://www.geeksforgeeks.org/python-gui-tkinter/


import matplotlib.pyplot as plt
from shapely.geometry import LineString

# All modules that can be used have been imported.


if __name__ == "__main__":

    """""   First Step is to import that data, in this format """""
    # Create variables containing the relevant data.
    # elevation = rasterio.open("../material/elevation/SZ.asc")
    # background = rasterio.open("../material/background/raster-50k_2724246.tif")

    # import and view the elevation data
    elevation = rasterio.open("elevation/SZ.asc", "r")
    background = rasterio.open("background/raster-50k_2724246.tif", "r")

    # Read elevation data as an array
    elevation_array = elevation.read(1)

    """""
    Task1: User Input
    The application should ask the user to input their current location as a British National Grid coordinate
    (easting and northing). Then, it should test whether the user is within a box (430000, 80000) and (465000, 95000).
    If the input coor- dinate is outside this box, inform the user and quit the application.
    This is done because the elevation raster provided to you extends only from (425000, 75000) to (470000, 100000)
    and the input point must be at least 5km from the edge of this raster.
                                                                                                               
    Task 2: Highest Point Identification                                                                                
    Identify the highest point within a 5km radius from the user location.                                              
    To successfully complete this task you could (1) use the window function in rasterio to limit the size of your      
    elevation array. If you do not use this window you may experience memory issues; or, (2) use a rasterised 5km buffer
    to clip an elevation array. Other solutions are also accepted. Moreover, if you are not capable to solve this task  
    you can select a random point within 5km of the user. 
    """""

    # Request coordinates from the user.
    print("Please input your location:")
    north, east = float(input("east: ")), float(input("north: "))

    # Assign the coordinates to a shapely point
    user_location = Point(east, north)
    print(user_location)

    # Create a 5km buffer around the point
    user_location_5km_buffer = user_location.buffer(5000)
    x_c, y_c = user_location_5km_buffer.exterior.xy

    # create a minimum bounding box polygon with the specified coordinates
    tile = Polygon([(430000, 80000), (430000, 95000), (465000, 95000), (465000, 80000)])
    # Create the coordinates for the exterior
    x_t, y_t = tile.exterior.xy

    # Some test coordinates
    # (459619, 85800)
    # (439619, 85800)
    # (450000, 90000) 

    intersect = user_location_5km_buffer.intersection(tile)
    intersect_coords = np.array(intersect.exterior)
    x_i, y_i = intersect.exterior.xy

    # State whether the coordinate lies on the tile
    if on_tile(user_location_5km_buffer, tile):
        print("Point is on tile")
    else:
        print("You are out of range, please quit the application")

    # If the buffer zone is within the tile, points outside the intersection zone are excluded.
    if on_tile(user_location_5km_buffer, tile):
        # mask the elevation area outside the buffer zone
        masked_elevation_array, transformed = rasterio.mask.mask(elevation, [intersect], crop=False)
        # Rescale the coordinates (5 Pixels for every m)
        rescaled_masked_elevation_array = np.kron(masked_elevation_array, np.ones((5, 5)))

        # Access the highest point
        highest_in_5km = np.amax(rescaled_masked_elevation_array)

        # Extract the coordinates of the highest point
        x_h, y_h = zip(*np.where(rescaled_masked_elevation_array == highest_in_5km))

    # Adjust the coordinates
    e_h = ((x_h[0]) + 75000)
    n_h = ((y_h[0]) + 425000)

    # Extract one of the sets of coordinates
    e_h = e_h[1]
    n_h = n_h[1]

    # Assign shapley point
    safe_zone = Point(e_h, n_h)

    # Calculate the distance between the user and the safe zone.
    distance_from_saftey = (user_location.distance(safe_zone)) / 1000

    print(e_h)
    print(n_h)

    # Prints details of the results
    print("The highest point within 5km from you is at ", n_h, "E,",
          e_h, "E", "at a height of", highest_in_5km, "meters, at a distance of", distance_from_saftey)

    """""
    Task 5: Map Plotting
    Plot a background map 10km x 10km of the surrounding area. You are free to use either a 1:50k Ordnance Survey raster
    (with internal color-map). Overlay a transparent elevation raster with a suitable color-map. Add the user’s starting
    point with a suitable marker, the highest point within a 5km buffer with a suitable marker, and the shortest route
    calculated with a suitable line. Also, you should add to your map, a color-bar showing the elevation range, a north
    arrow, a scale bar, and a legend.
    """""

    # Plot to show the user location, with a 5km radius, overlayed onto the elevation data

    plt.scatter(east, north, color="black", alpha=1)  # Specific coordinate
    plt.plot(x_t, y_t, color="wheat", alpha=1)  # Tile
    plt.fill(x_c, y_c, color="skyblue", alpha=0.1)  # 50km Buffer at 40% opacity
    plt.scatter(n_h, e_h, color="red", marker="x")
    plt.axis('equal')  # Ensures consistent scale
    plt.fill(x_i, y_i, color="tan", alpha=0.4)  # Intersection Zone
    plt.arrow(427500, 102000, 0, 1000, head_width=400, color="black")  # North Arrow
    plt.text(427100, 101100, "N", fontsize=7)  # North Arrow text
    plt.arrow(427500, 72000, 10000, 0, head_width=0, color="black")  # Scale bar
    plt.text(430200, 71250, "5 km", fontsize=5)  # Scale bar text
    plt.ylabel("Northings")
    plt.xlabel("Eastings")
    plt.title("Elevation map")
    rasterio.plot.show(elevation, alpha=1)  # Plot the elevation data
    plt.show()

    """""  
    Task 3: Nearest Integrated Transport Network
    Identify the nearest Integrated Transport Network (ITN) node to the user and the nearest ITN node to the highest
     point identified in the previous step. To successfully complete this task you could use r-trees.     
    ITN is:
    OS built transport network built to store data about Road Network (road geometry), Road Routing Information
    (routing information for drivers concerning mandatory and banned turns and other restrictions) and Urban Paths
    (man-made path geometry in urban areas).
    """""

    # Import the ITN file
    solent_itn_json = "itn/solent_itn.json"
    with open(solent_itn_json, "r") as f:
        solent_itn_json = json.load(f)

    # Users stating location - user_location
    # Target location - safe_zone

    # Plotting the ITN roadlinks
    # g = nx.Graph()
    # road_links = solent_itn_json['roadlinks']
    # for link in road_links:
    #     g.add_edge(road_links[link]['start'], road_links[link]['end'], fid=link, weight=road_links[link]['length'])
    # nx.draw(g, node_size=0.1, edge_size=0.0)
    # plt.show()

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
    Task 6: Extend the Region
    The position of the user is restricted to a region in where the user must be more than 5km from the edge of the
    elevation raster. Write additional code to overcome this limitation.
    """""
