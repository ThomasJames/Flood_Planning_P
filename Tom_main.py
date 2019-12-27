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
from numpy import asarray
from numpy import savetxt

# Function to test if any object is within a polygon
def on_tile(c, b):
    try:
        if b.contains(c):
            return True
        else:
            return False
    except IOError:
        print("Unable to perform this operation")


# Function to join lists into (x, y) coordinates
def generate_coordinates(p_x, p_y):
    try:
        return list(map(lambda x, y: (x, y), p_x, p_y))
    except IOError:
        print("Unable to perform this operation")

"""""
Extreme flooding is expected on the Isle of Wight and the authority in charge of planning the emergency response is
advising everyone to proceed by foot to the nearest high ground.
To support this process, the emergency response authority wants you to develop a software to quickly advise people 
of the quickest route that they should take to walk to the highest point of land within a 5km radius.
"""""

if __name__ == "__main__":

    """""  TASKS 1 & 2
    The application should ask the user to input their current location as a British National Grid coordinate
    (easting and northing). Then, it should test whether the user is within a box (430000, 80000) and (465000, 95000).
    If the input coor- dinate is outside this box, inform the user and quit the application. This is done because 
    the elevation raster provided to you extends only from (425000, 75000) to (470000, 100000) and the input point
    must be at least 5km from the edge of this raster.
    Identify the highest point within a 5km radius from the user location.
    To successfully complete this task you could (1) use the window function in rasterio to limit the size of your
    elevation array. If you do not use this window you may experience memory issues; or, (2) use a rasterised 5km buffer
    to clip an elevation array. Other solutions are also accepted. Moreover, if you are not capable to solve this task
    you can select a random point within 5km of the user.
    """""

    # Import elevation data
    elevation = rasterio.open('elevation/SZ.asc')

    # Import the background map
    background = rasterio.open('background/raster-50k_2724246.tif')

    # Ask the user for their location
    print("Please input your location:")
    north, east = float(input("east: ")), float(input("north: "))
    print(north, east)

    # Create a buffer zone of 5km
    location = Point(east, north)

    # create a to spec bounding box "tile"
    tile = Polygon([(430000, 80000), (430000, 95000), (465000, 95000), (465000, 80000)])

    buffer_zone = location.buffer(5000)
    if on_tile(buffer_zone, tile):
        print("Point is on tile")
    else:
        print("Please close the application")

    # Create an intersect polygon with the tile
    intersection_shape = location.buffer(5000).intersection(tile)

    # Get the buffer zone/ intersection coordinates
    x_bi, y_bi = intersection_shape.exterior.xy

    # Assign the bounding box of the intersect shape to window
    window = intersection_shape.bounds
    print("window dimensions are: ", window)

    # Create a numpy array subset of the elevation data
    with rasterio.open('elevation/SZ.asc') as raster:
        # Upper Left pixel coordinate
        ul = raster.index(*window[0:2])
        # Lower right pixel coordinate
        lr = raster.index(*window[2:4])
        # Create the usable window dimensions
        window_pixel = ((lr[0], ul[0]), (ul[1], lr[1]))
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
        all_touched=False,
    )

    #  Create a numpy array of the buffer zone
    masked_elevation_data = np.ma.array(data=elevation_window, mask=mask)
    # Rescale the elevation data
    rescaled_masked_elevation_array = np.kron(masked_elevation_data, np.ones((5, 5)))
    # Extract the coordinates of the highest points
    y, x = (np.where(rescaled_masked_elevation_array == np.amax(rescaled_masked_elevation_array)))
    # Choose the first value
    x = (x[0])
    y = (y[0])
    # Adjust the coordinates into the coordinate system
    easting = x + window[0]
    Northing = y + window[1]

    # Important variables:
    print(np.amax(masked_elevation_data))
    print("elevation window shape is ", elevation_window.shape)
    print("The elevation window shape is: ", elevation_window.shape)
    print("Highest point on the window is", np.amax(elevation_window))
    print("Highest point in the buffer zone", np.amax(masked_elevation_data))
    print("The window bounds are: ", window)

    # # Some test coordinates
    # # (459619, 85800)
    # # (439619, 85800)
    # # (450000, 90000) # Problem with this
    # # (430000, 90000)

    """""  Plotting
    Plot a background map 10km x 10km of the surrounding area. You are free to use either a 1:50k Ordnance Survey 
    raster (with internal color-map). Overlay a transparent elevation raster with a suitable color-map. Add the user’s
    starting point with a suitable marker, the highest point within a 5km buffer with a suitable marker, and the 
    shortest route calculated with a suitable line. Also, you should add to your map, a color-bar showing the elevation
     range, a north arrow, a scale bar, and a legend.
    """""

    # Plotting
    # Needs a 10km limit around the user
    # Needs to have an automatically adjusting North arrow and scale bar
    plt.ylabel("Northings")
    plt.xlabel("Eastings")
    plt.scatter(east, north, color="blue")
    plt.scatter(easting, Northing, color="red")  # High point
    plt.fill(x_bi, y_bi, color="skyblue", alpha=0.5)
    # rasterio.plot.show(background, alpha=0.2)
    rasterio.plot.show(elevation, background, alpha=0.5)
    plt.show()

    """""  TASKS 3 & 4
    Identify the nearest Integrated Transport Network (ITN) node to the user and the nearest ITN node to the highest 
    point identified in the previous step. To successfully complete this task you could use r-trees.
    Identify the shortest route using Naismith’s rule from the ITN node nearest to the user and the ITN node nearest 
    to the highest point.
    Naismith’s rule states that a reasonably fit person is capable of waking at 5km/hr and that an additional minute 
    is added for every 10 meters of climb (i.e., ascent not descent). To successfully complete this task you could 
    calculate the weight iterating through each link segment. Moreover, if you are not capable to solve this task 
    you could (1) approximate this algorithm by calculating the weight using only the start and end node elevation; 
    (2) identify the shortest distance from the node nearest the user to the node nearest the highest point using only
    inks in the ITN. To test the Naismith’s rule, you can use (439619, 85800) as a starting point.
    """""

    # Load the ITN network
    solent_itn_json = "itn/solent_itn.json"
    with open(solent_itn_json, "r") as f:
        solent_itn_json = json.load(f)
