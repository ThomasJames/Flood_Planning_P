import numpy as np
import sys
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

# Function to test if the user is on the polygon
def on_land(north, east):
    try:
        if polygon.contains(pt1):
            return True
        else:
            return False
    except IOError:
        print("Unable to perform this operation")

# Function to test if any object is within a polygon
def on_tile(c, b):
    try:
        if b.contains(c):
            return True
        else:
            return False
    except IOError:
        print("Unable to perform this operation")


# Function to join lists into list from 'list = [(x, y), (x, y)]'
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

    """""  
    USER INPUT 
    ----------
    The application should ask the user to input their current location as a British National Grid coordinate
    (easting and northing). Then, it should test whether the user is within a box (430000, 80000) and (465000, 95000).
    If the input coor- dinate is outside this box, inform the user and quit the application. This is done because 
    the elevation raster provided to you extends only from (425000, 75000) to (470000, 100000) and the input point
    must be at least 5km from the edge of this raster.

    HIGHEST POINT IDENTIFICATION    
    ----------------------------  
    Identify the highest point within a 5km radius from the user location.
    To successfully complete this task you could (1) use the window function in rasterio to limit the size of your
    elevation array. If you do not use this window you may experience memory issues; or, (2) use a rasterised 5km buffer
    to clip an elevation array. Other solutions are also accepted. Moreover, if you are not capable to solve this task
    you can select a random point within 5km of the user.
    """""

    # Import elevation map
    elevation = rasterio.open('elevation/SZ.asc')

    # Import the background map
    background = rasterio.open('background/raster-50k_2724246.tif')

    # Import island shapefile
    shape_file = gpd.read_file('shape/isle_of_wight.shp')

    # Ask the user for their location
    print("Please input your location")
    north, east = int(input("east: ")), int(input("north: "))
    print(north, east)

    # todo: Import a polygon of the isle of wight, let the user know if they are in the water.

    pt1 = Point(north, east)
    x, y = pt1.xy
    polygon = shape_file.geometry
    plt.plot(x, y, 'ro')
    polygon.contains(pt1)

    # Create a buffer zone of 5km
    location = Point(east, north)

    # create a to spec bounding box "tile"
    tile = Polygon([(430000, 80000), (430000, 95000), (465000, 95000), (465000, 80000)])

    # Create a 5km buffer
    buffer_zone = location.buffer(5000)

    # Create a 10km buffer for plotting purposes
    plot_buffer = location.buffer(10000)

    # Get the bounds for the 10km limits
    plot_buffer_bounds = tuple(plot_buffer.bounds)
    print(plot_buffer_bounds[0])

    # todo: Problem - POLYGON does not supprot indexing - Need to resolve

    # Test is coordinate buffer zone is within bounding box
    if on_tile(buffer_zone, tile):
        print("Point is on tile")
    else:
        # The user is advised to quit the application
        print("Please close the application")
        # The code stops running
        sys.exit()

    # Create an intersect polygon with the tile
    intersection_shape = buffer_zone.intersection(tile)

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

        # Extract the x and y pixel coordinates of the exterior
        intersection_pixel_x, intersection_pixel_y = raster.index(*intersection_shape.exterior.xy)

        # Generate the (x, y) coordinate format
        intersection_pixel_coords = generate_coordinates(intersection_pixel_x, intersection_pixel_y)

        # Generate a 'shapley' polygon
        intersection_pixel_polygon = Polygon(intersection_pixel_coords)

        # Rasterize the geometry
        mask = rasterio.features.rasterize(
            [(intersection_pixel_polygon, 0)],  # Masking shape
            out_shape=elevation_window.shape,
            all_touched=False
        )

    #  Create a numpy array of the buffer zone todo: does this actually mask the outter bounds?
    masked_elevation_data = np.ma.array(data=elevation_window, mask=mask)

    # Rescale the elevation data todo: is there a function to extract the coordinates without rescale
    # todo: maybe we could use pyproj/projection transformations?
    rescaled_masked_elevation_array = np.kron(masked_elevation_data, np.ones((5, 5)))
    highest_east_index, highest_north_index = (np.where(masked_elevation_data == np.amax(masked_elevation_data)))
    print(highest_east_index)
    print(highest_north_index)

    # Extract the coordinates of the highest points #
    highest_east_index, highest_north_index = (
        np.where(rescaled_masked_elevation_array == np.amax(rescaled_masked_elevation_array)))
    print(highest_north_index)
    print(highest_east_index)

    # Choose the first value
    highest_east_index = (highest_east_index[0])
    highest_north_index = (highest_north_index[0])

    # Adjust the coordinates into the coordinate system
    highest_east = highest_east_index + window[0]
    highest_north = highest_north_index + window[1]

    # Create a shapely point for the highest point
    highest_point_coord = Point(highest_east, highest_north)

    # Calculate the distance that the user will have to travel
    linear_distance_to_travel = highest_point_coord.distance(location) / 1000

    # Important variables:
    #print("The user is on the island: ", on_land(north, east))
    print("The distance to travel in kilometers is: ", linear_distance_to_travel)
    print("Highest point in masked elevation data", np.amax(masked_elevation_data))
    print("elevation window shape is ", elevation_window.shape)
    print("The elevation window shape is: ", elevation_window.shape)
    print("Highest point on the window is", np.amax(elevation_window))
    print("Highest point in the buffer zone", np.amax(masked_elevation_data))
    print("The window bounds are: ", window)

    # Some test coordinates
    # (85800, 439619) # Looks ok
    # (90000, 450000) # Out of range
    # (90000, 430000) # Out of range
    # (85500, 440619) # Looks ok
    # (85500, 460619) # Out of range
    # (85500, 450619) # Looks good
    # (90000, 450619) # Out of range
    # (92000, 460619) # In range but wrong

    """""  
    IDENTIFY THE NETWORK
    --------------------
    Identify the nearest Integrated Transport Network (ITN) node to the user and the nearest ITN node to the highest 
    point identified in the previous step. To successfully complete this task you could use r-trees.
    Identify the shortest route using Naismith’s rule from the ITN node nearest to the user and the ITN node nearest 
    to the highest point.

    Creating an index tutorial
    https://rtree.readthedocs.io/en/latest/tutorial.html#creating-an-index

    Worked example 
    https://towardsdatascience.com/connecting-pois-to-a-road-network-358a81447944
    """""

    # Load the ITN network
    solent_itn_json = "itn/solent_itn.json"
    with open(solent_itn_json, "r") as f:
        solent_itn_json = json.load(f)

    # Create a list formed of all the 'roadnodes' coordinates
    road_nodes = road_links = solent_itn_json['roadnodes']
    road_nodes_list = []
    for nodes in road_nodes:
        road_nodes_list.append(road_nodes[nodes]["coords"])

    # Check the coordinates
    print(road_nodes_list)

    # construct an index with the default construction
    idx = index.Index()

    # Insert the points into the index
    for i, p in enumerate(road_nodes_list):
        idx.insert(i, p + p, p)

    # The query start point is the user location:
    query_start = (east, north)
    print(query_start)

    # The query finish point is the highest point
    query_finish = (highest_east, highest_north)
    print(query_finish)

    # Find the nearest value to the start
    for i in idx.nearest(query_start, 1):
        nearest_node_to_start = road_nodes_list[i]

    # Find the nearest value to the finish
    for i in idx.nearest(query_finish, 1):
        nearest_node_to_finish = road_nodes_list[i]

    """""  
    FIND THE SHORTEST ROUTE
    -----------------------

    Naismith’s rule states that a reasonably fit person is capable of waking at 5km/hr and that an additional minute 
    is added for every 10 meters of climb (i.e., ascent not descent). To successfully complete this task you could 
    calculate the weight iterating through each link segment. Moreover, if you are not capable to solve this task 
    you could (1) approximate this algorithm by calculating the weight using only the start and end node elevation; 
    (2) identify the shortest distance from the node nearest the user to the node nearest the highest point using only
    inks in the ITN. To test the Naismith’s rule, you can use (439619, 85800) as a starting point.
    """""

    # Find the shortest path
    # todo: How do you access weights based gradient?
    # path = nx.dijkstra_path(g, source=start, target=end, weight="weight")
    # print(path)

    # The first step is to iterate through each of the nodes on the shortest path calculated. Ignore the first node, but
    # instead assign it to a variable called first_node. Starting with the second node, we find the fid of road link
    # that connects the first_node and node. Knowing the roadlink fid, we can find the coordinates and make a shapely
    # LineString object. The final step of each iteration is to set first_node so that it can be used in the next
    # iteration. On each iteration we append the feature id and the geometry to two lists links and geom which are used
    # to build the path_gpd GeoDataFrame.

    """""  
    PLOTTING
    --------
    Plot a background map 10km x 10km of the surrounding area. You are free to use either a 1:50k Ordnance Survey 
    raster (with internal color-map). Overlay a transparent elevation raster with a suitable color-map. Add the user’s
    starting point with a suitable marker, the highest point within a 5km buffer with a suitable marker, and the 
    shortest route calculated with a suitable line. Also, you should add to your map, a color-bar showing the elevation
    range, a north arrow, a scale bar, and a legend.
    To create a GeoDataFrame of the shortest path and then display it on top of a raster.
    We shall be using the following packages and the background map. 
    import rasterio
    import pyproj
    import numpy as np
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from shapely.geometry import LineString  
    """""

    # Plotting
    # todo: a 10km limit around the user
    # todo: an automatically adjusting North arrow and scale bar

    # y label
    plt.ylabel("Northings")
    # x label
    plt.xlabel("Eastings")
    # 10km northings limit
    plt.ylim((plot_buffer_bounds[1], plot_buffer_bounds[3]))
    # 10km easting limit
    plt.xlim((plot_buffer_bounds[0], plot_buffer_bounds[2]))
    # bounding box
    plt.plot([(430000, 80000), (430000, 95000), (465000, 95000), (465000, 80000)])
    # User location
    plt.scatter(east, north, color="black", marker="^")
    plt.scatter(nearest_node_to_start[0], nearest_node_to_start[1], color="black", marker="*")
    # Nearest node to user
    plt.scatter(highest_east, highest_north, color="red", marker="^")
    # highest point
    plt.scatter(nearest_node_to_finish[0], nearest_node_to_finish[1], color="red", marker="*")
    # Plotting of the buffer zone
    plt.fill(x_bi, y_bi, color="skyblue", alpha=0.4)
    # rasterio.plot.show(background, alpha=0.2) # todo work out how to overlay the rasterio plots
    # Plotting of the elevation
    rasterio.plot.show(elevation, background, alpha=0.5)
    # Create the plot
    plt.show()

    """""
    EXTENDING THE REGION
    --------------------
    The position of the user is restricted to a region in where the user must be more than 5km from the edge of the 
    elevation raster. Write additional code to overcome this limitation.   
    """""

    # Potential Solutions:
    # Create a function that generates a bounding box that adjusts to the limits of the existing raster.
    # If the user is outside the region, tell them.
    # Create a directory of raster files that correspond to the users coordinates, apply the relevant tile.

    """""
    ADDITONAL IDEAS 
    ---------------
    """""
    # Simple GUI to ask the user if they are walking / running / cycling
    # Return an answer if the user was on a bike or running
    # Return a value for the estimated number of steps the user will take
