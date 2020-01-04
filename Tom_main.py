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
from rasterio import plot, warp
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
from rasterio import mask

from shapely.wkt import loads
from numpy import asarray
from numpy import savetxt


# Function to test if any object is within a polygon
def on_tile(c, b):
    try:
        if b.contains( c ):
            return True
        else:
            return False
    except IOError:
        print( "Unable to perform this operation" )


# Function to join lists into list from 'list = [(x, y), (x, y)]'
def generate_coordinates(p_x, p_y):
    try:
        return list( map( lambda x, y: (x, y), p_x, p_y ) )
    except IOError:
        print( "Unable to perform this operation" )

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
    elevation = rasterio.open( 'elevation/SZ.asc' )

    # Import the background map
    background = rasterio.open( 'background/raster-50k_2724246.tif' )

    # Import the isle_of_wight shape
    island_shapefile = gpd.read_file( "shape/isle_of_wight.shp" )

    # todo: Import a polygon of the isle of wight, let the user know if they are in the water.

    # Ask the user for their location
    print( "Please input your location" )
    north, east = int( input( "east: " ) ), int( input( "north: " ) )

    # Create a buffer zone of 5km
    location = Point( east, north )

    # create a to spec bounding box "tile"
    tile = Polygon( [(430000, 80000), (430000, 95000), (465000, 95000), (465000, 80000)] )

    # Create a 5km buffer
    buffer_zone = location.buffer( 5000 )

    # Create a 10km buffer for plotting purposes
    plot_buffer = location.buffer( 10000 )

    # Get the bounds for the 10km limits
    plot_buffer_bounds = tuple( plot_buffer.bounds )

    # Test is coordinate buffer zone is within bounding box
    if on_tile( buffer_zone, tile ):
        print( " " )
    else:
        # The user is advised to quit the application
        print( "You location is not in range, please close the application" )
        # The code stops running
        sys.exit()

    # Create an intersect polygon with the tile
    intersection_shape = buffer_zone.intersection( tile )

    # Get the buffer zone/ intersection coordinates
    x_bi, y_bi = intersection_shape.exterior.xy

    # Create coordinate list to allow for iteration
    highest_east, highest_north = buffer_zone.exterior.xy
    easting_list = []
    northing_list = []
    for i in highest_east:
        easting_list.append( i )
    for i in highest_north:
        northing_list.append( i )
    buffer_coordinates = generate_coordinates( easting_list, northing_list )

    # Warp the coordinates
    roi_polygon_src_coords = warp.transform_geom( {'init': 'EPSG:27700'},
                                                  elevation.crs,
                                                  {"type": "Polygon",
                                                   "coordinates": [buffer_coordinates]} )

    # create an 3d array containing the elevation data masked to the buffer zone
    elevation_mask, out_transform = mask.mask( elevation,
                                               [roi_polygon_src_coords],
                                               crop=False )

    # Search for the highest point in the buffer zone
    highest_point = np.amax( elevation_mask )

    # Extract the indicies of the highest point in pixel coordinates
    z, highest_east, highest_north = np.where( highest_point == elevation_mask )

    # Isolate the first value from the list
    highest_east = highest_east[0]
    highest_north = highest_north[0]

    # Transform the pixel coordinates back to east/north
    highest_east, highest_north = rasterio.transform.xy( out_transform, highest_east, highest_north, offset='center' )

    # Create a 'shapley' point for the highest point
    highest_point_coordinates = Point( highest_east, highest_north )

    # Some test coordinates
    # (85810, 439619)
    # (85110, 450619
    # (85110, 455619)
    # (90000, 450000)
    # (90000, 430000)
    # (84651, 440619)
    # (85500, 439619)
    # (85500, 450619)
    # (90000, 450619)
    # (92000, 460619)

    print( "The coordinates of your location are ", east, north, ", You need to travel to", highest_east, highest_north,
           "This location has a linear distance of ", (location.distance( highest_point_coordinates ) / 1000),
           "in meters" )

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
    with open( solent_itn_json, "r" ) as f:
        solent_itn_json = json.load( f )

    # Create a list formed of all the 'roadnodes' coordinates
    road_nodes = road_links = solent_itn_json['roadnodes']
    road_nodes_list = []
    for nodes in road_nodes:
        road_nodes_list.append( road_nodes[nodes]["coords"] )

    # construct an index with the default construction
    idx = index.Index()

    # Insert the points into the index
    for i, p in enumerate( road_nodes_list ):
        idx.insert( i, p + p, p )

    # The query start point is the user location:
    query_start = (east, north)

    # The query finish point is the highest point
    query_finish = (highest_east, highest_north)

    # Find the nearest value to the start
    for i in idx.nearest( query_start, 1 ):
        first_node = road_nodes_list[i]

    # Find the nearest value to the finish
    for i in idx.nearest( query_finish, 1 ):
        last_node = road_nodes_list[i]

    print( "The start node is at: ", first_node )
    print( "The finish node is at: ", last_node )

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

    # Create an empty network
    g = nx.Graph()

    # Populate a network containing all the roadlinks
    road_links = solent_itn_json['roadlinks']
    for link in road_links:
        g.add_edge( road_links[link]['start'], road_links[link]['end'], fid=link, weight=road_links[link]['length'] )

    # Identify the start and finish nodes

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

    """""
    # Todo: Plotting check points:
    # Suitable marker for the user location
    # Suitable marker for the highest point
    # todo: Background map
    #  Elevation overlay
    # a 10km limit around the user
    # an automatically adjusting North arrow and scale bar
    # todo: Elevation side bar
    # todo: Elevation side bar
    # todo: A legend - Start / Highest / Shortest path
    plt.title( "Isle of Wight Flood Plan" )
    # y label
    plt.ylabel( "Northings" )
    # x label
    plt.xlabel( "Eastings" )
    # 10km northings limit
    plt.ylim( (plot_buffer_bounds[1], plot_buffer_bounds[3]) )
    # 10km easting limit
    plt.xlim( (plot_buffer_bounds[0], plot_buffer_bounds[2]) )
    # North Arrow (x, y) to (x+dx, y+dy).
    plt.arrow( plot_buffer_bounds[0] + 1000, plot_buffer_bounds[3] - 3000, 0, 1000, head_width=200 )
    plt.text( plot_buffer_bounds[0] + 800, plot_buffer_bounds[3] - 1000, "N" )
    # Scale bar (set to 5km)
    plt.arrow( plot_buffer_bounds[0] + 3000, plot_buffer_bounds[1] + 1000, 5000, 0 )
    plt.text( plot_buffer_bounds[0] + 3000 + 2500, plot_buffer_bounds[1] + 1200, "5km" )
    # User location
    plt.scatter( east, north, color="black", marker=11 )
    # Plot the first node
    plt.scatter( first_node[0], first_node[1], color="black", marker="x" )
    # Nearest node to user
    plt.scatter( highest_east, highest_north, color="red", marker=11 )
    # highest point
    plt.scatter( last_node[0], last_node[1], color="red", marker="x" )
    # Plotting of the buffer zone
    plt.fill( x_bi, y_bi, color="skyblue", alpha=0.4 )

    # rasterio.plot.show(background, alpha=0.2) # todo work out how to overlay the rasterio plots
    # Plotting of the elevation
    rasterio.plot.show( elevation, alpha=0.5 )

    # Create the plot
    plt.show()
    """""

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

    # Let the user know they are in the water, and plot it as a danger zone
    # Simple GUI to ask the user if they are walking / running / cycling
    # Return an answer if the user was on a bike or running
    # Return a value for the estimated number of steps the user will take
