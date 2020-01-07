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
from rasterio.transform import xy, rowcol, from_bounds


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


# Function to generate color path
def color_path(g, path, color="blue"):
    res = g.copy()
    first = path[0]
    for node in path[1:]:
        res.edges[first, node]["color"] = color
        first = node
    return res


# Function to obtain colours
def obtain_colors(graph, default_node="blue", default_edge="black"):
    node_colors = []
    for node in graph.nodes:
        node_colors.append( graph.nodes[node].get( 'color', default_node ) )
    edge_colors = []
    for u, v in graph.edges:
        edge_colors.append( graph.edges[u, v].get( 'color', default_edge ) )
    return node_colors, edge_colors


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

    # Create elevation array
    elevation_array = elevation.read( 1 )

    # Import the background map
    background = rasterio.open( "background/raster-50k_2724246.tif" )
    background_array = background.read( 1 )

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
    plot_buffer = location.buffer( 5000 )

    # Get the bounds for the 10km limits
    plot_buffer_bounds = tuple( plot_buffer.bounds )

    # Test is coordinate buffer zone is within bounding box
    if on_tile(
            buffer_zone,
            tile ):
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
    buffer_coordinates = generate_coordinates(
        easting_list,
        northing_list )

    # Warp the coordinates
    roi_polygon_src_coords = warp.transform_geom(
        {'init': 'EPSG:27700'},
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
    highest_east, highest_north = rasterio.transform.xy( out_transform,
                                                         highest_east,
                                                         highest_north,
                                                         offset='center' )

    # Create a 'shapley' point for the highest point
    highest_point_coordinates = Point( highest_east, highest_north )

    """""  
    IDENTIFY THE NETWORK
    --------------------
    Identify the nearest Integrated Transport Network (ITN) node to the user and the nearest ITN node to the highest 
    point identified in the previous step. To successfully complete this task you could use r-trees.
    Identify the shortest route using Naismith’s rule from the ITN node nearest to the user and the ITN node nearest 
    to the highest point.

    """""

    # Load the ITN network
    solent_itn_json = "itn/solent_itn.json"
    with open( solent_itn_json, "r" ) as f:
        solent_itn_json = json.load( f )
    road_links = solent_itn_json['roadlinks']

    # Create a list formed of all the 'roadnodes' coordinates
    road_nodes = solent_itn_json['roadnodes']
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
        start_node = road_nodes_list[i]

    # Use rTrees to query the nearest value to the finish
    for i in idx.nearest( query_finish, 1 ):
        finish_node = road_nodes_list[i]

    # Display the node coordinates
    print( "The start node is at: ", start_node )
    print( "The finish node is at: ", finish_node )

    # Extract the 'roadlinks' data
    road_links = solent_itn_json['roadlinks']

    # Place all road Id's into a list
    road_id_list = []
    for road_id in road_links:
        road_id_list.append( road_id )

    # Extract the first node
    for i in road_id_list:
        for j in range( len( road_links[i]["coords"] ) ):
            if road_links[i]["coords"][j] == start_node:
                first_node_id = str( road_links[i]["start"] )

    # Extract the finish node
    for i in road_id_list:
        for j in range( len( road_links[i]["coords"] ) ):
            if road_links[i]["coords"][j] == finish_node:
                last_node_id = str( road_links[i]["end"] )

    """""  
    FIND THE SHORTEST ROUTE
    -----------------------
    Naismith’s rule states that a reasonably fit person is capable of waking at 5km/hr and that an additional minute 
    is added for every 10 meters of climb (i.e., ascent not descent). To successfully complete this task you could 
    calculate the weight iterating through each link segment. Moreover, if you are not capable to solve this task 
    you could (1) approximate this algorithm by calculating the weight using only the start and end node elevation; 
    (2) identify the shortest distance from the node nearest the user to the node nearest the highest point using only
    inks in the ITN. To test the Naismith’s rule, you can use (439619, 85800) as a starting point.
    
    Every 0.72 seconds, we travel 1 meter 
    Therefore the time taken to travel each segment is 0.72 x the length of the segment 
    If the segment rises by more than 10 meters we can add a mintute 
    
    We need to turn the weighting value in to a time value
    
    """""


    # Some test coordinates
    # end to end
    # (85810, 439619) - Disjointed.
    # (85110, 450619  - Disjointed.
    # (85810, 457190) - good
    # (90000, 450000) - Disjointed
    # (90000, 430000) - Good
    # (85500, 439619) - Disjointed at start
    # (85500, 450619) - Very disjointed at end
    # (85970, 458898) - Good
    # (90000, 450619) - Good
    # (85110, 458898) - Disjointed.
    # (85810, 457190) = good
    # Shortest path test coordinate: (85800,  439619)

    # Create the network

    # Additional consideration - Function to ensure that all the roads are within the buffer zone
    def is_link_inside_polygon(coordinate, buffer):
        inside = True
        for x_y in coordinate:
            point_obj = Point( x_y )
            if not buffer.contains( point_obj ):
                inside = False
                break
        return inside


    def elevation_adjustment(coords, heights_array, affine_transform):
        total_climb = 0
        for i, point in enumerate( coords ):
            x, y = point
            if i == 0:
                last_height = heights_array[rowcol( affine_transform, x, y )]
            else:
                height = heights_array[rowcol( affine_transform, x, y )]
                if height > last_height:
                    total_climb += height - last_height

                last_height = height
        elevation_adjustment = total_climb / 10
        return elevation_adjustment


    # Populate a network with the edges and nodes
    network = nx.DiGraph()
    for link in road_links:
        road_length = road_links[link]['length']
        road_coordinates = road_links[link]['coords']

        # Exclude values that do not lie inside of the buffer zone
        if not is_link_inside_polygon(
                road_coordinates,
                buffer_zone ):
            continue

        # Calculate the basic travel time for a given road length
        basic_travel_time = road_length / 5000 * 60

        # For forward movements across the road link
        # Adjust the data to take into account the elevation change
        adjusted_to_elevation = elevation_adjustment(
            road_coordinates,
            elevation_array,
            out_transform )

        # Calculate the total time weight for forwards movement
        time_weight = basic_travel_time + adjusted_to_elevation

        # Populate the network with the forwards weighted edges
        network.add_edge(
            road_links[link]['start'],
            road_links[link]['end'],
            fid=link,
            length=road_links[link]['length'],
            time=time_weight )

        # For backwards movement across the road link
        # Adjust the elevation for the elevation
        adjusted_to_elevation = elevation_adjustment(
            (reversed( road_coordinates )),  # Reversed to account for moving backwards across the roadlink
            elevation_array,
            out_transform )

        # Calculate the total time weight for backwards movement
        time_weight = basic_travel_time + adjusted_to_elevation

        # Populate the network with the backwards weighted edges
        network.add_edge(
            road_links[link]['end'],
            road_links[link]['start'],
            fid=link,
            length=road_links[link]['length'],
            time=time_weight )

    # Identify the shortest path using dijkstra_path function
    path = nx.dijkstra_path(
        network,
        source=first_node_id,
        target=last_node_id,
        weight='time' )

    # assign the path the colour red
    shortest_path = color_path(
        network,
        path,
        "red" )

    # Retrieve the node colours
    node_colors, edge_colors = obtain_colors( shortest_path )

    links = []  # this list will be used to populate the feature id (fid) column
    geom = []  # this list will be used to populate the geometry column

    # Populate the shortest path
    first_node = path[0]
    for node in path[1:]:
        link_fid = network.edges[first_node, node]['fid']
        links.append( link_fid )
        geom.append( LineString( road_links[link_fid]['coords'] ) )
        first_node = node

    # Create Geopandas shortest path for plotting
    shortest_path_gpd = gpd.GeoDataFrame(
        {"fid": links, "geometry": geom} )

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
    shortest_path_gpd.plot( color="salmon", )
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
    plt.scatter( start_node[0], start_node[1], color="black", marker="x" )
    # Nearest node to user
    plt.scatter( highest_east, highest_north, color="white", marker=11 )
    # highest point
    plt.scatter( finish_node[0], finish_node[1], color="white", marker="x" )

    # plt.imshow(background.read(1)
    # Open rasterio
    # background.colormap(1)
    # put the colour scheme on the array
    # Plot the array
    # Value for key, value in background.colormap(1).items()
    # background)image = palette[back_array]
    # ax.imshow()
    # PLot the line between the user location and the and first node

    # PLot the line between the highest point and the last node

    # Plotting of the buffer zone
    plt.fill( x_bi, y_bi, color="skyblue", alpha=0.2 )

    # rasterio.plot.show(background, alpha=0.2) # todo work out how to overlay the rasterio plots
    # Plotting of the elevation
    rasterio.plot.show( elevation, alpha=1, contour=False )

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

    # Let the user know they are in the water, and plot it as a danger zone
    # Simple GUI to ask the user if they are walking / running / cycling
    # Return an answer if the user was on a bike or running
    # Return a value for the estimated number of steps the user will take
    # Returns some informatin about the weather conditions
    # Calorie Counter
    # Output text file

    print( "There is flooding in your area, you need to travel to The coordinates of your location are ", east, north,
           ", You need to travel to", highest_east, highest_north,
           "This location has a linear distance of ", (location.distance( highest_point_coordinates ) / 1000),
           "in meters" )
