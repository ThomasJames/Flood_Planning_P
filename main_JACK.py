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
from rasterio.enums import Resampling
import rasterio.transform as transform
from rasterio import windows
from tkinter import *
from tkinter import ttk


class InputForm():
    def __init__(self, prompt):
        self.prompt = prompt
        self.response = ""

        def ok():
            self.response = entry1.get(), entry2.get()
            master.destroy()

        master = Tk()
        style = ttk.Style()
        style.configure("BW.TLabel", foreground="black", background="white")
        w = Label(master, text="Please enter your position in Eastings and Northings")
        w.pack()
        entry1 = Entry(master)
        entry2 = Entry(master)
        entry1.pack()
        entry2.pack()
        master.geometry("170x200+30+30")
        entry1.focus_set()

        butt = ttk.Button(master, text="RUN", width=10, command=ok, style="")
        butt.pack()

        mainloop()


# The above class structure was found here: https://stackoverflow.com/questions/51832502/returning-a-value-from-a-tkinter-form

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


# Function to generate color path
def color_path(g, path, color="blue"):
    res = g.copy()
    first = path[0]
    for node in path[1:]:
        res.edges[first, node]["color"] = color
        first = node
    return res


# def create_window(point, raster_layer, buffer_distance=5000):
#    row_offset, col_offset = raster_layer.index(west_bound, north_bound)
#    row_opposite, col_opposite = raster_layer.index(east_bound, south_bound)
#    width = col_opposite - col_offset
#    height = row_opposite - row_offset
#    window = Window(col_offset, row_offset, width, height)
#    return window
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

    input_points = InputForm("Enter position").response

    print("Your coordinates are:", input_points)
    east = (int(input_points[0]))
    north = (int(input_points[1]))

    # Import elevation map
    elevation = rasterio.open('elevation/SZ.asc')

    # Import the background map
    background = rasterio.open("background/raster-50k_2724246.tif")


    # Upscaling raster to higher res
    # upscale_factor = 2
    #
    # with background as dataset:
    #
    #    # resample data to target shape
    #    data = dataset.read(
    #        out_shape=(dataset.count, int(dataset.width * upscale_factor), int(dataset.height * upscale_factor)),
    #        resampling=Resampling.bilinear)
    #
    #    # scale image transform
    #    transform = dataset.transform * dataset.transform.scale(dataset.width / data.shape[-2]), (
    #                dataset.height / data.shape[-1])

    # Import the isle_of_wight shape
    island_shapefile = gpd.read_file("shape/isle_of_wight.shp")

    # Create a buffer zone of 5km
    location = Point(north, east)

    # Is user on island
    user_on_land = (island_shapefile.contains(location))
    if user_on_land[0] == True:
        print("User is on land")
    else:
        print("Please swim to shore and start again")
        sys.exit()

    x_window_lower = (east - 5000)
    x_window_higher = (east + 5000)
    y_window_lower = (north - 5000)
    y_window_higher = (north + 5000)

    # x, y = (background.bounds.left, background.bounds.top)
    # row, col = background.index(x, y)
    # print(row, col)

    background_transform = background.transform
    bottom_left = background.transform * (0, 0)
    top_right = background.transform * (background.width, background.height)
    # print("top_right", top_right)
    # print("bottom left", bottom_left)

    window_lower_lim = rasterio.transform.rowcol(background_transform, y_window_lower, x_window_lower)
    window_upper_lim = rasterio.transform.rowcol(background_transform, y_window_higher, x_window_higher)
    print(window_lower_lim)
    # create a to spec bounding box "tile"
    tile = Polygon([(430000, 80000), (430000, 95000), (465000, 95000), (465000, 80000)])

    # Read a window of data
    slice_ = (slice(window_upper_lim[0], window_lower_lim[0]), slice(window_lower_lim[1], window_upper_lim[1]))
    window_slice = windows.Window.from_slices(*slice_)
    print("window slice", window_slice)
    print("background raster info", background.height, background.width, background.transform, background.crs)

    # Transform the window
    transform_window = windows.transform(window_slice, background.transform)

    window_map = background.read(1, window=window_slice)
    print("window map", window_map)

    palette = np.array([value for key, value in background.colormap(1).items()])
    island_raster_image = palette[window_map.astype(int)]
    window_map_raster = rasterio.plot.reshape_as_raster(island_raster_image)

    rasterio.plot.show(window_map_raster)

    # MAKE WINDOW AFTER BUFFER (LOOK UP)
    # RASTER WITH 10 KM WINDOW
    # TIFF AS NUMPY ARRAY WITH CORRECT DIMENSIONS
    # WINDOW TRANSFORM
    # APPLYING COLOUrMAP

    # this will make sense when i've got a window sorted
    # then, rasterio.plot.reshape_as_raster(island_raster_image)

    # print(window_map)
    # rasterio.plot.show(window_map)
    #
    # img = np.stack([background.read(4 - i, window=window_slice) for i in range(1, 4)], axis=-1)
    # img = np.clip(img, 0, 2200) / 2200
    #
    # print(img.shape)
    #
    # plt.figure(figsize=(8,8))
    # plot.show(img.transpose(2, 0, 1), transform=transform_window)

    # Create a 5km buffer
    buffer_zone = location.buffer(5000)

    # Create a 10km buffer for plotting purposes
    plot_buffer = location.buffer(10000)

    # Get the bounds for the 10km limits
    plot_buffer_bounds = tuple(plot_buffer.bounds)

    ##################
    # getting map in

    ####################

    # Test is coordinate buffer zone is within bounding box
    if on_tile(buffer_zone, tile):
        print(" ")
    else:
        # The user is advised to quit the application
        print("You location is not in range, please close the application")
        # The code stops running
        sys.exit()

    # Create an intersect polygon with the tile
    intersection_shape = buffer_zone.intersection(tile)

    # Get the buffer zone/ intersection coordinates
    x_bi, y_bi = intersection_shape.exterior.xy

    # Create coordinate list to allow for iteration
    highest_east, highest_north = buffer_zone.exterior.xy
    easting_list = []
    northing_list = []
    for i in highest_east:
        easting_list.append(i)
    for i in highest_north:
        northing_list.append(i)
    buffer_coordinates = generate_coordinates(easting_list, northing_list)

    # Warp the coordinates
    roi_polygon_src_coords = warp.transform_geom({'init': 'EPSG:27700'},
                                                 elevation.crs,
                                                 {"type": "Polygon",
                                                  "coordinates": [buffer_coordinates]})

    # create an 3d array containing the elevation data masked to the buffer zone
    elevation_mask, out_transform = mask.mask(elevation,
                                              [roi_polygon_src_coords],
                                              crop=False)

    # Search for the highest point in the buffer zone
    highest_point = np.amax(elevation_mask)

    # Extract the indicies of the highest point in pixel coordinates
    z, highest_east, highest_north = np.where(highest_point == elevation_mask)

    # Isolate the first value from the list
    highest_east = highest_east[0]
    highest_north = highest_north[0]

    # Transform the pixel coordinates back to east/north
    highest_east, highest_north = rasterio.transform.xy(out_transform, highest_east, highest_north, offset='center')

    # Create a 'shapley' point for the highest point
    highest_point_coordinates = Point(highest_east, highest_north)

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

    print("The coordinates of your location are ", east, north, ", You need to travel to", highest_east, highest_north,
          "This location has a linear distance of ", (location.distance(highest_point_coordinates) / 1000),
          "in meters")

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
    with open(solent_itn_json, "r") as f:
        solent_itn_json = json.load(f)

    # Create a list formed of all the 'roadnodes' coordinates
    road_nodes = solent_itn_json['roadnodes']
    road_nodes_list = []
    for nodes in road_nodes:
        road_nodes_list.append(road_nodes[nodes]["coords"])

    # construct an index with the default construction
    idx = index.Index()

    # Insert the points into the index
    for i, p in enumerate(road_nodes_list):
        idx.insert(i, p + p, p)

    # The query start point is the user location:
    query_start = (east, north)

    # The query finish point is the highest point
    query_finish = (highest_east, highest_north)

    # Find the nearest value to the start
    for i in idx.nearest(query_start, 1):
        start_node = road_nodes_list[i]

    # Find the nearest value to the finish
    for i in idx.nearest(query_finish, 1):
        finish_node = road_nodes_list[i]

    # Display the node coordinates
    print("The start node is at: ", start_node)
    print("The finish node is at: ", finish_node)

    # Extract the 'roadlinks' data
    road_links = solent_itn_json['roadlinks']

    # Place all road Id's into a list
    road_id_list = []
    for road_id in road_links:
        road_id_list.append(road_id)

    # Extract the first node
    for i in road_id_list:
        for j in range(len(road_links[i]["coords"])):
            if road_links[i]["coords"][j] == start_node:
                first_node_id = str(road_links[i]["end"])

    # Show the first node id
    print("First node id is: ", first_node_id)

    # Extract the finish node
    for i in road_id_list:
        for j in range(len(road_links[i]["coords"])):
            if road_links[i]["coords"][j] == finish_node:
                last_node_id = str(road_links[i]["end"])

    test_first_node = 'osgb4000000026146800'
    test_last_node = 'osgb4000000026145458'

    # Show the last node id
    print("last node id is: ", last_node_id)

    """""  
    FIND THE SHORTEST ROUTE
    -----------------------

    Naismith’s rule states that a reasonably fit person is capable of waking at 5km/hr and that an additional minute 
    is added for every 10 meters of climb (i.e., ascent not descent). To successfully complete this task you could 
    calculate the weight iterating through each link segment. Moreover, if you are not capable to solve this task 
    you could (1) approximate this algorithm by calculating the weight using only the start and end node elevation; 
    (2) identify the shortest distance from the node nearest the user to the node nearest the highest point using only
    inks in the ITN. To test the Naismith’s rule, you can use (439619, 85800) as a starting point.

    Let’s make a simple 3x3 Manhattan road network:
    g = nx.Graph()
    w, h = 3, 3
    We label our nodes in accordance with the formula defined by this function:
    def get_id(r, c):
        return r + c * w
    We now add the nodes to the graph:
    for r in range(h):
        for c in range(w):
            g.add_node(get_id(r, c))
            print(get_id(r, c))

    """""

    # Create an empty network
    g = nx.Graph()

    # Populate a network containing all the roadlinks
    road_links = solent_itn_json['roadlinks']
    for link in road_links:
        g.add_edge(road_links[link]['start'], road_links[link]['end'], fid=link, weight=road_links[link]['length'])

    # Identify the shortest path
    path = nx.dijkstra_path(g, source=first_node_id, target=last_node_id)

    # assign the path the colour red
    shortest_path = color_path(g, path, "red")

    # Retrieve the nod coloutsd
    # node_colors, edge_colors = obtain_colors(shortest_path)

    links = []  # this list will be used to populate the feature id (fid) column
    geom = []  # this list will be used to populate the geometry column

    # Populate the shortest path
    first_node = path[0]
    for node in path[1:]:
        link_fid = g.edges[first_node, node]['fid']
        links.append(link_fid)
        geom.append(LineString(road_links[link_fid]['coords']))
        first_node = node

    # Create Geopandas shortest path for plotting
    shortest_path_gpd = gpd.GeoDataFrame({"fid": links, "geometry": geom})

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
    shortest_path_gpd.plot(color="salmon", )

    plt.title("Isle of Wight Flood Plan")
    # y label
    plt.ylabel("Northings")
    # x label
    plt.xlabel("Eastings")
    # 10km northings limit
    plt.ylim((plot_buffer_bounds[1], plot_buffer_bounds[3]))
    # 10km easting limit
    plt.xlim((plot_buffer_bounds[0], plot_buffer_bounds[2]))
    # North Arrow (x, y) to (x+dx, y+dy).
    plt.arrow(plot_buffer_bounds[0] + 1000, plot_buffer_bounds[3] - 3000, 0, 1000, head_width=200)
    plt.text(plot_buffer_bounds[0] + 800, plot_buffer_bounds[3] - 1000, "N")
    # Scale bar (set to 5km)
    plt.arrow(plot_buffer_bounds[0] + 3000, plot_buffer_bounds[1] + 1000, 5000, 0)
    plt.text(plot_buffer_bounds[0] + 3000 + 2500, plot_buffer_bounds[1] + 1200, "5km")
    # User location
    plt.scatter(east, north, color="black", marker=11)
    # Plot the first node
    plt.scatter(start_node[0], start_node[1], color="black", marker="x")
    # Nearest node to user
    plt.scatter(highest_east, highest_north, color="white", marker=11)
    # highest point
    plt.scatter(finish_node[0], finish_node[1], color="white", marker="x")

    # PLot the line between the user location and the and first node

    # PLot the line between the highest point and the last node

    # Plotting of the buffer zone
    plt.fill(x_bi, y_bi, color="skyblue", alpha=0.2, zorder=0)

    # rasterio.plot.show(background, alpha=0.2) # todo work out how to overlay the rasterio plots
    # Plotting of the elevation
    # rasterio.plot.show(elevation, alpha=1, contour=False, zorder=0)
    # rasterio.plot.show(background, alpha=0.5, contour=False, zorder=1)

    fig, ax = plt.subplots(dpi=300)
    ax.set_xlim([y_window_lower, y_window_higher])
    ax.set_ylim([x_window_lower, x_window_higher])
    rasterio.plot.show(window_map_raster, ax=ax, zorder=1, transform=transform_window)
    rasterio.plot.show(elevation_mask, transform=out_transform, ax=ax, zorder=2, alpha=0.5, cmap='inferno')
    plt.show()
    # then put all of the rasterio plots on after this
    # YOU MUST SPECIFY WHAT AXIS YOU ARE ON WITH ax=ax
    # use the correct transforms
    cmap =
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

