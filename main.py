from shapely.geometry import Point, box
from shapely.geometry import LineString
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from rasterio import plot, warp
from rtree import index
import networkx as nx
import rasterio
import numpy as np
import geopandas as gpd
import json
from rasterio import mask
from rasterio import windows
import tkinter as tk
from tkinter import *
from rasterio.transform import xy, rowcol, from_bounds


# GUI used to input co-ordinates
# This class was an adaptation of code by Azat Lbrakov, 2017:
# Link: https://stackoverflow.com/questions/51832502/returning-a-value-from-a-tkinter-form
class MyWindow:
    def __init__(self, win):
        self.lbl1 = Label(win, text='Northing')
        self.lbl2 = Label(win, text='Easting')
        self.ibl3 = Label(win, text='Please input your current location and press enter').pack()
        self.var1 = tk.StringVar("")
        self.var2 = tk.StringVar("")
        self.t1 = Entry(bd=3, textvariable=self.var1)
        self.t2 = Entry(bd=3, textvariable=self.var2)
        self.lbl1.place(x=100, y=50)
        self.t1.place(x=200, y=50)
        self.lbl2.place(x=100, y=100)
        self.t2.place(x=200, y=100)
        self.b1 = Button(win, text='Run "Higher Ground" Protocol', command=self.add)
        self.b1.place(x=100, y=150)

    def add(self):
        try:
            global east1, north1
            east1 = self.var1.get()
            north1 = self.var2.get()
            print(east1, north1)
            mywin.t1.delete(0, 'end')  # clear the input each time after press the button
            mywin.t2.delete(0, 'end')
            window.destroy()

        except:
            print("You should input the coordinate of your location")


window = tk.Tk()

mywin = MyWindow(window)

# Give the window a title
window.title('Flood Protection Program')

# Assign the dimensions of the user interface box
window.geometry("400x300+10+10")

# be able to use "Enter" key to run the GUI
window.bind('<Return>', lambda event: mywin.add())

# make the cursor appear in the first entry initially
mywin.t1.focus()

window.mainloop()

# Variables assigned from GUI to start user position in main
eastinput = int(east1)
northinput = int(north1)


# Function to join lists into list from 'list = [(x, y), (x, y)]'
# p_x must be a list of x coordinates
# p_y must be a list of y coordinates
def generate_coordinates(p_x, p_y):
    try:
        return list(map(lambda x, y: (x, y), p_x, p_y))
    except IOError:
        print("Unable to perform this operation")


# Function to test if any object is within a polygon
# shapley_object can be any shapley object
# Shape - Must be a shapley shape
def is_point_or_shape_in_shape(shapley_object, shape):
    try:
        if shape.contains(shapley_object):
            return True
        else:
            return False
    except IOError:
        print("Unable to perform this operation")


# Additional consideration - Function to ensure that all the roads are within the buffer zone.
# Coordinate argument - Can be in (x, y) form
# Buffer argument - Must be a shapely file.
# Code adapted from a function by Mahmoud Abdelrazek, 2019 - https://github.com/razekmh
def is_link_inside_polygon(coordinate, buffer):
    try:
        for coord_x_y in coordinate:
            point = Point(coord_x_y)  # Convert to shapley point
            if buffer.contains(point):  # Test for point in polygon
                return True
            else:
                return False
    except IOError:
        print("Unable to perform this operation")


# Function to return adjustment time
# coords - Must be the coordinates in [x, y]
# elevation_array - Must be a numpy array containing the elevation data
# transformation_matrix - The transformation matrix output.
# Code adapted from a function by Mahmoud Abdelrazek, 2019 - https://github.com/razekmh
def elevation_adjustment(coords, elevation_array, transformation_matrix):
    rise = 0  # Initialise rise to zero
    try:
        for i, point in enumerate(coords):  # Extract coordinates
            x_coord, y_coord = point
            if i == 0:  # First value
                back_height = elevation_array[rowcol(transformation_matrix, x_coord, y_coord)]  # Get elevation
            else:
                fore_height = elevation_array[rowcol(transformation_matrix, x_coord, y_coord)]  # Get elevation
                if fore_height > back_height:  # Ignore negative elevation changes
                    rise += fore_height - back_height  # Add each posotive changes to the rise
                back_height = fore_height
        elevation_adjustment = rise * 0.1  # To get the adjustment value in minutes.
        return elevation_adjustment
    except IOError:
        print("Unable to perform this operation")


# Function to generate color path
# network - The network that contains the edges
# path - the output path from the dijkstra_path
# color - Built in default to blue
# Adaptation of code by Aldo Lipani, 2019
def color_path(ntwrk, path, color="blue"):
    try:
        res = ntwrk.copy()
        first = path[0]
        for node in path[1:]:
            res.edges[first, node]["color"] = color
            first = node
        return res
    except IOError:
        print("Unable to perform this operation")


# Function to obtain colours
# Graph to which you wish to obtain the colours
# Default nodes and edges are assigned within the function
# Adaptation of code by Aldo Lipani, 2019
def obtain_colors(graph, default_node="blue", default_edge="black"):
    try:
        nde_clr = []
        for node in graph.nodes:
            nde_clr.append(graph.nodes[node].get('color', default_node))
        edge_clr = []
        for u, v in graph.edges:
            edge_clr.append(graph.edges[u, v].get('color', default_edge))
        return nde_clr, edge_clr
    except IOError:
        print("Unable to perform this operation")


# Function to create a smaller interior box given a buffer area
# buffer argument takes the buffer range yo would like to create
# x is a list of x values
# y is a list of y values
def create_buffer_box(buffer, x, y):
    return \
        [(x[2] + buffer, y[0] + buffer),
         (x[2] + buffer, y[1] - buffer),
         (x[0] - buffer, y[1] + buffer),
         (x[0] - buffer, y[0] - buffer)]


if __name__ == "__main__":

    """""  
    IMPORTING AN MANIPULATION OF DATA 
    """""

    # Import the background map
    background = rasterio.open("...Material/background/raster-50k_2724246.tif")

    # Create a background NumPy array
    background_array = background.read(1)

    # Import the isle_of_wight shape
    island_shapefile = gpd.read_file("...Material/shape/isle_of_wight.shp")

    # Import elevation map
    elevation = rasterio.open("...Material/elevation/SZ.asc")

    # Retrieve the coordinates of the elevation box
    elevation_box_xy = elevation.bounds
    elevation_raster_box = box(*list(elevation.bounds))
    elevation_box_x, elevation_box_y = elevation_raster_box.exterior.xy

    # Append the x and y values to lists to be used in buffer box function
    e_x = []
    e_y = []
    for i in elevation_box_x:
        e_x.append(i)
    for i in elevation_box_y:
        e_y.append(i)

    # Create the buffer box file by calling the create buffer box function
    buffer_box = Polygon(create_buffer_box(5000, e_x, e_y))

    # Create elevation numpy array
    elevation_array = elevation.read(1)

    """""  
    RETRIEVE AND MANIPULATE THE USER INPUT   
    """""

    # Request the user location
    print("Your location has been input.")
    # east, north = int(input("north: ")), int(input("east: "))

    east = int(eastinput)
    north = int(northinput)

    # Create a shapley point
    location = Point(north, east)

    buffer_zone = location.buffer(5000)

    # Create a 10km buffer for plotting purposes
    plot_buffer = location.buffer(5000)

    # Get the bounds for the 10km limits
    plot_buffer_bounds = tuple(plot_buffer.bounds)

    # Create
    elevation_raster_buffer_intersect = elevation_raster_box.intersection(buffer_zone)

    # Check if user is on island
    user_on_land = (island_shapefile.contains(location))
    if user_on_land[0] == True:
        print("User is on land")
    else:
        print("Swim to shore")

    # Test is coordinate within buffer zone is within the specified bounding box
    if is_point_or_shape_in_shape(
            buffer_zone,
            buffer_box):
        print(" ")
    else:
        print("You aren't in the specified bounding box please wait...")
        if input("click \"y\" if you Would you like to extend the region? ") == "y":
            if is_point_or_shape_in_shape(elevation_raster_buffer_intersect, elevation_raster_box):
                print("Region has been extended")
            else:
                print("You location is not in range")
                sys.exit()

    """""  
    PREPARE FOR PLOTTING 
    """""

    # Create window bounds
    x_window_lower = (east - 5000)
    x_window_higher = (east + 5000)
    y_window_lower = (north - 5000)
    y_window_higher = (north + 5000)

    # Transform of shape of background raster array
    background_transform = background.transform
    bottom_left = background.transform * (0, 0)
    top_right = background.transform * (background.width, background.height)

    # Transform the lower limit
    window_lower_lim = rasterio.transform.rowcol(background_transform,
                                                 y_window_lower,
                                                 x_window_lower)

    # Transform the upper limit
    window_upper_lim = rasterio.transform.rowcol(background_transform,
                                                 y_window_higher,
                                                 x_window_higher)

    # Read a window of data
    slice_ = (slice(window_upper_lim[0],
                    window_lower_lim[0]),
              slice(window_lower_lim[1],
                    window_upper_lim[1]))

    # extract rasterio data from array as slice
    window_slice = windows.Window.from_slices(*slice_)

    # Transform the window
    transform_window = windows.transform(window_slice,
                                         background.transform)
    # Extract array rows as slices
    window_map = background.read(1, window=window_slice)

    # Assign colour palette to array values for plotting
    palette = np.array([value for key, value in background.colormap(1).items()])

    # Array as plotting image
    island_raster_image = palette[window_map.astype(int)]

    # Convert rasterio image data with palette applied to new plotting variable
    window_map_raster = rasterio.plot.reshape_as_raster(island_raster_image)

    # Create coordinate list to allow for iteration
    highest_east, highest_north = buffer_zone.exterior.xy

    easting_list = []
    northing_list = []

    for i in highest_east:
        easting_list.append(i)
    for i in highest_north:
        northing_list.append(i)
    buffer_coordinates = generate_coordinates(
        easting_list,
        northing_list)

    # Warp the coordinates
    warped_elevation_coordinates = warp.transform_geom(
        {'init': 'EPSG:27700'},
        elevation.crs,
        {"type": "Polygon",
         "coordinates": [buffer_coordinates]})

    # create an 3d array containing the elevation data masked to the buffer zone
    elevation_mask, out_transform = mask.mask(elevation,
                                              [warped_elevation_coordinates],
                                              crop=False)

    # Search for the highest point in the buffer zone
    highest_point = np.amax(elevation_mask)

    # Extract the index of the highest point in pixel coordinates
    z, highest_east, highest_north = np.where(highest_point == elevation_mask)

    # Isolate the first value from the list
    highest_east = highest_east[0]
    highest_north = highest_north[0]

    # Transform the pixel coordinates back to east/north
    highest_east, highest_north = rasterio.transform.xy(out_transform,
                                                        highest_east,
                                                        highest_north,
                                                        offset='center')

    # Create a 'shapley' point for the highest point
    highest_point_coordinates = Point(highest_east, highest_north)

    # Get dimensions of the entire raster
    raster_pixel_xy_max = (elevation_array.shape[0], elevation_array.shape[1])

    """""  
    IDENTIFY THE NETWORK
    """""
    # Load the ITN network
    solent_itn_json = "...Material/itn/solent_itn.json"
    with open(solent_itn_json, "r") as f:
        solent_itn_json = json.load(f)
    road_links = solent_itn_json['roadlinks']

    # Create a list formed of all the 'roadnodes' coordinates
    road_nodes = solent_itn_json['roadnodes']
    road_nodes_list = []
    for nodes in road_nodes:
        road_nodes_list.append(road_nodes[nodes]["coords"])

    # construct an index with the default construction
    idx = index.Index()

    # Populate the index with the points
    for i, p in enumerate(road_nodes_list):
        idx.insert(i, p + p, p)

    # The query start point is the user location:
    query_start = (north, east)

    # The query finish point is the highest point
    query_finish = (highest_east, highest_north)

    # Find the nearest value to the start
    for i in idx.nearest(query_start, 1):
        start_node = road_nodes_list[i]

    # Use rTrees to query the nearest value to the finish
    for i in idx.nearest(query_finish, 1):
        finish_node = road_nodes_list[i]

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
                first_node_id = str(road_links[i]["start"])

    # Extract the finish node
    for i in road_id_list:
        for j in range(len(road_links[i]["coords"])):
            if road_links[i]["coords"][j] == finish_node:
                last_node_id = str(road_links[i]["end"])

    """""  
    FIND THE SHORTEST ROUTE
    """""

    # Populate a network with the edges and nodes
    network = nx.DiGraph()
    for link in road_links:
        road_length = road_links[link]['length']
        road_coordinates = road_links[link]['coords']

        # Exclude values that do not lie inside of the buffer zone
        if not is_link_inside_polygon(
                road_coordinates,
                buffer_zone):
            continue

        # For forward movements across the road link:
        # Adjust the data to take into account the elevation change
        adjusted_to_elevation = (elevation_adjustment(
            road_coordinates,
            elevation_array,
            out_transform))

        # Calculate the basic travel time for a given road length
        basic_travel_time = road_length / 5000 * 60

        # Calculate the total time weight for forwards movement
        time_weight = basic_travel_time + adjusted_to_elevation

        # Populate the network with the forwards weighted edges
        network.add_edge(
            road_links[link]['start'],
            road_links[link]['end'],
            fid=link,
            length=road_links[link]['length'],
            time=time_weight)

        # For backwards movement across the road link
        # Adjust the elevation for the elevation
        adjusted_to_elevation = elevation_adjustment(
            (reversed(road_coordinates)),  # Reversed to account for moving backwards across the roadlink
            elevation_array,
            out_transform)

        # Calculate the total time weight for backwards movement
        time_weight = basic_travel_time + adjusted_to_elevation

        # Populate the network with the backwards weighted edges
        network.add_edge(
            road_links[link]['end'],
            road_links[link]['start'],
            fid=link,
            length=road_links[link]['length'],
            time=time_weight)

    # Identify the shortest path using dijkstra_path function
    path = nx.dijkstra_path(network,
                            source=first_node_id,
                            target=last_node_id,
                            weight='time')

    # assign the path the colour red
    shortest_path = color_path(network,
                               path,
                               "red")

    # Retrieve the node colours
    node_colors, edge_colors = obtain_colors(shortest_path)

    # this list will be used to populate the feature id (fid) column
    links = []
    # this list will be used to populate the geometry column
    geom = []

    # Populate the shortest path
    first_node = path[0]
    for node in path[1:]:
        link_fid = network.edges[first_node, node]['fid']
        links.append(link_fid)
        geom.append(LineString(road_links[link_fid]['coords']))
        first_node = node

    # Create Geopandas shortest path for plotting
    shortest_path_gpd = gpd.GeoDataFrame({"fid": links,
                                          "geometry": geom})

    """""
    PLOTTING                                           
    --------
    """""

    # Custom paint job
    cmap = plt.get_cmap('inferno')

    # Set level of colour map
    cmap.set_under('r', alpha=0)

    # All plotting small but collated
    fig, ax = plt.subplots(dpi=300)

    # Assign elevation raster as image to plot
    elevation_plot = ax.imshow(elevation_mask[0, :, :],
                               cmap='inferno',
                               zorder=2)

    # Plot colourbar
    fig.colorbar(elevation_plot,
                 ax=ax)

    # Plot buffer
    ax.set_xlim([plot_buffer_bounds[0],
                 plot_buffer_bounds[2]])

    ax.set_ylim([plot_buffer_bounds[1],
                 plot_buffer_bounds[3]])

    # Plot background map
    rasterio.plot.show(window_map_raster,
                       ax=ax, zorder=1,
                       transform=transform_window)
    # Plot elevation data
    rasterio.plot.show(elevation_mask,
                       transform=out_transform,
                       ax=ax,
                       zorder=5,
                       alpha=0.3,
                       cmap=cmap,
                       vmin=0.01,
                       label="elevation buffer")
    # Plot shortest path
    shortest_path_gpd.plot(ax=ax,
                           edgecolor='black',
                           linewidth=3,
                           label="shortest path",
                           zorder=10)

    # Title
    plt.title("Isle of Wight Flood Plan")

    # y label
    plt.ylabel("Northings")

    # x label
    plt.xlabel("Eastings")

    # North scale bar
    plt.text(plot_buffer_bounds[0] + 800,
             plot_buffer_bounds[3] - 1000,
             "N",
             zorder=11)

    # Scale bar (set to 5km)
    plt.arrow(plot_buffer_bounds[0] + 3000,
              plot_buffer_bounds[1] + 1000,
              5000,
              0,
              zorder=11)

    # Arrow head
    plt.arrow(plot_buffer_bounds[0] + 1000,
              plot_buffer_bounds[3] - 3000,
              0,
              1000,
              head_width=200,
              zorder=15)

    # Scale "5km" label
    plt.text(plot_buffer_bounds[0] + 3000 + 2500,
             plot_buffer_bounds[1] + 1200,
             "5km",
             zorder=11)

    # User location
    plt.scatter(east,
                north,
                color="black",
                marker=11)

    # Plot the first node
    plt.scatter(start_node[0],
                start_node[1],
                color="red",
                marker="x",
                label="user position",
                zorder=11)

    # Nearest node to user
    plt.scatter(highest_east,
                highest_north,
                color="white",
                marker=11)

    # Highest point
    plt.scatter(finish_node[0],
                finish_node[1],
                color="green",
                marker="x",
                label="highest point",
                zorder=11)
    # Legend
    plt.legend(loc="upper right")

    plt.show()

    """"" ADDITIONAL FEATURES"""""

    # Loop to retrieve the lengths of every roadlink segment.
    lengths_of_shortest_path = []
    for i in range(len(path)):
        for link in road_links:
            if path[i] == road_links[link]["start"]:
                lengths_of_shortest_path.append(road_links[link]["length"])

    # Calculate all the road lengths summed together
    total_distance_travelled = sum(lengths_of_shortest_path)

    # Ask the user for their weight and height.
    if input("key \"y\" if you Would like to know how many calories you will burn? ") == "y":
        weight = int(input("how much do you weigh in kg? :"))
        height = int(input("How tall are you in meters? "))
        # Calculate the calories burnt
        calories_burnt_per_second = (0.35 * weight) + ((1.38889 ** 2) / height) * (0.029) * weight
        travel_time_s = 0.72 * total_distance_travelled / 3600
        calories_burnt = round(travel_time_s * calories_burnt_per_second)
        print("You will burn ", calories_burnt, "calories")
    else:
        print("I guess you will just get fat then...")

    # Create a list of strings to be written to the fie
    information_list = [
        "Distance travelled (km): ",
        str(round(total_distance_travelled) / 1000),
        "Length of journey (minutes): ",
        str(round(travel_time_s)),
        "Calories burnt: ",
        str(calories_burnt)]

    # Write the information output to a file.
    information_file = open("Information_about_your_journey.txt", "w")
    for line in information_list:
        # write line to output file
        information_file.write(line)
        information_file.write("\n")
    information_file.close()
