

# Isle of Wight flood plan 

## Scenario
Extreme flooding is expected on the Isle of Wight and the authority in charge
of planning the emergency response is advising everyone to proceed by foot to
the nearest high ground.
To support this process, the emergency response authority wants you to develop
a software to quickly advise people of the quickest route that they should take
to walk to the highest point of land within a 5km radius.

<img src="https://github.com/ThomasJames/Isle_of_Wight_Flood_Plan/blob/master/Example%20of%20use.png" width="500">


## This program works in the following steps:

- Request user for coordinates. 

- Check the user is on land.

- Identify the highest point within 5km.

- Nearest Integrated Transport Network - Used to find the nearest transport node.

- The shortest path along the ITN is determined using Naismith’s rule.

- Plots a background map 10km x 10km of the surrounding area - Gives the user an on screen display of their location. 

## Additional Features:

- Calorie counter 
- GUI interface for data entry 

### Prerequisites

You must be running python 3.6
The following librairies will also need to be installed: 
```
pip install matplotlib
pip install numpy
pip install shapely
pip install geopandas
pip install json
pip install networx
pip intsall tkinter
pip install rasterio 
pip install rtree 
```
## Clone
Clone this repo to your local machine using ```git clone git@github.com:ThomasJames/Isle_of_Wight_Flood_Plan.git```

## Contributers 
- Jack Pearce: https://github.com/JackPearce
- Thomas James: https://github.com/ThomasJames
- Yufan Zhao: https://github.com/YuJaden




