

# Isle of Wight flood plan 

## Scenario
Extreme flooding is expected on the Isle of Wight and the authority in charge
of planning the emergency response is advising everyone to proceed by foot to
the nearest high ground.
To support this process, the emergency response authority wants you to develop
a software to quickly advise people of the quickest route that they should take
to walk to the highest point of land within a 5km radius.

## This program works in the following steps:

1. Request user for coordinates 

2. Identify the highest point within 5km

3. Nearest Integrated Transport Network - Used to find the nearest transport node

4. The shortest path along the ITN is determined using Naismith’s rule.

5. Plots a background map 10km x 10km of the surrounding area - Gives the user an on screen display of their location. 

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
Clone this repo to your local machine using https://github.com/ThomasJames/Isle_of_Wight_Flood_Plan

## Contributers 
Jack Pearce 
Thomas James




