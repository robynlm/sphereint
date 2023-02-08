# sphereint
Class to compute the numerical volume in a sphere.
    
This class provides a weight for each grid position based on whether or not it is in (weight = 1), out (weight = 0), or partially in (weight in between 0 and 1) a sphere of a given radius.

A cubic cell is placed around each grid position and the volume of the cell in the sphere (assuming a flat suface in the cell) is calculated and normalised by the cell volume to obtain the weight.
