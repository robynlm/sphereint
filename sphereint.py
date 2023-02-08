"""This module provides the SphereIntegrate class.

Copyright (C) 2022  Robyn L. Munoz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

You may contact the author at : robyn.munoz@yahoo.fr
"""

import numpy as np

class SphereIntegrate:
    """Class to compute the numerical volume in a sphere.
    
    This class provides a weight for each grid position based on whether or not 
    it is in (weight = 1), out (weight = 0), or partially in 
    (weight in between 0 and 1) a sphere of a given radius.
    
    A cubic cell is placed around each grid position and the volume of 
    the cell in the sphere (assuming a flat suface in the cell) 
    is calculated and normalised by the cell volume to obtain the weight.
    
    """
    def __init__(self, N, L, centre_location_coordinate):
        """Define grid and sphere center
        
        Parameters
            N : int, number of data points in each direction of the grid
            L : float, size of grid
                Here I assume the data box is a cude 
                that ranges from -L/2 to L/2.
            centre_location_coordinate : (3) list of integers
                                         Indexes of the grid position where
                                         the centre of the sphere is placed.
        """
        self.N = N
        self.L = L
        self.centre = centre_location_coordinate
        
        self.dx = L / N # grid spacing
        self.volume_cell = self.dx**3 # volume contained in one grid cell
    
    def get_box_weights(self, radius):
        """Compute weight at every grid position for given radius.
        
        Parameters : 
            radius : float, radius of sphere
        
        Returns :
            (N, N, N) array_like
        """
        if radius > self.L / 2:
            # Radius can't be bigger than data box.
            # Because I use periodic boundary conditions 
            # the sphere would fold on itself.
            print('ERROR: that radius is too big for me sorry')
        else:
            weight = np.zeros((self.N, self.N, self.N))
            grid = np.arange(-self.L/2, self.L/2, self.dx)
            for ix in range(self.N):
                x = grid[ix]
                for iy in range(self.N):
                    y = grid[iy]
                    for iz in range(self.N):
                        xyz = np.array([x, y, grid[iz]])
                        weight[ix, iy, iz] = self.cell_weight(xyz, radius)
            weight = self.shift(weight)
            return weight
    
    def cell_weight(self, xyz, radius):
        """Compute weight of given position for given radius.
        
        Parameters : 
            xyz : (3) array_like
                  Coordinate position of grid point.
            radius : float, radius of sphere
            
        Returns :
            float, weight value
        """
        # First check if cell in or outside of spheres
        cell_diag = self.dx * np.sqrt(3)
        cell_in_sphere = self.check_if_in_radius(xyz, 
                                                 radius - cell_diag)
        cell_outside_sphere = ~self.check_if_in_radius(xyz, 
                                                       radius + cell_diag)
        if cell_in_sphere:
            return 1.0
        elif cell_outside_sphere:
            return 0.0
        else:       
            # Next for sphere boundary contained in grid cell
            points = self.cell_corner_positions(xyz)
            points_in_sphere = [self.check_if_in_radius(pi, radius)
                                for pi in points]
            nbr_points_in_sphere = np.sum(points_in_sphere)
            if nbr_points_in_sphere > 0:
                volume = self.compute_volumes(radius, points, points_in_sphere)
                return volume / self.volume_cell
            else:
                return 0.0
    
    def shift(self, phi):
        """Shift values to be around wanted grid center. 
        
        Parameters :
            phi : (N, N, N) array_like
                  Value to be shifted.
        
        Returns :
            (N, N, N) array_like
        """
        xshift = int(self.N/2) - self.centre[0]
        yshift = int(self.N/2) - self.centre[1]
        zshift = int(self.N/2) - self.centre[2]
        phi = np.append(phi[xshift:, :, :], phi[:xshift, :, :], axis=0)
        phi = np.append(phi[:, yshift:, :], phi[:, :yshift, :], axis=1)
        phi = np.append(phi[:, :, zshift:], phi[:, :, :zshift], axis=2)
        return phi
    
    def check_if_in_radius(self, pos, radius):
        """Check if coordinate position is contained in given radius."""
        return np.sqrt(np.sum(pos**2)) <= radius
        
    def cell_corner_positions(self, xyz):
        """ Provide cell corner positions.
        
        A cubic cell of size dx is placed around the grid point,
        the coordinate position of the corners are provided here.
        
        Parameters :
            xyz : (3) array_like
                  Coordinate position of grid point.
            
        Returns :
            (8) list
            Each element is a (3) array_like coordinate position
        """
        dxmax = self.dx / 2
        x, y, z = xyz[0], xyz[1], xyz[2]
        return [np.array([x + dxmax, y + dxmax, z + dxmax]),
                np.array([x - dxmax, y + dxmax, z + dxmax]),
                np.array([x + dxmax, y - dxmax, z + dxmax]),
                np.array([x - dxmax, y - dxmax, z + dxmax]),
                np.array([x + dxmax, y - dxmax, z - dxmax]),
                np.array([x - dxmax, y - dxmax, z - dxmax]),
                np.array([x + dxmax, y + dxmax, z - dxmax]),
                np.array([x - dxmax, y + dxmax, z - dxmax])]
        
    def compute_volumes(self, radius, points, points_in_sphere):
        """Compute volume of the cell contained in the sphere.
        
        Parameters :
            radius : float, radius of sphere
            points : (8) list
                     Each element is a (3) array_like coordinate 
                     position of the cell corner.
            points_in_sphere : (8) list
                               Each element is a boolean:
                                - True : cell corner in sphere
                                - False : cell corner not in sphere
            
        Returns :
            float, volume of the cell contained in the sphere
        """
        points = np.array(points)
        # Identify the neighbouring corner point of each corner.
        # For example, the corner : points[0] (x + dx/2, y + dx/2, z + dx/2)
        # Has the neighbours: 
        # points[1] (x - dx/2, y + dx/2, z + dx/2), neighbour along x
        # points[2] (x + dx/2, y - dx/2, z + dx/2), neighbour along y
        # points[6] (x + dx/2, y + dx/2, z - dx/2), neighbour along z
        neighbouring_points = np.array([[points[1], points[2], points[6]], 
                                        [points[0], points[3], points[7]], 
                                        [points[0], points[3], points[4]], 
                                        [points[1], points[2], points[5]], 
                                        [points[2], points[5], points[6]], 
                                        [points[3], points[4], points[7]], 
                                        [points[0], points[4], points[7]], 
                                        [points[1], points[5], points[6]]])
        
        sphere_mask = np.where(points_in_sphere)
        not_sphere_mask = np.where(~np.array(points_in_sphere))
        nbr_points_in_sphere = np.sum(points_in_sphere)
        
        # Compute volume.
        # The radius, corner positions and neighbouring points (masked depending
        # on the sphere) are passed to the volume_*_points function 
        # according to the number of corners that are in the sphere.
        if nbr_points_in_sphere == 8:
            return self.volume_cell
        elif nbr_points_in_sphere == 7:
            volume = self.volume_1_point(radius, points[not_sphere_mask],
                                         neighbouring_points[not_sphere_mask])
            return self.volume_cell - volume
        elif nbr_points_in_sphere == 6:
            volume = self.volume_2_points(radius, points[not_sphere_mask], 
                                          neighbouring_points[not_sphere_mask])
            return self.volume_cell - volume
        elif nbr_points_in_sphere == 5:
            volume = self.volume_3_points(radius, points[not_sphere_mask], 
                                          neighbouring_points[not_sphere_mask])
            return self.volume_cell - volume
        elif nbr_points_in_sphere == 4:
            volume = self.volume_4_points(radius, points[sphere_mask], 
                                          neighbouring_points[sphere_mask])
            return volume
        elif nbr_points_in_sphere == 3:
            volume = self.volume_3_points(radius, points[sphere_mask],
                                          neighbouring_points[sphere_mask])
            return volume
        elif nbr_points_in_sphere == 2:
            volume = self.volume_2_points(radius, points[sphere_mask],
                                          neighbouring_points[sphere_mask])
            return volume
        elif nbr_points_in_sphere == 1:
            volume = self.volume_1_point(radius, points[sphere_mask], 
                                         neighbouring_points[sphere_mask])
            return volume
    
    def interpolate_radius_get_distance(self, radius, point, neighbourpoint):
        """Compute distance between cell corner and sphere boundary on cell edge.
        
        1) Compute intersection between sphere and direction passing 
        the cell corner in the sphere and its' neighbour 
        that is outside the sphere. 
        Each neighbouring point share two coordinate values with 
        the cell corner in the sphere and the third coordinate 
        is + or - dx different. 
        The intersecting point shares the first two coordinate values, 
        but the last one needs to be computed.
        
        2) The intersecting point will then have the 
        Then compute the distance between the cell corner 
        and the intersecting point.
        
        Parameters :
            radius : float, radius of sphere
            points : (3) array_like 
                     Coordinate position of the cell corner.
            neighbourpoint : (?) list
                             Each element is a (3) array_like coordinate 
                             position of the cell corner neighbours 
                             that are outside of the sphere, 
                             dimension can go from 1 to 3.
                             
        Returns :
            list : depths, ixyz
                   depths : (?) array_like
                            Distances between the cell corner in the sphere
                            and the point intersecting the sphere 
                            and the neighbouring edge, 
                            dimension can go from 1 to 3.
                   ixyz : (?) list
                          Direction along which intersecting point lies, 
                          with 0 -> x, 1 -> y, 2 -> z, 
                          dimension can go from 1 to 3.
        """
        distances_to_point = []
        ixyz = []
        for p in neighbourpoint:
            # Two of the coordinates are the same
            interpolated_point = p.copy()
            
            # Index of coordinate that needs to change
            ctochange = np.where(point - p != 0)
            ixyz += [ctochange]
            
            # Sign of new coordinate
            direction = np.sign(point[ctochange])[0]
            if int(direction) == 0:
                direction = np.sign(p[ctochange])[0]
            
            # New coordinate
            new_coord = (direction * np.sqrt( radius**2 - np.sum(point**2) 
                                             + point[ctochange]**2 ))
            interpolated_point[ctochange] = new_coord
            
            # Distance between interpolated point and cell corner in sphere
            d_to_point = np.sqrt(np.sum( (point - interpolated_point)**2 ))
            distances_to_point += [d_to_point]
            
            # Check that distance is smaller than cell size
            if (d_to_point - self.dx)/self.dx > 1e-14:
                print('WARNING: (depth - dx) / dx > 1e-14')
        return np.array(distances_to_point), ixyz
    
    def volume_1_point(self, radius, point, pointneighbour):  
        """Compute cell volume in the sphere when 1 corner is in the sphere.
        
        Compute the volume of a trirectangular tetrahedron.
        
        Parameters :
            radius : float, radius of sphere
            point : (3) array_like 
                    Coordinate position of the cell corner.
            neighbourpoint : (3) array_like
                             Each element is a (3) array_like coordinate 
                             position of the cell corner neighbours.  
            
        Returns :
            float, volume            
        """
        depth, ixyz = self.interpolate_radius_get_distance(radius, point[0],
                                                           pointneighbour[0])
        volume = np.prod(depth) / 6
        volume_max = self.volume_cell / 6
        if (volume - volume_max)/volume_max > 1e-14:
            print('WARNING: Volume 1 point too big')
        return volume
        
        
    def volume_2_points(self, radius, points, pointneighbours):  
        """Compute cell volume in the sphere when 2 corners are in the sphere.
        
        Compute the volume of a trirectangular tetrahedron 
        that extends larger than the cell size in one direction. 
        Then remove that extension that correspondes to 
        a smaller trirectangular tetrahedron
        such that we only consider the part in the cell.
        To find the side that needs to be extended, 
        the area of each triangular base, the smallest one is extended.
        If the two areas are equal then we have a right triangular prism.
        
        Parameters :
            radius : float, radius of sphere
            point : (2, 3) array_like 
                    Coordinate positions of the 2 cell corners.
            neighbourpoint : (2, 3) array_like
                             Coordinate positions of the cell corner neighbours.  
            
        Returns :
            float, volume     
        """
        # Cell corners in the sphere
        point1 = points[0]
        point2 = points[1]
        
        # Neighbouring points that are outside the sphere
        pointneighbour1 = [neighbour 
                           for neighbour in pointneighbours[0] 
                           if list(neighbour)!=list(point2)]
        pointneighbour2 = [neighbour 
                           for neighbour in pointneighbours[1] 
                           if list(neighbour)!=list(point1)]
        
        # Distances between corner cell in sphere and sphere boundary.
        depth1, ixyz1 = self.interpolate_radius_get_distance(radius, point1,
                                                             pointneighbour1)
        depth2, ixyz2 = self.interpolate_radius_get_distance(radius, point2,
                                                             pointneighbour2)
        # Area of each base
        area1 = np.prod(depth1) / 2
        area2 = np.prod(depth2) / 2
        
        if abs(area1 / area2 - 1)>1e-12: 
            # Base triangles different so trirectangular tetrahedron considered.
            # To find this: volume = big_tetrahedron - small_tetrahedron
            #                      = entire_shape - extended_part
            # I'm making sure the depth/width of the triangles overlap
            # Create dict with {coord_direction: depth_val}
            dict1 = {ixyz1[i][0][0]:depth1[i] 
                     for i in range(len(ixyz1))}
            dict2 = {ixyz2[i][0][0]:depth2[i] 
                     for i in range(len(ixyz2))}
            idict = list(dict1.keys())
            if area1 > area2: # Point1 has the bigger base
                big_width = dict1[idict[0]]
                big_depth = dict1[idict[1]]
                small_width = dict2[idict[0]]
                small_depth = dict2[idict[1]]
            else: # Point2 has the bigger base
                big_width = dict2[idict[0]]
                big_depth = dict2[idict[1]]
                small_width = dict1[idict[0]]
                small_depth = dict1[idict[1]]
                
            extended_height = (small_width * self.dx 
                               / ( big_width - small_width ))
            
            vol_big = ( (self.dx + extended_height) 
                       * big_width * big_depth) / 6
            vol_small = ( extended_height * small_width * small_depth) / 6
            volume = vol_big - vol_small
        else:
            # This is a right triangular prism
            volume = self.dx * np.average([area1, area2])
            
        volume_max = self.volume_cell / 2
        if (volume - volume_max)/volume_max > 1e-14:
            print('WARNING: Volume 2 points too big')
        return volume
    
    def volume_3_points(self, radius, points, pointneighbours):
        """Compute cell volume in the sphere when 3 corners are in the sphere.
        
        Compute the volume of a trirectangular tetrahedron 
        that extends larger than the cell size in two directions. 
        Then remove those extensions that correspond to 
        smaller trirectangular tetrahedrons
        such that we only consider the part in the cell.
        
        Parameters :
            radius : float, radius of sphere
            point : (3, 3) array_like 
                    Coordinate positions of the 3 cell corners.
            neighbourpoint : (3, 3) array_like
                             Coordinate positions of the cell corner neighbours.  
            
        Returns :
            float, volume     
        """
        # Cell corners in the sphere
        point1 = points[0]
        point2 = points[1]
        point3 = points[2]
        
        # Neighbouring points that are outside the sphere
        pointneighbour1 = [neighbour 
                           for neighbour in pointneighbours[0] 
                           if (list(neighbour)!=list(point2) 
                               and list(neighbour)!=list(point3))]
        pointneighbour2 = [neighbour 
                           for neighbour in pointneighbours[1] 
                           if (list(neighbour)!=list(point1) 
                               and list(neighbour)!=list(point3))]
        pointneighbour3 = [neighbour 
                           for neighbour in pointneighbours[2] 
                           if (list(neighbour)!=list(point1) 
                               and list(neighbour)!=list(point2))]
        
        # Distances between corner cell in sphere and sphere boundary.
        depth1, ixyz1 = self.interpolate_radius_get_distance(radius, point1, 
                                                             pointneighbour1)
        depth2, ixyz2 = self.interpolate_radius_get_distance(radius, point2, 
                                                             pointneighbour2)
        depth3, ixyz3 = self.interpolate_radius_get_distance(radius, point3, 
                                                             pointneighbour3)
        depth = [depth1, depth2, depth3]
        ixyz = [ixyz1, ixyz2, ixyz3]
        
        # ID points
        # Point with only one depth value is A
        iA = np.where(np.array(list(map(len, depth)))==1)[0][0]
        ixyzA = ixyz[iA]
        depthA = depth[iA]
        # The two other points are B and C 
        # they will each have a tetrahedron extended through their side.
        iB, iC = np.delete(np.arange(3), iA)
        ixyzB = ixyz[iB]
        ixyzC = ixyz[iC]
        depthB = depth[iB]
        depthC = depth[iC]
            
        # ID depths
        A_width = depthA
        # {coord_direction: depth_val}
        dictB = {ixyzB[i][0][0]:depthB[i] for i in range(len(ixyzB))}
        dictC = {ixyzC[i][0][0]:depthC[i] for i in range(len(ixyzC))}
        # All corners have a width
        key_width = list(dictB.keys() & dictC.keys())[0]
        B_width = dictB[key_width]
        C_width = dictC[key_width]
        # Remaining coordinates correspond to depth and height
        B_depth = [dictB[key] for key in dictB if key!=key_width][0]
        C_height = [dictC[key] for key in dictC if key!=key_width][0]
        
        # Compute extended part
        B_height = B_width * self.dx / (A_width - B_width)
        C_depth = C_width * self.dx / (A_width - C_width)
            
        # Compute tetrahedron volume
        vol_B = B_height * B_width * B_depth / 6
        vol_C = C_height * C_width * C_depth / 6
        vol_big = (self.dx + B_height) * A_width * (self.dx + C_depth) / 6
        volume = vol_big - vol_B - vol_C
        
        volume_max = self.volume_cell * 4 / 6
        if (volume - volume_max)/volume_max > 1e-14:
            print('WARNING: Volume 3 points too big')
        return volume
    
    def volume_4_points(self, radius, points, pointneighbours):
        """Compute cell volume in the sphere when 4 corners are in the sphere.
        
        Two cases :
            
        1 ) A truncated right square prism
        
        2 ) Compute the volume of a trirectangular tetrahedron 
            that extends larger than the cell size in three directions. 
            Then remove those extensions that correspond to 
            smaller trirectangular tetrahedrons
            such that we only consider the part in the cell.
        
        Parameters :
            radius : float, radius of sphere
            point : (4, 3) array_like 
                    Coordinate positions of the 3 cell corners.
            neighbourpoint : (4, 3) array_like
                             Coordinate positions of the cell corner neighbours.  
            
        Returns :
            float, volume     
        """
        # Cell corners in the sphere
        point1 = points[0]
        point2 = points[1]
        point3 = points[2]
        point4 = points[3]
        
        # Neighbouring points that are outside the sphere
        pointneighbour1 = [neighbour 
                           for neighbour in pointneighbours[0] 
                           if (list(neighbour)!=list(point2) 
                               and list(neighbour)!=list(point3) 
                               and list(neighbour)!=list(point4))]
        pointneighbour2 = [neighbour 
                           for neighbour in pointneighbours[1] 
                           if (list(neighbour)!=list(point1) 
                               and list(neighbour)!=list(point3) 
                               and list(neighbour)!=list(point4))]
        pointneighbour3 = [neighbour 
                           for neighbour in pointneighbours[2] 
                           if (list(neighbour)!=list(point1) 
                               and list(neighbour)!=list(point2) 
                               and list(neighbour)!=list(point4))]
        pointneighbour4 = [neighbour 
                           for neighbour in pointneighbours[3] 
                           if (list(neighbour)!=list(point1) 
                               and list(neighbour)!=list(point2) 
                               and list(neighbour)!=list(point3))]
        
        # Distances between corner cell in sphere and sphere boundary.
        depth1, ixyz1 = self.interpolate_radius_get_distance(radius, point1, 
                                                             pointneighbour1)
        depth2, ixyz2 = self.interpolate_radius_get_distance(radius, point2, 
                                                             pointneighbour2)
        depth3, ixyz3 = self.interpolate_radius_get_distance(radius, point3, 
                                                             pointneighbour3)
        depth4, ixyz4 = self.interpolate_radius_get_distance(radius, point4, 
                                                             pointneighbour4)
        depth = [depth1, depth2, depth3, depth4]
        
        if sum(map(len, depth)) == 4:
            # A truncated right square prism
            volume = ( np.max(depth) + np.min(depth) ) * self.dx * self.dx / 2
            volume_max = self.volume_cell
            if (volume - volume_max)/volume_max > 1e-14:
                print('WARNING: Volume 4 points, 1st case, too big')
        else: 
            # A trirectangular tetrahedron
            # ID points
            # Point whose neighbours are all in the sphere is A
            iA = np.where(np.array(list(map(len, depth)))==0)[0][0]
            # The two other points are B, C and D
            # they will each have a tetrahedron extended through their side.
            ixyz = [ixyz1, ixyz2, ixyz3, ixyz4]
            iB, iC, iD = np.delete(np.arange(4), iA)
            ixyzB = ixyz[iB]
            ixyzC = ixyz[iC]
            ixyzD = ixyz[iD]
            depthB = depth[iB]
            depthC = depth[iC]
            depthD = depth[iD]
            
            # ID depths
            # {coord_direction: depth_val}
            dictB = {ixyzB[i][0][0]:depthB[i] for i in range(len(ixyzB))}
            dictC = {ixyzC[i][0][0]:depthC[i] for i in range(len(ixyzC))}
            dictD = {ixyzD[i][0][0]:depthD[i] for i in range(len(ixyzD))}
            # ID the depths
            key_width = list(dictB.keys() & dictC.keys())[0]
            B_width = dictB[key_width] 
            C_width = dictC[key_width] 
            key_depth = list(dictB.keys() & dictD.keys())[0]
            B_depth = dictB[key_depth] 
            D_depth = dictD[key_depth] 
            key_height = list(dictC.keys() & dictD.keys())[0]
            C_height = dictC[key_height] 
            D_height = dictD[key_height] 
            
            # Compute extended part
            B_height = B_depth * (self.dx - C_height) / (self.dx - B_depth)
            C_depth = C_width * (self.dx - D_depth) / (self.dx - C_width)
            D_width = D_height * (self.dx - B_width) / (self.dx - D_height)
            
            # Compute tetrahedron volume
            vol_big = ((self.dx + B_height)
                       * (self.dx + C_depth)
                       * (self.dx + D_width)) / 6
            vol_B = B_height * B_depth * B_width / 6
            vol_C = C_height * C_depth * C_width / 6
            vol_D = D_height * D_depth * D_width / 6
            volume = vol_big - vol_B - vol_C - vol_D
            
            volume_max = self.volume_cell
            if (volume - volume_max)/volume_max > 1e-14:
                print('WARNING: Volume 4 points too big')
        return volume
    