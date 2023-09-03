import numpy as np
import scipy as sp


def create_spline_from_points(start_point, max_height, end_point, x_step_size):
    """
    This method will get a 3 points and step size and return location and velocity in each time stamp
    """
    # start and end point are in z=0
    start_point[2] = 0
    end_point[2] = 0

    # middle points is the middle between start and end with max_height
    middle_point = (np.array(end_point)-np.array(start_point)) / 2 + np.array(start_point)
    middle_point[2] = max_height

    #create list of x,y,z points
    x = [start_point[0], middle_point[0], end_point[0]]
    y = [start_point[1], middle_point[1], end_point[1]]
    z = [start_point[2], middle_point[2], end_point[2]]
    
    #create interpolate
    try:
        spline = sp.interpolate.Rbf(x,y,z,function='multiquadric')
    except:
        return []
    if (end_point[0] > start_point[0]):
        xi = np.arange(start_point[0], end_point[0], x_step_size).tolist()
        y_step_size = (end_point[1]-start_point[1]) * x_step_size / (end_point[0]-start_point[0])
    else:
        xi = np.arange(start_point[0], end_point[0], -x_step_size).tolist()
        y_step_size = (end_point[1]-start_point[1]) * -x_step_size / (end_point[0]-start_point[0])
    
    #y step size depand on x step size, we must have to same number of points
    if (start_point[1] == end_point[1]):
        yi = [start_point[1]]*len(xi)
    else:
        yi = np.arange(start_point[1], end_point[1], y_step_size).tolist()

    if (len(xi) != len(yi)):
        if (len(xi) == len(yi)+1):
            xi = xi[:-1]
        if (len(xi) == len(yi)-1):
            yi = yi[:-1]
        matan=1

    zi = spline(xi,yi)
    
    #create 3d points along the path
    path_points = [[xi[index],yi[index],zi[index]] for index in range(len(xi))]
    return path_points