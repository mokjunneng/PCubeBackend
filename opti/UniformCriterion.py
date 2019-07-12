# UC Version 3.0
# Changed permlist such that it uses LHS to select 10000 points to compare rather than every single possible point
# removed def permu():
# added dependency on RBF
# used matrix broadcasting to get eucledian dist instead of nested for loopp
    

import numpy as np
import copy
import itertools
import RBF

# Uniform Criteron Thing
def unicri(x_vals, xmin, xmax, step_sizes):
    # x_vals: array of past x values (parameter). array([[x11,x12,..,x1n],[x21,x22,..,x2n],...,[]])
    # xmin: lowest value allowable for x. array([])
    # xmax: highest value allowable for x. array([])
    # step_sizes: array of step sizes corresponding to the parameters. array([])
    
    ucsamples = RBF.LHSsample(xmin, xmax, 10**4) # Use LHS to get samples in range to compare
    
    dist = np.linalg.norm((ucsamples[:,np.newaxis,:] - x_vals), axis = 2) # matrix broadcast version of eucl dist
    mindist = np.min(dist, axis=1) # new point, assign the distance to the closest x_vals to be the mindist
    unicri_x = ucsamples[np.argmax(mindist)] # select the point with the greatest assigned distance
    
    #returns the coordinates of the point
    #currently in the form array([x1,x2,..,xn])
    return unicri_x
    
#olduc_time 1.791870355606079
#newuc_time 0.04682469367980957

  
    