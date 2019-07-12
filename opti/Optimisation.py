# version 1.0 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm #https://matplotlib.org/3.1.0/gallery/color/colormap_reference.html
from mpl_toolkits.mplot3d import axes3d #for 3D plotting
import copy
import os
import uuid

os.chdir("D:/School Stuff/01 .400 Capstone/GA & Opti")
import RBF as RBF #v0.5
import GA as GA #v2.1a
import UniformCriterion as UC #v3.0
import AWS as AWS #v1.0

## MISC (How to used the imported RBF, GA and UC)
# 1a) Sampling
# Use RBF.LHSsample to get samples
# # Parameters needed:
# # xmin - what are the minimum bounds | 1D: [-1]; 2D: [-5, -5]
# # xmax - what are the maximum bounds | 1D: [1.6]; 2D: [5, 5]
# # number - how many samples do you want
# ### Samples = RBF.LHSsample(dimensions,number)
# ### 1D: Samples = array[[0.1, 0.8, 0.3]] 
# ### 2D: Samples = array([[0.2, 0.7, 0.3],  << 1st parameter 3 samples
#                          [0.9, 0.6, 0.2]]) << 2nd parameter 3 samples
# this assumes a range of 0-1, so if we are using a larger range, we will need to map the results accordingly
# do this by MULTIPLYING by the range (max-min) and adding the min 

# 1b) Import from AWS

# 2) Prepare Samples 
# x-values should be in the form 
# array([[x11,x12,..,x1n],  <- 1st parameter, all x1 to xn values
#        [x21,x22,..,x2n],
#    ...,[xi1,xi2,..,xin]]) <- ith parameter, all x1 to xn values
# this is different from the LHS format!

# 3) Optimising hyperparameter
# use RBF.optimise_c to obtain the appropriate hyperparameter to use. 
# IMPT: YOU MUST REDEFINE THE RBF WITH THE NEW C
# # Parameters needed:
# # x_GT - the past x-values
# # y_GT - the past (observed) y-values
# ### c = RBF.optimise_c(x_GT,y_GT)
# ### def rbf(x):
# ###    return np.e**(-c*(x**2))

# 4) Acquiring Model
# use RBF.gen_model to get the beta values to be used for the model input to GA
# # Parameters needed:
# # x_GT - the past x-values
# # y_GT - the past (observed) y-values
# ### beta = RBF.gen_model(x_GT,y_GT)
# ### beta = array([[b1],[b2],...,[bi]) where i is number of data points

# 5) GA
# Use GA.RunGA to get next predicted value and x-value
# # Parameters needed:
# # parameters_list - a list of parameters (in string form) 
# # parameters_dict - a dictionary of the parameters and their associate values they can take
# # number_of_iterations - how many instances of GA to run (the best will be used)
# # modeldata - a tuple/array containing the beta values from RBF and past x-values 
# ###(pred_y, new_x) = GA.RunGA(parameters_list, parameters_dict, number_of_iterations, (beta, x_GT))
# ### pred_y = yn
# ### new_x = [x1,x2,...,xn]

# 6) UC
# Use UC.unicri to get the x-value via the uniform criterion
# # Parameters neded:
# # x_vals: array of past x values (parameter). array([[],[],...,[]])
# # xmin: lowest value allowable for x. array([])
# # xmax: highest value allowable for x. array([])
# # step_sizes: array of step sizes corresponding to the parameters. array([])
# ### new_x = UC.unicri(x_GT,xmin,xmax,step_sizes)
# ### new_x = [x1,x2,...,xn]


##  INITIALISATION
def initialisation(fn_min, fn_max, step_sizes, parameters_list, initial, experiment_id, exp_dur):
    # fn_min = list of min x vals for parameter eg. [1] for 1D or [1,0] for 2D   #from user **MOK 
    # fn_max = list of max x vals for parameter eg. [24] #from user **MOK  
    # step_sizes = list of step sizes for the parameter eg. [0.5]  #from user **MOK
    # parameters_list = from user as list eg. ['Daylight_Hours'] #List of parameter names **MOK 
    # initial = number of inital sample points (at least 2n+1 for n parameters) eg. 5 #from user **MOK
    # experiment_id = name of experiment as string #from user **MOK
    # exp_dur = duration of experiment in hours as float #from user **MOK
    
    # Stuff for GA
    table_name='LettuceGrowth'
    parameters_dict = {}
    for i in range(len(parameters_list)): #create using above information
        parameters_dict[parameters_list[i]] = np.arange(fn_min[i], fn_max[i]+step_sizes[i], step_sizes[i])
        #Dictionary containing the (min/max)  
    
    # initialise samples
    lhssample = RBF.LHSsample(fn_min, fn_max, initial) #draw samples using Latin Hypercube Sampling
    
    # # # #round to the appropriate dp (see step size)
    x = lhssample #but round to appropirate dp **MOK
    #x=np.array([[14.0],[5.0],[22.0],[24.0],[18.0]]) for 1D
    #x=np.array([[ 6.5, 24.5], [ 9.0, 20.0], [ 5.5, 14.5], [ 9.0, 30], [ 5.0, 17.0]]) for 2D
    # add samples to Dynamodb
    recipe_uuids = []
    for item in lhssample:
        recipe_uuid = uuid.uuid4()
        recipe_uuids.append(recipe_uuid)
        AWS.add_optimisation_recipe(recipe_uuid, parameters_list, item, experiment_id, exp_dur, type="init")
    return recipe_uuids
 
## Add start time
def start_experiment(experiment_id, uuid, timestamp):
    # experiment_id #from user **MOK
    # uuid #from user **MOK
    # timestamp #from user **MOK
    AWS.addtime(experiment_id, uuid, timestamp)  
    
## Add growth score
def update_growth(experiment_id, uuid, growthscore):
    # experiment_id #from user **MOK
    # uuid #from user **MOK
    # growthscore #from user **MOK
    AWS.update_growth(experiment_id, uuid, growthscore)  


## Optimisation
def do_optimisation(experiment_id, parameters_list, parameters_dict):
    # experiment_id 
    # parameters_list
    # parameters_dict
    # **MOK need some way to load these 3 params here
    
    x_GT, y_GT = AWS.getparams(experiment_id, table_name) #get values from AWS
    number_of_iterations = 100
    round = 0    
    # # RBF
    #reacquire hyperparameter
    c = RBF.optimise_c(x_GT,y_GT)
    #redefine rbf
    def rbf(x):
        return np.e**(-c*(x**2))
    #get model    
    beta = RBF.gen_model(x_GT,y_GT) # using actual values
    
    # # GA
    #RunGA with list of parameters, dictionary of parameters, number of iterations and , beta + GT values
    (pred_y, new_x) = GA.RunGA(parameters_list, parameters_dict, number_of_iterations, (beta, x_GT))
        # pred_y = predicted growth score (float)
        # new_x = list of parameters
    print(f"Optimal point at parameters {new_x} for {parameters_list}. Est. growth = {pred_y}")
    
    #generate uuid **MOK
    AWS.add_optimisation_recipe(uuid, parameters_list, new_x, experiment_id, type='optim')
    return (pred_y, new_x) #to display to user **MOK

## Exploration
def do_exploration(experiment_id, parameters_list, parameters_dict, fn_min, fn_max, step_sizes):    
    # experiment_id 
    # parameters_list
    # parameters_dict
    # fn_min
    # fn_max
    # step_sizes
    # **MOK need some way to load these 6 params here
    
    uc_x = UC.unicri(x_GT,np.array(fn_min),np.array(fn_max),np.array(step_sizes))
    # round uc_x to appropriate dp (based on step_sizes) #**MOK
    print(f"Exploration point at parameters {uc_x} for {parameters_list}.")
    
    #generate uuid **MOK
    AWS.add_optimisation_recipe(uuid, parameters_list, new_x, experiment_id, type='explore',)
    return uc_x #to display to user **MOK


