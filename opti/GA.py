#Version 2.0 
#Removed Linear Model
#Added RBF Model
#Added 1D graph testing
    #Version 2.1 
    #Modified Metadata to include past samples (affected functions: def model()
    #Added description to functions
    #Bugfix in def model()
        #expected_parameters = x_GT.shape[1]  instead of beta,shape[1]
    #Version 2.1a
    #Changed initial "best_growth_score" in def RunGA(): to -inf

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import random as rand
import copy
import math as m

from sklearn.model_selection import train_test_split
from scipy.stats import norm
import lhsmdu #pip install lhsmdu # https://github.com/sahilm89/lhsmdu
import os

os.chdir("D:/School Stuff/01 .400 Capstone/GA & Opti")
import RBF as RBF

## INITIALIZATION   
def create_starting_population(size, dict):    
    #this just creates a population of different params combinations of a fixed size. 
    #here we unpack the dictionary to see how long our gene needs to be, para_length is a list of the lengths of the parameter lists
    para_length = []
    for i in dict.keys():
        para_length.append(len(dict[i]))
        
    #Find out how many bits is required to encode each parameter (size of each allele) eg. [4,4]
    code_length = []
    for i in para_length:
        code_length.append(m.ceil(m.log(i,2)))    
        
    #we initalize a random population
    population = []    
    for i in range(0,size):
        population.append(create_new_member(para_length, code_length))
        
    return population, code_length, para_length
#      
def create_new_member(para_length, code_length):  
    #takes in a list which contains information about how long is the list for each parameter eg. [9,9]
    #returns a list of a gene, which encodes the information for each allele (parameter) into a long list eg. [0, 1, 1, 0, 1, 0, 0, 0]
    
    #generates 1 child
    member = []
    for i in range(0,len(para_length)):
        #random int between 1-length
        uncoded_rand_number = m.floor(rand.uniform(0,para_length[i]))
        #convert this number to a binary list
        member = member + encode_binary(uncoded_rand_number, code_length[i])
    
    return member
#
def encode_binary(number, length):
    #takes in an number to be converted (int) and the size of string (int) (eg, 3,8)
    #returns a list of converted number in binary [0, 0, 0, 0, 0, 0, 1, 1]
    binary_list = [int(x) for x in bin(number)[2:]]
    if length > len(binary_list):
        temp = [0]* (length-len(binary_list))
        binary_list = temp+binary_list
    return binary_list
    
    member.append(round(rand.uniform(0,1)))  
    return list  

## Scoring

def score_population(population, metadata):
    #scores the individual members of the population based on the fitness
    #population = an array of genes
    #metadata = wrapped information to be passed along
    scores = []
    
    for i in range(0, len(population)):
        scores += [fitness(population[i], metadata)]
    
    return scores
    
def fitness(member, metadata):
    #Decode the gene and run parameters into model to get score
    #member = individual gene (eg. [1,1,0,0]) from the population
    #metadata = wrapped information to be passed along
    #Debug line: #print("Parsing Member: {}".format(member))
    score = 0
    values = []
    
    #Unpack Metadata
    code_length = metadata[0]
    para_length = metadata[1]
    dict = metadata[2]
    dict_keys = metadata[3]
    modeldata = metadata[4]
    
    #Decode Gene to the numbers here 
    values,errorflag = decode(dict, dict_keys, code_length, member)
    
    if errorflag == 0:
        #Calculate score here
        score = model(modeldata, member, code_length, values)

    return score
    
def decode(dict, dict_keys, code_length, member):
    #Takes in dict, dict_keys, code length and list of binary, returns respective values from dictionary
    #(eg. gene [0, 1, 0, 1, 0, 1, 1, 1] with code_length [4,4] -> [5, 7])
    #(eg. gene [0, 1, 0, 1, 0, 1, 1, 1] with code_length [3,4,1] -> [2, 11, 1])
    start_pointer = 0
    list_of_decoded_numbers = []
    values = []
    errorflag = 1
    
    for i in code_length:
        #iterate through the alleles
        binary_to_be_decoded=[] #temp array to store allele information
        for j in range(start_pointer,start_pointer+i):
            #add number into temp array
            binary_to_be_decoded.append(member[j])
        #change start position (for next allele to be read)
        start_pointer = start_pointer+i
        #Convert "binary_to_be_decoded" into a string and then make the binary to decimal conversion
        list_of_decoded_numbers.append(int("".join(map(str,binary_to_be_decoded)), base=2))
    
    #Check if number of parameters match the lists stored in the dictionary
    if len(dict_keys) != len(list_of_decoded_numbers):
        print("!!! \n ************************ WARNING ************************ \n Parameter number mismatch! \n Parsing {} parameters but dictionary only has {} lists \n!!!".format(len(list_of_decoded_numbers),dict.keys()))
    
    else:
        #Extract values from parameter list
        for i in range(0,len(list_of_decoded_numbers)):
            try:
                values.append(dict[dict_keys[i]][list_of_decoded_numbers[i]]) 
                errorflag = 0
            except:
                #print("Child's Parameters exceeded defined range, culling sample.")
                errorflag = 2
        
        
    return values, errorflag
#
# 
def pick_mate(scores):
    #not sure what this does, probably pick a good mate or something?
    #refer to here: https://github.com/gmichaelson/GA_in_python/blob/master/GA%20example.ipynb
    array = np.array(scores)
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))

    fitness = [len(ranks) - x for x in ranks]
    
    cum_scores = copy.deepcopy(fitness)
    
    for i in range(1,len(cum_scores)):
        cum_scores[i] = fitness[i] + cum_scores[i-1]
        
    probs = [x / cum_scores[-1] for x in cum_scores]
    
    randomnum = rand.random()
    
    for i in range(0, len(probs)):
        if randomnum < probs[i]:
            
            return i
#
#
def crossover(a,b):
    
    if len(a) != len(b):
        print("!!! \n ************************ WARNING ************************ \n Mating Partner mismatch! Length of partners are different! \n Attempted to mate {} with {} \n!!!".format(a,b))
        return a,b
    
    new_a = []
    new_b = []
    cut = m.floor(rand.uniform(0,len(a))) #crossover point
    
    new_a1 = copy.deepcopy(a[0:cut])
    new_a2 = copy.deepcopy(b[cut:])
    
    new_b1 = copy.deepcopy(b[0:cut])
    new_b2 = copy.deepcopy(a[cut:])
    
    new_a = new_a1 + new_a2
    new_b = new_b1 + new_b2
    
    if len(new_a) != len(new_b):
        print("!!! \n ************************ WARNING ************************ \n Children length mismatch! Length of Children are different! \n Children {} and {} \n!!!".format(new_a,new_b))
        return a,b
       
    return new_a, new_b
#
#   
def mutate (member, probability):
    output = copy.deepcopy(member)
    for i in range(0,len(member)):
        if rand.random() < probability:
            output[i] = int(member[i] == 0)
    return output
#
#    
    
def RunGA(list, dict, number_of_iterations, modeldata):
    # list = list of parameter values
    # dict = dictionary mapping parameter values to the associate parameter
    # number_of_iteration = number of GA cycles to run
    # modeldata = tuple containing beta values and past x-values in the form (beta, x_GT)
    population_size = 30
    number_of_couples = 9
    number_of_winners_to_keep = 2
    mutation_probability = 0.05
    number_of_groups = 1
    
    best_growth_score = -1e100#0
    best_growth_params = ['Error']
    plots = []
    # create the starting population
    population, code_length, para_length = create_starting_population(population_size, dict)
    
    metadata = (code_length, para_length, dict, list, modeldata)
 
    # run the iterations
    for i in range(0,number_of_iterations):
        new_population = []
        
        # evaluate the fitness of the current population
        scores = score_population(population, metadata)
        best_member = population[np.argmax(scores)] #best member of the population
        growth_score = max(scores) #best score
        
        if growth_score > best_growth_score:
            best_growth_score = growth_score
            best_growth_params, errorflag = decode(dict, list, code_length, best_member)
        #print('Iteration {}: Best growth so far is {}. Using {} of {}'.format(i, best_growth_score, best_growth_params,list))

        
        # allow members of the population to breed based on their relative score; 
            # i.e., if their score is higher they're more likely to breed
        for j in range(0, number_of_couples):  
            new_1, new_2 = crossover(population[pick_mate(scores)], population[pick_mate(scores)])
            new_population = new_population + [new_1, new_2]
  
        # mutate (maybe problem with deepcopy?)
        for j in range(0, len(new_population)):
            new_population[j] = copy.deepcopy(mutate(new_population[j], 0.05))
        
         
        # keep members of previous generation
        new_population += [population[np.argmax(scores)]]
        for j in range(1, number_of_winners_to_keep):
            keeper = pick_mate(scores)            
            new_population += [population[keeper]]
            
        # add new random members
        while len(new_population) < population_size:
            new_population += [create_new_member(para_length,code_length)]
            
        #replace the old population with a real copy
        population = copy.deepcopy(new_population)
        plots.append(best_growth_score)
    
    print("\n \n ************************************\n GA is Finished.")
    print("Growth Score of {}\n With Parameters {} from {}".format(best_growth_score, best_growth_params, list))
   
    #plt.plot(plots)
    #plt.show()
    
    return (best_growth_score, best_growth_params) # returns the best predicted score and assoscated x value
    
    
## Use RBF Model
def model(modeldata, member, code_length, values):
    #takes in a list of values for parameters and returns the score (expected growth score)
    #modeldata = tuple containing (beta, past_x_values)
    #member = which member of the population we are testing now
    #code_length = length of the gene
    #values = the decoded gene ie the x value(s) to evaluate/predict
    
    beta = modeldata[0]
    x_GT = modeldata[1]
    
    
    #Change this number to reflect numb of parameters the model requires
    expected_parameters = x_GT.shape[1]
    score = 0
    
    if(len(values) < expected_parameters):
        return score
    
    #Check if okay to parse to model
    if expected_parameters != len(values):
        print("!!! \n ************************ WARNING ************************ \n Member {} has more values than input model. \n Model has {} variables while gene has {}. \n Check the code_length input, the current code_length is {} \n Values are {}. \n!!!".format(member,expected_parameters,len(values),code_length,values))
       
    else:
        new_x = np.array(values).reshape(1,-1)
        score = np.matmul(np.transpose(beta.reshape(-1)), RBF.get_f(x_GT, new_x))
        #if x_GT is no longer used outside the GA code, then need to modify metadata to include all previous x values so that we can use it here to form the matrix
           
    return score
    



