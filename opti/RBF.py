# Version 0.5
# improved def gen_model(): , def get_f():, and def k_fold(): 
        # used matrix broadcasting to get x matrix instead of nested forloop

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import matplotlib.pyplot as plt
import copy
#import lhsmdu #pip install lhsmdu # https://github.com/sahilm89/lhsmdu
from smt.sampling_methods import LHS # pip install smt # https://github.com/SMTorg/SMT
import os

# get sample values, x and y from dynamo
# depends on what we get, maybe need to convert
# y = [y1,...,yn], x = [(x11,...,x1n), ... , (xn1,...,xnn)]
# x is in m dimensions so x1n is comparing the eucledian distance between the POINT of sample 1 and sample n
# get beta = np.matmul(F_inv,y)
# apply f(x|c) over F, using gaussian rbf --> exp(-cr^2), where r = ||x-xi||
# check against y
# MSE to calculate y vs y_actual loss

#training c
#init algo:
    #init c 
    #run k-fold & get loss
    #repeat for 5 c's
#while not converged:
    #generate acq fn
    #sample at next c:
        #run k-fold

## initial rbf
c = np.random.rand()
def rbf(x):
    return np.e**(-c*(x**2))

## function to get beta values
def gen_model(x,y):
    # create F matrix
    # # F = []
    # # size = y.shape[0]
    # # for i in x:
    # #     for j in x:
    # #         F.append(np.linalg.norm(i-j)) # eucledian distance
    # # F = np.array(F).reshape(size,size) # reshape into numpy array
    F = np.linalg.norm((x[:,np.newaxis,:] - x), axis = 2) # use broadcasting instead
    F = rbf(F) # exp(-cr^2)
    
    #print(f"\nF = {F}")
    #print(f"\ndeterminant = {np.linalg.det(F)}")
    F_inv = np.linalg.pinv(F) #np.linalg.inv(F) use SVD to approximate initial matrix
    beta = np.matmul(F_inv,y) # beta = F_inv * y
    return beta
    
## Function to compare each new_x value to the old x value
def get_f(x, new_x):
    if len(new_x.shape) != len(x.shape):
        print("\n****** ERROR in get_f: dimensions of new_x does not match that of x ******")
        return x
    else:  
        # # f = []
        # # for i in x:
        # #     f.append(np.linalg.norm(i-new_x))
        f = np.linalg.norm(x-new_x,axis =1) # use broadcasting
        return rbf(f)#np.array(f))

## K_fold Cross validation Function    
def k_fold(x, y, c, k=5): # inputs: all x, y values; hyper-param c; k-fold
    def rbf(x): # regen rbf
        return np.e**(-c*(x**2))
    
    if len(x) != len(y):
        print("\n***** ERROR in k_fold: length of x =/= length of y *****")
    
    if len(x) < k:
        print(f"\n***** WARNING in k_fold: length of x: {len(x)} < k: {k} *****")
        k = len(x)
        
    # prepare for k-fold testing
    indexes = np.arange(len(x)) # create list of indexes 
    np.random.shuffle(indexes) # shuffle the indexs
    
    k_fold_indexes = [] # stores indexes 
    for i in range(k): k_fold_indexes.append([])
    
    k_count = 0
    for i in indexes:
        k_fold_indexes[k_count].append(i)
        if k_count >= (k-1):
            k_count = 0 
        else:     
            k_count += 1
    # k_fold_indexes now stores k sets which each set containing the index of the obs to be used
    
    k_fold_loss = []
    for iter in range(k): #repeat k times
        holdout_indx = k_fold_indexes[iter]           #set iter-th value to be left out fold
        therest = np.delete(k_fold_indexes, iter, axis = 0)  #rest to be in observation fold
        # merge the observation folds
        # therest_indx = []
        # for i in range(len(therest)):
        #     therest_indx += therest.tolist()[i]        
        therest_indx = np.concatenate(therest)
        
        y_val = [] # y-values for validation
        x_val = [] # x-values for validation
        y_train = []  # y-values for training
        x_train = []  # x-values for training
        
        for i in holdout_indx:
            y_val.append(y[i])
            x_val.append(x[i])
        
        for i in therest_indx:
            y_train.append(y[i])
            x_train.append(x[i])
        
        # convert to numpy array
        y_val = np.array(y_val)
        x_val = np.array(x_val)
        y_train = np.array(y_train)
        x_train = np.array(x_train)
        
        beta = gen_model(x_train,y_train) # generate beta
        pred_y = [] # get predictions
        for new_x in x_val:
            new_x = np.array([new_x])
            pred_y.append(np.matmul(np.transpose(beta), get_f(x_train, new_x)))
        
        k_fold_loss.append(sum((pred_y - y_val)**2)) #sum of squared loss
        #k_fold_loss.append(sum(abs(pred_y - y_val))) # calculate loss   
    loss = np.average(k_fold_loss) # average the loss value accross the k-folds
    return loss
    
   
## Optimise Hyperparameter
def optimise_c(x,y):     
    c_list = np.arange(-10,10.1,0.1)
    # get the respective loss values for the 
    loss_list = []
    for c in c_list:
        loss_list.append(k_fold(x, y, c, k=5))
    neg_loss = np.negative(loss_list)
    
    print('Best value: {:.5f} at c = {:.5f}' .format(np.max(neg_loss), c_list[np.argmax(neg_loss)]))
    return c_list[np.argmax(neg_loss)]

## LHS
# def LHSsample(dim, number):
#     return np.array(lhsmdu.sample(dim, number))
def LHSsample(xmin,xmax,number):
    # xmin: lowest value allowable for x. array([])
    # xmax: highest value allowable for x. array([])
    # number = number of samples
    xlimits = np.transpose(np.array([xmin,xmax]))
    
    # returns an array of samples in coordinate form
    return LHS(xlimits=xlimits)(number)
