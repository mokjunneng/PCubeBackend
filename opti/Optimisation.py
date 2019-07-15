# Version 1.0

import numpy as np
import time
from datetime import datetime
from decimal import Decimal
import boto3
from boto3.dynamodb.conditions import Key, Attr
import uuid
#https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/GSI.html

DYNAMO_DB = boto3.resource('dynamodb')
PARAMETERS_MINMAX_TABLE = ""  # To be decided
OPTIMISATION_RECIPE_TABLE = "LettuceGrowth"

########### MOK AWS UTILS ###################
def initialize_parameters_minmax(experiment_id, params):
    dynamo_table = DYNAMO_DB.Table(PARAMETERS_MINMAX_TABLE)
    dynamo_table.put_item(
        Item={
            'parameters_list': params,
            'experiment_id': experiment_id,
        }
    )

def create_sample_recipe(experiment_id, duration):
    parameters_minmax_list = get_saved_parameters_minmax(experiment_id)
    add_optimisation_recipe(uuid.uuid4(), )

def get_saved_parameters_minmax(experiment_id):
    dynamo_table = dynamodb.Table(PARAMETERS_MINMAX_TABLE)
    response = dynamo_table.query(
        KeyConditionExpression=Key('experiment_id').eq(experiment_id)
    )
    mins = response['fnmin']
    maxs = response['fnmax']
    steps = response['steps']
    params =response['params_list']
    return mins, maxs, steps, params

#############################################

def reciperesults(experiment_id, table_name):
    # interacts with DynamoDB table
    # returns all data for the given recipe
    dynamodb = boto3.resource('dynamodb')
    data = dynamodb.Table(table_name)
    response = data.query(
        KeyConditionExpression=Key('experiment_id').eq(experiment_id) #current recipe is take from arg
    )
    table_items = response['Items']
    if len(table_items) == 0:
        print(f"**** WARNING: No items for experiment {experiment_id} ****")
    else:
        print(f"Acquiring items for experiment {experiment_id}...")
    return table_items

def getparams(experiment_id, table_name):
    #gets all x-values and y-values for given recipe from the table
    table_items = reciperesults(experiment_id, table_name) # gets table data for given recipe
    x_params = []
    y_params = []
    for item in table_items:
        if item['pending'] == 'false':             ##only add non-pending items
            x_params.append(item['parameters'])    ##change this if you change the header name for x-params
            y_params.append(item['actualgrowth'])  ##change this if you change the header name for y-param
        x_GT = np.array(x_params, dtype=np.float64) #convert to float & same format as used in the other codes
        y_GT = np.array(y_params, dtype=np.float64)
    return (x_GT, y_GT)
        
def getpending(experiment_id, table_name):
    #gets items that are still pending an 'actual growth' value
    table_items = reciperesults(experiment_id, table_name) # gets table data for given recipe
    pending_uuid = []
    pending_param = []
    for i in range(len(table_items)):
        if table_items[i]['pending'] == 'true':
            pending_uuid.append(table_items[i]['uuid'])
            pending_param.append(table_items[i]['parameters'])
    
    if len(pending_uuid) == 0:
        print(f"No items pending items for experiment {experiment_id}")
    else:
        print(f"{len(pending_uuid)} pending item(s) for experiment {experiment_id}:")
        for i in range(len(pending_uuid)):
            print(f"UUID {pending_uuid}—{ np.array(pending_param, dtype=np.float64)}")
    return pending_uuid


def add_optimisation_recipe(uuid, parameters_list, parameters, experiment_id, exp_dur, type):
    #adds item to table
    dynamo_db = DYNAMO_DB.Table(OPTIMISATION_RECIPE_TABLE)
    dec_param = []
    for i in parameters:
        dec_param.append(Decimal(i))
    dynamo_db.put_item(
        Item={
            'uuid': uuid,
            'parameters_list': parameters_list,
            'parameters': dec_param,
            'experiment_id': experiment_id,
            #'actualgrowth':
            'exp_dur' : exp_dur,
            'type': type,
            'pending': 'start',
            #'timestamp': timestamp
        })
    print(f"Added UUID {uuid} to experiment {experiment_id}.")
    
def addtime(experiment_id, uuid, timestamp, table_name):
    #adds time to uuid
    dynamodb = boto3.resource('dynamodb')
    data = dynamodb.Table(table_name) 
    table_items = reciperesults(experiment_id, table_name) #get all items under the experiment
    temp = {}
    for item in table_items:
        if uuid == item['uuid']: # search for the uuid
            temp = item
    temp['timestamp'] = Decimal(timestamp) #add timestamp
    temp['pending'] = 'growth' #change pending from start to growth
    data.put_item(Item=temp)
    print(f"Added time {timestamp} to {experiment_id}, UUID: {uuid}")
    
def update_growth(experiment_id, uuid, growthscore):
    #adds growthscore to uuid
    try:
        Decimal(growthscore)
        dynamodb = boto3.resource('dynamodb')
        data = dynamodb.Table(OPTIMISATION_RECIPE_TABLE) 
        table_items = reciperesults(experiment_id, OPTIMISATION_RECIPE_TABLE) #get all items under the experiment
        temp = {}
        for item in table_items:
            if uuid == item['uuid']: # search for the uuid
                temp = item
        temp['actualgrowth'] = round(Decimal(growthscore),10) #add timestamp
        temp['pending'] = 'false' #change pending from growth to false
        data.put_item(Item=temp)
        print(f"Added growthscore of {growthscore} to {experiment_id}, UUID: {uuid}")
    except:
        print(f"ERROR: growthscore of '{growthscore}' is not valid")
    


def filltable():
    dynamodb = boto3.resource('dynamodb')
    data = dynamodb.Table(table_name)
    data.put_item(
        Item={
            'uuid': '1',
            'parameters_list': parameters_list,
            'parameters': [Decimal('14.0')],
            'experiment_id': '1d_test',
            'actualgrowth': Decimal('918.53108652'), 
            'exp_dur': Decimal('72'),
            'type': 'init',
            'pending': 'false',
            'timestamp': 1559620800,
        })
        
    data.put_item(Item={'uuid': '2','parameters_list': parameters_list,'parameters':  [Decimal('5.0')],'experiment_id': '1d_test','actualgrowth': Decimal('396.20033535'), 'exp_dur': Decimal('72'), 'type': 'init','pending': 'false','timestamp': 1560744000})
    data.put_item(Item={'uuid': '3','parameters_list': parameters_list,'parameters':  [Decimal('22.0')],'experiment_id': '1d_test','actualgrowth': Decimal('3804.8058396'), 'exp_dur': Decimal('72'), 'type': 'init','pending': 'false','timestamp': 1561021200})
    data.put_item(Item={'uuid': '4','parameters_list': parameters_list,'parameters':  [Decimal('24.0')],'experiment_id': '1d_test','actualgrowth': Decimal('1943.87051092'), 'exp_dur': Decimal('72'), 'type': 'opti','pending': 'false','timestamp': 1561365000})
    data.put_item(Item={'uuid': '5','parameters_list': parameters_list,'parameters':  [Decimal('18.0')],'experiment_id': '1d_test','actualgrowth': Decimal('606.21632107'), 'exp_dur': Decimal('72'), 'type': 'opti','pending': 'false','timestamp': 1562214600})
    data.put_item(Item={'uuid': '6','parameters_list': parameters_list,'parameters':  [Decimal('9.5')],'experiment_id': '1d_test','actualgrowth': Decimal('670.32018962'), 'exp_dur': Decimal('72'), 'type': 'opti','pending': 'false','timestamp': 1562558400})

    data.put_item(
        Item={
            'uuid': '7',
            'parameters_list': parameters_list,
            'parameters':  [Decimal('20.5')],
            'experiment_id': '1d_test',
            'exp_dur': Decimal('72'),
            #'actualgrowth': Decimal('1474.51903652'), 
            'type': 'opti',
            'pending': 'growth',#'false',
            'timestamp': 1562832000
        })

##testing stuff
# def main():
#     recipe = '1d_test'
#     print("getpending")
#     getpending(recipe)
#     print("\ngetparam")
#     g = getparams(recipe)
#     print(f"X_GT: {g[0]}\nY_GT: {g[1]}")
# 
# main()
# 
# experiment_id = '1d_test'











