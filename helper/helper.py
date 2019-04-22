#!/usr/bin/env python
import pandas as pd
import json

attack_scenario_list = ['zero_lat_long', 'similar_values']
models = ['LOF', 'KNN']
learning_type_list = ['discriminative', 'generative', 'instance-based']

def isNonZero(args):
	return args[0] != 0 and args[1] != 0

def convertToNum(boolean_val):
	return 1 if (boolean_val==True) else -1 

def formatData(filename, attack_scenario):
	# get the full path to file
	path_to_file = "./data/" + attack_scenario + "/" + filename
	# load the data in Pandas
	data = pd.read_csv(path_to_file, delimiter=',')
	# drop the timestamp
	data = data.drop('rosbagTimestamp', axis=1)
	# define classes based on attack scenarios
	# if attack_scenario == 'zero_lat_long':
	# 	data['label'] = data.apply(isNonZero, axis=1)
	# 	# do boolean indexing to assign value to 
	# 	data['label'] = data['label'].apply(lambda x: convertToNum(x))

	return data

def saveToJson(filename, jsonData):
	# create file and save the json data into that file
	with open('./results/' + str(filename) + ".txt", 'w') as outfile:  
		json.dump(jsonData, outfile)