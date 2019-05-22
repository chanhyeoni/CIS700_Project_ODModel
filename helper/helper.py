#!/usr/bin/env python
import pandas as pd
import json

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split




attack_scenario_list = ['zero_lat_long', 'similar_values', 'infinity']
instance_based_models_list = ['lof', 'knn']
generative_models_list = ['hmm']
discriminative_models_list = ["arima", "svm", "lstm", "dnn", "gru"]

all_models_list = instance_based_models_list + generative_models_list + discriminative_models_list 

models_dic = {
	'instance-based' : instance_based_models_list,
	'generative' : generative_models_list,
	'discriminative' : discriminative_models_list
}




def printModelsInStr(all_models_list):
	# a string representation of all possible models used 
	models = ''
	for i in range(0, len(all_models_list)):
		models = models + all_models_list[i]
		if (i < len(all_models_list) - 1):
			models = models + ", "

	return models


def isNonZero(args):
	return args[0] != 0 and args[1] != 0

def convertToNum(boolean_val):
	return 1 if (boolean_val==True) else -1 

def toPandasData(attack_scenario, filename):
	# get the full path to file
	path_to_file = "./data/" + attack_scenario + "/" + filename
	# load the data in Pandas
	data = pd.read_csv(path_to_file, delimiter=',')
	# drop the timestamp
	data = data.drop('rosbagTimestamp', axis=1)
	# drop all columns whose values are all zeroes
	# data = data.loc[:, (data != 0).any(axis=0)]
	columns = data.columns.tolist()
	# define classes based on attack scenarios
	# if attack_scenario == 'zero_lat_long':
	# 	data['label'] = data.apply(isNonZero, axis=1)
	# 	# do boolean indexing to assign value to 
	# 	data['label'] = data['label'].apply(lambda x: convertToNum(x))
	return data, columns

def saveToJson(filename, jsonData):
	# create file and save the json data into that file
	with open('./results/' + str(filename) + ".json", 'w') as outfile:  
		# json.dump(jsonData, outfile)
		json.dump(jsonData, outfile, indent=4, separators=(',', ': '), sort_keys=True)
    	#add trailing newline for POSIX compatibility
    	


def getNOutliers(data):
	# get the number of outliers	
	# data : pandas DataFrame
	return data[data['label'] == -1].count()


def conductKFoldCV(k, data):
	kf = KFold(n_splits=k)
	results = []
	for train_idx, test_idx in kf.split(data):
		train, test = data.iloc(train_idx), data.iloc(test_idx)

		results.append({
			'train': train.to_json(orient='index'), 
			'test': test.to_json(orient='index'), 
		})

	return results


def split_data(split_ratio, data):

	train, test = train_test_split(data,test_size=split_ratio, random_state=42)

	result = [{
			'train': train.to_json(orient='index'), 
			'train': test.to_json(orient='index'), 
	}]

	return result





