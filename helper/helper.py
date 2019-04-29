#!/usr/bin/env python
import pandas as pd
import json

from sklearn.model_selection import KFold

attack_scenario_list = ['zero_lat_long', 'similar_values']
instance_models_list = ['LOF', 'knn']
generative_models_list = ['hmm']
discriminative_models_list = ["arima", "svm", "lstm", "dnn"]
learning_type_list = ['discriminative', 'generative', 'instance-based']

models_dic = {
	'instance-based' : instance_models_list,
	'generative' : generative_models_list,
	'discriminative' : discriminative_models_list
}

instance_models_list.extend(generative_models_list)
instance_models_list.extend(discriminative_models_list)
all_models_list = instance_models_list
instance_models_list = ['LOF', 'KNN']
models_dic['instance-based'] = instance_models_list


def getLearningType(modelname):
	# by looking at the name of the model, get the learning type of the model
	for learningType, models_list in models_dic.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
		if modelname in models_list:

			return learningType

	return ""



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


def getNOutliers(data):
	# get the number of outliers	
	# data : pandas DataFrame
	return data[data['label'] == -1].count()


def conductKFoldCV(k, data, column_list, label):
	kf = KFold(n_splits=k)
	k_tracker = 1
	result = {}
	for train_idx, test_idx in kf.split(data[column_list]):
		X_train, X_test = data[column_list].iloc(train_idx), data[column_list].iloc(test_idx)
		y_train, y_test = data[label].iloc(train_idx), data[label].iloc(test_idx)

		result[k_tracker] = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

	return result




