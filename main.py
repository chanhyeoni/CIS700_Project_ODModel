#!/usr/bin/env python

import sys
import os
import json

from helper.helper import *
from models.InstanceBased import *
from models.Generative import *
from models.Discriminative import *

# create json data structures for results
arguments_for_graphs_json = {}


def loadData(data_dic):
	###### check data part of configuration file ######

	# assign arguments to parameters
	filename = data_dic['filename']
	#filename = 'gps_0.1_10.csv'
	attack_scenario = data_dic['attack_scenario']
	#attack_scenario = 'similar_values'
	publish_rate = data_dic['publish_rate']

	if attack_scenario.lower() not in attack_scenario_list:
		print 'attack_scenario must be one of zero_lat_long, similar_values, infinity ...'
		sys.exit(1)

	# get the data
	label = data_dic['labelname']
	data, columns = toPandasData(attack_scenario,filename)
	try:
		columns.remove(label)
	except ValueError:
		print 'the label' + label + ' does not exist in the dataset loaded'
		sys.exit(1)

	return data, label, columns, data_dic['k-fold']

# def splitData(split_method_dic, data):
# 	# split the data based on the method specified by config file
# 	dataset = {}
# 	method_name = ''
# 	data_json = None
# 	if (split_method_dic != None or split_method_dic != {}):
# 		method_name = split_method_dic['name']
# 		if (method_name == 'train_test_split'):
# 			data_json = split_data(data)
# 		else (method_name == 'k_fold'):
# 			data_json = conductKFoldCV(data)
# 			pass

# 	return data_json


def train(model_config, dataset, columns, label, fold_val):

	# parse dataset

	for model_info in model_config:
		modelname = model_info['modelname']
		params = model_info["params"]
		learning_model = None
		if (modelname.upper()=='LOF'):
			
			learning_model = LoF(params, fold_val) # cross-validation
			# for distance_metric in dist_metrics_results_json.keys():
			# 	arguments_for_graphs_json[distance_metric] = dist_metrics_results_json[distance_metric]

		elif (modelname.upper() == 'SVM'):
			
			learning_model = OneClassSVM(params, fold_val)

			# for kernel_metric in kernel_metrics_results_json.keys():
			# 	arguments_for_graphs_json[kernel_metric] = kernel_metrics_results_json[kernel_metric]

		elif (modelname.upper() == 'LSTM'):
		 	pass
		elif (modelname.upper() == 'HMM'):
		 	pass
		elif (modelname.upper() == ''):
		 	pass	

		 # cross-validation
		arguments_for_graphs_json[modelname] = learning_model.analyze(dataset, columns, label)



if __name__=="__main__":

	###### load configuration file ######
	if len(sys.argv) < 2:
		print 'Usage: %s </path/to/config_file>' % sys.argv[0]
		sys.exit(1)

	filename = sys.argv[1]
	with open(filename, 'r') as json_file:
		print filename
		config_data = json.load(json_file)
		###### load configuration file ######


		###### data part ######
		data_dic = config_data['data']
		if (data_dic == None or data_dic == '' or data_dic == {}):
			print "No information about data exists in config file."
			sys.exit(1)
		data_pd, data_pd_label, data_pd_columns, fold_val = loadData(data_dic)




		###### model part ######
		models_dic = config_data['models']
		if (models_dic == None or models_dic == '' or models_dic == {}):
			print "no information about model exists in config file. "
			sys.exit(1)

		###### model part ######
		train(models_dic, data_pd, data_pd_columns, data_pd_label, fold_val)
		###### model part ######


		arguments_for_graphs_json['filename'] = config_data['data']['filename']
		arguments_for_graphs_json['attack_scenario'] = config_data['data']['attack_scenario']
		arguments_for_graphs_json['publish_rate'] = config_data['data']['publish_rate']
		# arguments_for_graphs_json['model'] = modelname

		filename = "result_" + arguments_for_graphs_json['filename'] + "_" + arguments_for_graphs_json['attack_scenario'] 
		saveToJson(filename,arguments_for_graphs_json)
























