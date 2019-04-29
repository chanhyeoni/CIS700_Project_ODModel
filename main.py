#!/usr/bin/env python

import sys
import os

from helper.helper import *
from models.instancebased import *
from models.generative import *
from models.discriminative import *



if __name__=="__main__":
	if len(sys.argv) < 5:
		print 'Usage: %s <data_filename> <attack_scenario> <modelname> <publish_rate> <params>' % sys.argv[0]
		sys.exit(1)

	# assign arguments to parameters
	filename = sys.argv[1]
	#filename = 'gps_0.1_10.csv'
	attack_scenario = sys.argv[2]
	#attack_scenario = 'similar_values'
	modelname = sys.argv[3]
	publish_rate = sys.argv[4]
	params = {}

	# n_neighbors


	if attack_scenario not in attack_scenario_list:
		print 'attack_scenario must be one of zero_lat_long, similar_values ...'
		sys.exit(1)

	if modelname not in all_models_list:
		print 'modelname must be one of the following: [' + printModelsInStr(all_models_list) + ']'
		sys.exit(1)

	# get the data
	data = formatData(filename, attack_scenario)

	# create json data structures for results
	arguments_for_graphs_json = {}
	arguments_for_graphs_json['filename'] = filename
	arguments_for_graphs_json['attack_scenario'] = attack_scenario
	arguments_for_graphs_json['publish_rate'] = publish_rate
	arguments_for_graphs_json['model'] = modelname

	# get instance type by looking at modelname
	learning_type = getLearningType(modelname)

	if learning_type == None or learning_type == "":
		print 'learning_type must be one of the following: [' + learning_type.keys() + ']'
		sys.exit(1)		

	# train model
	model = None
	params = {}

	if learning_type == 'instance-based':

		if (modelname=='LOF'):
			params = {"n_neighhbors": [100]}

			learning_model = InstanceBased(modelname, params) # cross-validation
			arguments_for_graphs_json['distance_metric_list'] = learning_model.getDistanceMetricsList()

			dist_metrics_results_json = learning_model.analyze(data, ['latitude', 'longitude'], 'label')

			for distance_metric in dist_metrics_results_json.keys():
				arguments_for_graphs_json[distance_metric] = dist_metrics_results_json[distance_metric]

			saveToJson(modelname,arguments_for_graphs_json)

	elif learning_type == 'discriminative':
		# TODO
		if (modelname == 'svm'):
			params = {'gamma': [0.1]}

			learning_model = Discriminative(modelname, params)
			arguments_for_graphs_json['kernel_metric_list'] = learning_model.getKernelMetricsList()

			kernel_metrics_results_json = learning_model.analyze(data, ['latitude', 'longitude'], 'label')

			for kernel_metric in kernel_metrics_results_json.keys():
				arguments_for_graphs_json[kernel_metric] = kernel_metrics_results_json[kernel_metric]

			saveToJson(modelname,arguments_for_graphs_json)	

		elif (modelname == 'lstm'):
			params = {}

			pass


	elif learning_type == 'generative':
		# TODO
		pass























