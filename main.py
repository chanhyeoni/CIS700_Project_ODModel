from __future__ import division
#!/usr/bin/env python


import numpy as np
import pandas as pd
from sklearn import neighbors

import sys
import os

from helper.helper import *


if __name__=="__main__":
	if len(sys.argv) < 4:
		print 'Usage: %s <data_filename> <attack_scenario> <learning_type> <publish_rate>' % sys.argv[0]
		sys.exit(1)

	# assign arguments to parameters
	filename = sys.argv[1]
	#filename = './data/gps_bag_1.csv'
	attack_scenario = sys.argv[2]
	learning_type = sys.argv[3]
	publish_rate = sys.argv[4]

	# n_neighbors
	n_neighbors = 

	if attack_scenario not in attack_scenario_list:
		print 'attack_scenario must be one of zero_lat_long, similar_values ...'
		sys.exit(1)

	if learning_type not in learning_type_list:
		print 'learning_type must be discriminative, generative, or instance-based'
		sys.exit(1)

	# get the data
	data = formatData(filename, attack_scenario)

	# create json data structures for results
	arguments_for_graphs_json = {}
	arguments_for_graphs_json['filename'] = filename
	arguments_for_graphs_json['attack_scenario'] = attack_scenario
	arguments_for_graphs_json['publish_rate'] = publish_rate


	# train model
	model = None
	for modelname in models:
		if (modelname == 'LOF'):
			arguments_for_graphs_json['model'] = modelname
			distance_metrics = ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']
			arguments_for_graphs_json['distance_metric_list'] = distance_metrics
			for distance_metric in distance_metrics:
				dist_metrics_result_json = {}
				model = neighbors.LocalOutlierFactor(n_neighbors=100, algorithm='auto', metric=distance_metric)
				y_pred = model.fit_predict(data[['latitude', 'longitude']])
				n_errors = (y_pred != data['label']).sum()
				n_rights = (y_pred == data['label']).sum()
				error = n_errors / data.shape[0]
				accuracy = n_rights / data.shape[0]
				X_scores = model.negative_outlier_factor_
				#print_statement_1 = "N of Errors of of model " + modelname + " using argument " + distance_metric + ": " + str(n_errors)
				#print_statement_2 = "Error of model " + modelname + " using argument " + distance_metric + ": " + str(error)

				dist_metrics_result_json = {
					'X_scores'		:		X_scores.tolist(),
					'n_errors'		:		n_errors,
					'error' 		:		error,
					'accuracy'		:		accuracy
				}

				arguments_for_graphs_json[distance_metric] = dist_metrics_result_json

				# os.system(command)
			saveToJson(modelname,arguments_for_graphs_json)

		elif (modelname == 'KNN'):
			distance_metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']























