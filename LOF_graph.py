#!/usr/bin/env python
import sys
import os
import json

import numpy as np
import matplotlib.pyplot as plt

from helper.helper import *


if __name__=="__main__":

	path_to_result_json = sys.argv[1]
	# path_to_result_json = "./results/result_data.csv_similar_values.json"

	xmin = int(sys.argv[2])
	xmax = int(sys.argv[3])
	ymin = int(sys.argv[4])
	ymax = int(sys.argv[5])

	with open(path_to_result_json, 'r') as json_file:  
		data = json.load(json_file)
		filename = data['filename']
		attack_scenario = data['attack_scenario']
		publish_rate = data['publish_rate']

		data = data["lof"]

		df, columns = toPandasData(attack_scenario,filename)
		X = np.asarray(df)

		X_scores = np.array(data['X_scores'])

		error_train = data['error_train']
		accuracy_train = data['accuracy_train']
		error_test = data['error_test']
		accuracy_test = data['accuracy_test']
		n_neighbors = data['n_neighbors']
		distance_metric = data['distance_metric']

		plt.title("Local Outlier Factor for GPS using " + distance_metric + " and " + str(n_neighbors) + " neighbors")
		plt.scatter(X[:,0], X[:,1], color='k', s=10., label='Data points')
		# plot circles with radius proportional to the outlier scores
		radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())

		plt.scatter(X[:,0], X[:,1], s=1000 * radius, edgecolors='r',
		            facecolors='none', label='Outlier scores')
		plt.axis('tight')
		plt.xlim((xmin, xmax))
		plt.ylim((ymin, ymax))
		plt.xlabel("training error: %f, testing error: %f" % (error_train, error_test))
		legend = plt.legend(loc='upper left')
		legend.legendHandles[0]._sizes = [10]
		legend.legendHandles[1]._sizes = [20]
		# plt.show()

		filename = './graphs/LOF.png'
		plt.savefig(filename)

		plt.close()