#!/usr/bin/env python
import sys
import os
import json

import numpy as np
import matplotlib.pyplot as plt

from helper.helper import *


if __name__=="__main__":

	path_to_result_json = sys.argv[1]
	#path_to_result_json = "./results/LOF.txt"

	xmin = int(sys.argv[2])
	xmax = int(sys.argv[3])
	ymin = int(sys.argv[4])
	ymax = int(sys.argv[5])

	with open(path_to_result_json) as json_file:  
		data = json.load(json_file)
		filename = data['filename']
		attack_scenario = data['attack_scenario']
		publish_rate = data['publish_rate']

		X = np.asarray(formatData(filename, attack_scenario))

    	distance_metric_list = data['distance_metric_list']

    	for distance_metric in distance_metric_list:

			X_scores = np.asarray(data[distance_metric]['X_scores'])

			n_errors = data[distance_metric]['n_errors']
			error = data[distance_metric]['error']
			accuracy = data[distance_metric]['accuracy']

			plt.title("Local Outlier Factor for GPS (" + distance_metric + ")")
			plt.scatter(X[:,0], X[:,1], color='k', s=3., label='Data points')
			# plot circles with radius proportional to the outlier scores
			radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())

			plt.scatter(X[:,0], X[:,1], s=1000 * radius, edgecolors='r',
			            facecolors='none', label='Outlier scores')
			plt.axis('tight')
			plt.xlim((xmin, xmax))
			plt.ylim((ymin, ymax))
			plt.xlabel("accuracy: %f, error: %f" % (accuracy, error))
			legend = plt.legend(loc='upper left')
			legend.legendHandles[0]._sizes = [10]
			legend.legendHandles[1]._sizes = [20]
			# plt.show()

			filename = './graphs/LOF_' + distance_metric + '_' + publish_rate + '.png'
			plt.savefig(filename)

			plt.close()