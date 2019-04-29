from __future__ import division
#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn import neighbors

from helper.helper import *

distance_metrics_dic = {
	'LOF' : ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan'],
	'KNN' : ['minkowski', 'chebyshev', 'euclidean', 'wminkowski', 'seuclidean', 'manhattan', 'mahalanobis','haversine']
}


class InstanceBased(object):

	def __init__(self, modelname, params):

		self.__modelname = modelname
		self.__distMetricsList = distance_metrics_dic[modelname]
		if (self.__modelname == 'LOF'):
			self.__n_neighbors_list = parmas['n_neighbors']


	def getDistanceMetricsList(self):
		return self.__distMetricsList


	def analyze(self, data, columns, label):
		results_in_json = {}

		# k_fold_data = conductKFoldCV(data, columns, label)
		
		if (self.__modelname == 'LOF'):


			for n_neighbors_param in self.__n_neighbors_list:
				for distance_metric in self.__distMetricsList:

					model = neighbors.LocalOutlierFactor(n_neighbors=self.__n_neighbors, algorithm='auto', metric=distance_metric)
					y_pred = model.fit_predict(data[columns])
					n_errors = (y_pred != data[label]).sum()
					n_rights = (y_pred == data[label]).sum()
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

					results_in_json[distance_metric] = dist_metrics_result_json

		elif (self.__modelname == 'KNN'):
		# 	#TODO

			pass

		return results_in_json


	# def __estimateNofNeighhbors(self):
	# 	nOutliers = getNOutliers(self.__data)

	# 	nTotalData = self.__data.count()

	# 	self.__n_neighbors = nTotalData / nOutliers


