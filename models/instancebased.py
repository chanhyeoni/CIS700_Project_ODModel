from __future__ import division
#!/usr/bin/env python

import numpy as np
import pandas as pd

from sklearn import neighbors
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from helper.helper import *


class LoF(object):

	def __init__(self, params, k):
		self.__distMetricsList = params["distance_metrics"]
		self.__n_neighbors_list = params['n_neighbors']
		self.__model = neighbors.LocalOutlierFactor()
		self.__fold_val = k

	def __train(self, train_data, columns, label, n_neighbors=None, distance_metric=None):
		if (n_neighbors== None):
			n_neighbors = self.__n_neighbors_list[0]
		if (distance_metric==None):
			distance_metric=self.__distMetricsList[0]

		self.__model = neighbors.LocalOutlierFactor(n_neighbors=n_neighbors, algorithm='auto', metric=distance_metric)
		self.__model.fit_predict(train_data[columns])


	def __get_test_error(self, test_data, columns, label):
		y_pred = self.__model.fit_predict(test_data[columns])
		n_errors = (y_pred != test_data[label]).sum()
		error = n_errors / test_data.shape[0]

		return error

	def __cross_validate(self, data, columns, label):
		kf = KFold(n_splits=self.__fold_val)

		min_error_avr = float("inf")
		best_model_param = {}
		for n_neighbors_val in self.__n_neighbors_list:
			for distance_metric in self.__distMetricsList:

				new_error_avr = 0
				for train_idx, test_idx in kf.split(data):
					train_data, test_data = data.iloc[train_idx,:], data.iloc[test_idx,:]
					self.__train(train_data, columns, label, n_neighbors_val, distance_metric)
					error_val = self.__get_test_error(test_data, columns, label)
					new_error_avr = new_error_avr + error_val

				new_error_avr = new_error_avr / self.__fold_val

				if (new_error_avr < min_error_avr):
					best_model_param['distance_metric'] = distance_metric
					best_model_param['n_neighbors'] = n_neighbors_val
					min_error_avr = new_error_avr

		return best_model_param


	def analyze(self, data, columns, label):
		results_in_json = {}
		
		best_model_param = self.__cross_validate(data, columns, label)

		print best_model_param
		train, test = train_test_split(data,test_size=0.25, random_state=42)

		self.__train(train, columns, label, best_model_param['n_neighbors'], best_model_param['distance_metric'])
		y_pred_train = self.__model.fit_predict(train[columns])
		n_errors_train = (y_pred_train != train[label]).sum()
		n_rights_train = (y_pred_train == train[label]).sum()
		error_train = n_errors_train / train.shape[0]
		accuracy_train = n_rights_train / train.shape[0]

		y_pred_test = self.__model.fit_predict(test[columns])
		n_errors_test = (y_pred_test != test[label]).sum()
		n_rights_test = (y_pred_test == test[label]).sum()
		error_test = n_errors_test / test.shape[0]
		accuracy_test = n_rights_test / test.shape[0]

		X_scores = self.__model.negative_outlier_factor_
		#print_statement_1 = "N of Errors of of model " + modelname + " using argument " + distance_metric + ": " + str(n_errors)
		#print_statement_2 = "Error of model " + modelname + " using argument " + distance_metric + ": " + str(error)

		result_json = {
			'X_scores'					:		X_scores.tolist(),
			'error_train' 				:		error_train,
			'accuracy_train'			:		accuracy_train,
			'error_test'				:		error_test,
			'accuracy_test'				:		accuracy_test,
			'n_neighbors'				:		best_model_param['n_neighbors'],
			'distance_metric'			:		best_model_param['distance_metric']
		}

		return result_json


	# def __estimateNofNeighhbors(self):
	# 	nOutliers = getNOutliers(self.__data)

	# 	nTotalData = self.__data.count()

	# 	self.__n_neighbors = nTotalData / nOutliers


