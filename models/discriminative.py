from __future__ import division
#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn import svm

import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_erro


#TODO
kernel_metrics = {
	'svm': [ 'rbf', 'poly','sigmoid']
}

class Discriminative(object):

	def __init__(self, modelname, params):

		self.__modelname = modelname

		if (self.__modelname == 'svm'):
			self.__svmKernelMetrics = kernel_metrics[modelname]
			# self.__degree_list = params['degree']
			self.__gamma_list = params['gamma']

		elif (self.__modelname == 'lstm'):
			pass


	def getKernelMetricsList(self):
		if (self.__modelname == 'svm'):
			return self.__svmKernelMetrics
		else:
			return None



	def analyze(self, data, columns, label):
		results_in_json = {}

		if (self.__modelname == 'svm'):
			# estimate the outlier fraction
			for gamma_val in self.__gamma_list:
				for kernel_metric in self.__svmKernelMetrics:
					# separate the data into normal and abnormal data
					data_size = data[label].count()
					outliers = data[data[label]==-1][columns]
					non_outliers = data[data[label]==1][columns]
					n_outliers = outliers.count()
					# outlier_fraction = data[data[label]==-1][label].count() / data[label].count()
					model = svm.OneClassSVM(nu=0.2, kernel=kernel_metric, gamma=gamma_val)

					model.fit(non_outliers)

					y_pred_normal = model.predict(non_outliers)
					y_pred_abnormal = model.predict(outliers)

					n_correct_labels_normal = y_pred_normal[y_pred_normal==1].size
					n_correct_labels_abnormal = y_pred_abnormal[y_pred_abnormal==-1].size

					n_error_normal = y_pred_normal[y_pred_normal == -1].size
					n_error_outliers = y_pred_abnormal[y_pred_abnormal == 1].size
					n_errors = n_error_normal + n_error_outliers
					error = n_errors / data_size
					n_corrects = n_correct_labels_normal + n_correct_labels_abnormal
					accuracy = n_corrects / data_size

					result_in_json = {
						'error' : error,
						'accuracy' : accuracy,
						# 'non_outliers' : non_outliers.to_json(orient='columns'),
						# 'outliers' : outliers.to_json(orient='columns'),
						'gamma_val' : gamma_val
					}

					results_in_json[kernel_metric] = result_in_json

		elif (self.__modelname == 'lstm'):

			pass

		return results_in_json

