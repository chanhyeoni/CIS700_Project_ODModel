from __future__ import division
#!/usr/bin/env python

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

from helper.helper import *


if __name__ == '__main__':
	path_to_result_json = sys.argv[1]

	xmin = int(sys.argv[2])
	xmax = int(sys.argv[3])
	ymin = int(sys.argv[4])
	ymax = int(sys.argv[5])


	with open(path_to_result_json) as json_file:
		data = json.load(json_file)
		filename = data['filename']
		attack_scenario = data['attack_scenario']
		publish_rate = data['publish_rate']
		data = data["svm"]

		X = np.asarray(toPandasData(attack_scenario,filename)[0])

		outliers = X[X[:,2]==-1]
		non_outliers = X[X[:,2]==1]

		outliers = np.delete(outliers, 2, 1)
		non_outliers = np.delete(non_outliers, 2, 1)

		gamma_val = data['gamma_val']
		kernel_metric = data['kernel_metric']

		xx, yy = np.meshgrid(np.linspace(xmin, xmax, 500), np.linspace(ymin, ymax, 500))
		model = svm.OneClassSVM(nu=0.25, kernel=kernel_metric, gamma=gamma_val)
		model.fit(non_outliers)

		Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)

		plt.title("Outlier Detection (" + kernel_metric + ")")

		plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
		a = plt.contour(xx, yy, Z, levels=[0], linewidths=4, colors='darkred')
		print (a.collections[0])
		plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

		s = 40
		b1 = plt.scatter(non_outliers[:, 0], non_outliers[:, 1], c='white', s=s, edgecolors='k') # b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s, edgecolors='k')
		c = plt.scatter(outliers[:, 0], outliers[:, 1], c='gold', s=s, edgecolors='k')
		plt.axis('tight')
		plt.xlim((xmin, xmax))
		plt.ylim((ymin, ymax))
		plt.legend(
			[a.collections[0], b1,  c],
			["learned frontier", "non-anomalous observations", "anomalous observations"],loc="upper left",
			prop=matplotlib.font_manager.FontProperties(size=11)
		)

		plt.xlabel("error: %f ; accuracy: %f" % (data[kernel_metric]['error'], data[kernel_metric]['accuracy']))

		filename = './graphs/svm.png'
		plt.savefig(filename)

			# plt.show()