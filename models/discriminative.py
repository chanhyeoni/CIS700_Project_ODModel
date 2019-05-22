from __future__ import division
#!/usr/bin/env python

import os
import math
import datetime as dt
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from helper.LSTMDataLoader import LSTMDataLoader
from helper.Timer import Timer

############ OneClassSVM ############
class OneClassSVM(object):

    def __init__(self, params, k):

        self.__svmKernelMetrics = params["kernel_metrics"]
        self.__gamma_list = params['gamma']
        self.__model = svm.OneClassSVM(nu=0.2, degree=3)
        self.__fold_val = k

    def __split(self, data, label):
        # separate the data into normal and abnormal data
        outliers = data[data[label]==-1]
        non_outliers = data[data[label]==1]

        return outliers, non_outliers


    def __train(self, train_data, columns=None, label=None, gamma_val=None, kernel_metric=None):
        # train the model
        if (gamma_val== None):
            gamma_val = self.__gamma_list[0]
        if (kernel_metric==None):
            kernel_metric=self.__svmKernelMetrics[0]

        self.__model = svm.OneClassSVM(nu=0.2, kernel=kernel_metric, gamma=gamma_val, degree=3)
        self.__model.fit(train_data[columns])


    def __get_test_error(self, test_data, columns, label):
        
        outliers_test, non_outliers_test = self.__split(test_data, label)

        y_pred_normal = self.__model.predict(non_outliers_test[columns])
        if (outliers_test.size <= 0):
            y_pred_abnormal = np.array([])
        else:
            y_pred_abnormal = self.__model.predict(outliers_test[columns])

        n_error_normal = y_pred_normal[y_pred_normal == -1].size
        n_error_outliers = y_pred_abnormal[y_pred_abnormal == 1].size
        n_errors = n_error_normal + n_error_outliers
        error = n_errors / test_data.shape[0]

        return error

    def __cross_validate(self,data, columns, label):
        kf = KFold(n_splits=self.__fold_val)

        min_error_avr = float("inf")
        best_model_param = {}
        for gamma_val in self.__gamma_list:
            for kernel_metric in self.__svmKernelMetrics:

                new_error_avr = 0
                for train_idx, test_idx in kf.split(data):
                    train, test = data.iloc[train_idx,:], data.iloc[test_idx,:]

                    outliers_train, non_outliers_train = self.__split(train, label)

                    n_outliers = outliers_train.count()

                    self.__train(non_outliers_train, columns=columns, gamma_val=gamma_val, kernel_metric=kernel_metric)

                    error_val = self.__get_test_error(test, columns, label)
                    new_error_avr = new_error_avr + error_val

                new_error_avr = new_error_avr / self.__fold_val

                if (new_error_avr < min_error_avr):
                    best_model_param['kernel_metric'] = kernel_metric
                    best_model_param['gamma_val'] = gamma_val
                    min_error_avr = new_error_avr

        return best_model_param


    def analyze(self, data, columns, label):
        results_in_json = {}

        best_model_param = self.__cross_validate(data, columns, label)

        outliers, non_outliers = self.__split(data, label)

        self.__train(non_outliers, columns, label, best_model_param['gamma_val'], best_model_param['kernel_metric'])

        y_pred_normal = self.__model.predict(non_outliers[columns])
        y_pred_abnormal = self.__model.predict(outliers[columns])

        n_correct_labels_normal = y_pred_normal[y_pred_normal==1].size
        n_correct_labels_abnormal = y_pred_abnormal[y_pred_abnormal==-1].size
        n_corrects = n_correct_labels_normal + n_correct_labels_abnormal

        n_error_normal = y_pred_normal[y_pred_normal == -1].size
        n_error_outliers = y_pred_abnormal[y_pred_abnormal == 1].size
        n_errors = n_error_normal + n_error_outliers

        error = n_errors / data.shape[0]
        accuracy = n_corrects / data.shape[0]

        result_in_json = {
            'error' : error,
            'accuracy' : accuracy,
            'gamma_val' : best_model_param['gamma_val'],
            'kernel_metric' : best_model_param['kernel_metric']
        }

        return result_in_json


    
    
    
    
    
    
############ LSTM ############
class LSTM_AD():
    def __init__(self, params, save_dir, is_in_memory_training=False, train_test_split_ratio=0.25,k=None):

        self.__model = Sequential()
        self.__configs = params
        self.__save_dir = save_dir
        self.__in_memory = is_in_memory_training
        self.__fold_val = k # used for cross_val_score or cross_validate
        self.__train_test_split_ratio = train_test_split_ratio

        self.__build_model() # build the model

    def __build_model(self):
        for layer in self.__configs['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.__model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.__model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.__model.add(Dropout(dropout_rate))

        self.__model.compile(loss=self.__configs['loss'], optimizer=self.__configs['optimizer'])

        print('[Model] Model Compiled')
    
    def __train_in_memory(self, X, y):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (self.__configs['epochs'], self.__configs['batch_size']))

        save_fname = os.path.join(self.__save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(self.__configs['epochs'])))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        self.__model.fit(
            X,
            y,
            epochs=self.__configs['epochs'],
            batch_size=self.__configs['batch_size'],
            callbacks=callbacks
        )
        self.__model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()
        
    
    def __train_out_memory(self, data_gen, steps_per_epoch): # train_generator
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (self.__configs['epochs'], self.__configs['batch_size'], steps_per_epoch))

        save_fname = os.path.join(self.__save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(self.__configs['epochs'])))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.__model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.__configs['epochs'],
            callbacks=callbacks,
            workers=1
        )

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.__model.predict(curr_frame[np.newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs
    
    def predict_sequence_full(self, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.__model.predict(curr_frame[np.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted
    
    def predict_point_by_point(self, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.__model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted
    def get_residual(self, real_y, pred_y, err_threshold):
        val = abs(real_y - pred_y)
#         err = np.full(real_y.shape, err_threshold)
        ret = []
        for i in range(len(real_y)):
            if val[i] <= err_threshold:
                ret.append(True)
            else:
                ret.append(False)
        return np.array(ret)
        
    def __get_result(self, real_y, pred_y, real_labels, err_threshold):
        real_y = np.reshape(real_y, pred_y.shape)
        generate_pred_labels = np.vectorize(lambda x: 1 if x==True else -1)

        residuals = self.get_residual(real_y, pred_y, err_threshold)
        
        pred_labels = generate_pred_labels(residuals)
        
        results = np.equal(real_labels, pred_labels)
        
        print (type(results))
        print (results)
        
        n_corrects = np.sum(results)
        n_error = len(results) - n_corrects
        
        error = n_error / len(results)
        accuracy = n_corrects / len(results)

        return error, accuracy

    def analyze(self, data, columns, label, err_threshold=0.001):
        
        # create LSTMDataLoader
        data_lstm = LSTMDataLoader(data, self.__train_test_split_ratio, columns, label)
        
        ##### TRAIN ######
        X_train, y_train = data_lstm.get_train_data(seq_len=self.__configs['data_sequence_len'],
                                                    normalise=self.__configs['normalize'])
        
        if (self.__in_memory):
            self.__train_in_memory(X_train,y_train)
        else:
            steps_per_epoch = (data_lstm.len_train - self.__configs['data_sequence_len']) / self.__configs['batch_size']
            steps_per_epoch = math.ceil(steps_per_epoch)
            
            data_gen=data_lstm.generate_train_batch(
                seq_len=self.__configs['data_sequence_len'],
                batch_size=self.__configs['batch_size'],
                normalise=self.__configs['normalize']
            )
            self.__train_out_memory(data_gen, steps_per_epoch)
            
#         train_predictions = self.predict_sequences_multiple(
#             y_train,
#             self.__configs['data_sequence_len'], 
#             self.__configs['data_sequence_len']
#         )
#         pred_y_train = self.predict_sequence_full(y_train, self.__configs['data_sequence_len'])

        pred_y_train = self.predict_point_by_point(X_train)
        real_labels_train = data.loc[:pred_y_train.shape[0],'label'].values
        error_train, accuracy_train = self.__get_result(y_train, pred_y_train, real_labels_train, err_threshold)
                          

        ##### TRAIN ######
        
        
        ##### TEST ######
        X_test, y_test = data_lstm.get_test_data(
            seq_len=self.__configs['data_sequence_len'],
            normalise=self.__configs['normalize']
        )
        
#         test_predictions = self.predict_sequences_multiple(
#             X_test, 
#             self.__configs['data_sequence_len'], 
#             self.__configs['data_sequence_len']
#         )
#         test_predictions = self.predict_sequence_full(X_test, self.__configs['data_sequence_len'])        

        pred_y_test = self.predict_point_by_point(X_test)
        real_labels_test = data.loc[pred_y_train.shape[0]:,'label'].values
        error_test, accuracy_test = self.__get_result(y_test, pred_y_test, real_labels_test, err_threshold)
        
        # predictions = model.predict_sequence_full(x_test, self.__configs['data_sequence_len'])
        # predictions = model.predict_point_by_point(x_test)
        ##### TEST ######
        
        
        ##### RESULT #####
        result_in_json = {
            'error_train': error_train,
            'accuracy_train' : accuracy_train,
            'error_test' : error_test,
            'accuracy_test' : accuracy_test,
            'y_pred' : test_predictions.tolist(),
            'y_true' : np.swapaxes(y_test,0,1).tolist(),
            'data_sequence_len' : self.__configs['data_sequence_len']
        }
        
        ##### RESULT #####
        
        return result_in_json

        
    from __future__ import division
#!/usr/bin/env python

import os
import math
import datetime as dt
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from helper.LSTMDataLoader import LSTMDataLoader
from helper.Timer import Timer

############ OneClassSVM ############
class OneClassSVM(object):

    def __init__(self, params, k):

        self.__svmKernelMetrics = params["kernel_metrics"]
        self.__gamma_list = params['gamma']
        self.__model = svm.OneClassSVM(nu=0.2, degree=3)
        self.__fold_val = k

    def __split(self, data, label):
        # separate the data into normal and abnormal data
        outliers = data[data[label]==-1]
        non_outliers = data[data[label]==1]

        return outliers, non_outliers


    def __train(self, train_data, columns=None, label=None, gamma_val=None, kernel_metric=None):
        # train the model
        if (gamma_val== None):
            gamma_val = self.__gamma_list[0]
        if (kernel_metric==None):
            kernel_metric=self.__svmKernelMetrics[0]

        self.__model = svm.OneClassSVM(nu=0.2, kernel=kernel_metric, gamma=gamma_val, degree=3)
        self.__model.fit(train_data[columns])


    def __get_test_error(self, test_data, columns, label):
        
        outliers_test, non_outliers_test = self.__split(test_data, label)

        y_pred_normal = self.__model.predict(non_outliers_test[columns])
        if (outliers_test.size <= 0):
            y_pred_abnormal = np.array([])
        else:
            y_pred_abnormal = self.__model.predict(outliers_test[columns])

        n_error_normal = y_pred_normal[y_pred_normal == -1].size
        n_error_outliers = y_pred_abnormal[y_pred_abnormal == 1].size
        n_errors = n_error_normal + n_error_outliers
        error = n_errors / test_data.shape[0]

        return error

    def __cross_validate(self,data, columns, label):
        kf = KFold(n_splits=self.__fold_val)

        min_error_avr = float("inf")
        best_model_param = {}
        for gamma_val in self.__gamma_list:
            for kernel_metric in self.__svmKernelMetrics:

                new_error_avr = 0
                for train_idx, test_idx in kf.split(data):
                    train, test = data.iloc[train_idx,:], data.iloc[test_idx,:]

                    outliers_train, non_outliers_train = self.__split(train, label)

                    n_outliers = outliers_train.count()

                    self.__train(non_outliers_train, columns=columns, gamma_val=gamma_val, kernel_metric=kernel_metric)

                    error_val = self.__get_test_error(test, columns, label)
                    new_error_avr = new_error_avr + error_val

                new_error_avr = new_error_avr / self.__fold_val

                if (new_error_avr < min_error_avr):
                    best_model_param['kernel_metric'] = kernel_metric
                    best_model_param['gamma_val'] = gamma_val
                    min_error_avr = new_error_avr

        return best_model_param


    def analyze(self, data, columns, label):
        results_in_json = {}

        best_model_param = self.__cross_validate(data, columns, label)

        outliers, non_outliers = self.__split(data, label)

        self.__train(non_outliers, columns, label, best_model_param['gamma_val'], best_model_param['kernel_metric'])

        y_pred_normal = self.__model.predict(non_outliers[columns])
        y_pred_abnormal = self.__model.predict(outliers[columns])

        n_correct_labels_normal = y_pred_normal[y_pred_normal==1].size
        n_correct_labels_abnormal = y_pred_abnormal[y_pred_abnormal==-1].size
        n_corrects = n_correct_labels_normal + n_correct_labels_abnormal

        n_error_normal = y_pred_normal[y_pred_normal == -1].size
        n_error_outliers = y_pred_abnormal[y_pred_abnormal == 1].size
        n_errors = n_error_normal + n_error_outliers

        error = n_errors / data.shape[0]
        accuracy = n_corrects / data.shape[0]

        result_in_json = {
            'error' : error,
            'accuracy' : accuracy,
            'gamma_val' : best_model_param['gamma_val'],
            'kernel_metric' : best_model_param['kernel_metric']
        }

        return result_in_json


    
    
    
    
    
    
############ LSTM ############
class LSTM_AD():
    def __init__(self, params, save_dir, is_in_memory_training=False, train_test_split_ratio=0.25,k=None):

        self.__model = Sequential()
        self.__configs = params
        self.__save_dir = save_dir
        self.__in_memory = is_in_memory_training
        self.__fold_val = k # used for cross_val_score or cross_validate
        self.__train_test_split_ratio = train_test_split_ratio

        self.__build_model() # build the model

    def __build_model(self):
        for layer in self.__configs['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.__model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.__model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.__model.add(Dropout(dropout_rate))

        self.__model.compile(loss=self.__configs['loss'], optimizer=self.__configs['optimizer'])

        print('[Model] Model Compiled')
    
    def __train_in_memory(self, X, y):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (self.__configs['epochs'], self.__configs['batch_size']))

        save_fname = os.path.join(self.__save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(self.__configs['epochs'])))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        self.__model.fit(
            X,
            y,
            epochs=self.__configs['epochs'],
            batch_size=self.__configs['batch_size'],
            callbacks=callbacks
        )
        self.__model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()
        
    
    def __train_out_memory(self, data_gen, steps_per_epoch): # train_generator
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (self.__configs['epochs'], self.__configs['batch_size'], steps_per_epoch))

        save_fname = os.path.join(self.__save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(self.__configs['epochs'])))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.__model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.__configs['epochs'],
            callbacks=callbacks,
            workers=1
        )

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.__model.predict(curr_frame[np.newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs
    
    def predict_sequence_full(self, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.__model.predict(curr_frame[np.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted
    
    def predict_point_by_point(self, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.__model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted
    def get_residual(self, real_y, pred_y, err_threshold):
        val = abs(real_y - pred_y)
#         err = np.full(real_y.shape, err_threshold)
        ret = []
        for i in range(len(real_y)):
            if val[i] <= err_threshold:
                ret.append(True)
            else:
                ret.append(False)
        return np.array(ret)
        
    def __get_result(self, real_y, pred_y, real_labels, err_threshold):
        print (pred_y.shape)
        print (real_labels.shape)
        real_y = np.reshape(real_y, pred_y.shape)
        generate_pred_labels = np.vectorize(lambda x: 1 if x==True else -1)

        residuals = self.get_residual(real_y, pred_y, err_threshold)
        
        pred_labels = generate_pred_labels(residuals)
        
        results = np.equal(real_labels, pred_labels)
        
        n_corrects = np.sum(results)
        n_error = len(results) - n_corrects
        
        error = n_error / len(results)
        accuracy = n_corrects / len(results)

        return error, accuracy

    def analyze(self, data, columns, label, err_threshold=0.001):
        
        # create LSTMDataLoader
        data_lstm = LSTMDataLoader(data, self.__train_test_split_ratio, columns, label)
        
        ##### TRAIN ######
        X_train, y_train = data_lstm.get_train_data(seq_len=self.__configs['data_sequence_len'],
                                                    normalise=self.__configs['normalize'])
        
        if (self.__in_memory):
            self.__train_in_memory(X_train,y_train)
        else:
            steps_per_epoch = (data_lstm.len_train - self.__configs['data_sequence_len']) / self.__configs['batch_size']
            steps_per_epoch = math.ceil(steps_per_epoch)
            
            data_gen=data_lstm.generate_train_batch(
                seq_len=self.__configs['data_sequence_len'],
                batch_size=self.__configs['batch_size'],
                normalise=self.__configs['normalize']
            )
            self.__train_out_memory(data_gen, steps_per_epoch)
            
#         train_predictions = self.predict_sequences_multiple(
#             y_train,
#             self.__configs['data_sequence_len'], 
#             self.__configs['data_sequence_len']
#         )
#         pred_y_train = self.predict_sequence_full(y_train, self.__configs['data_sequence_len'])

        pred_y_train = self.predict_point_by_point(X_train)
        real_labels_train = data.loc[1:pred_y_train.shape[0],'label'].values
        error_train, accuracy_train = self.__get_result(y_train, pred_y_train, real_labels_train, err_threshold)
                          

        ##### TRAIN ######
        
        
        ##### TEST ######
        X_test, y_test = data_lstm.get_test_data(
            seq_len=self.__configs['data_sequence_len'],
            normalise=self.__configs['normalize']
        )
        
#         test_predictions = self.predict_sequences_multiple(
#             X_test, 
#             self.__configs['data_sequence_len'], 
#             self.__configs['data_sequence_len']
#         )
#         test_predictions = self.predict_sequence_full(X_test, self.__configs['data_sequence_len'])        

        pred_y_test = self.predict_point_by_point(X_test)
        real_labels_test = data.loc[pred_y_train.shape[0]:pred_y_train.shape[0]+pred_y_test.shape[0]-1,'label'].values
        error_test, accuracy_test = self.__get_result(y_test, pred_y_test, real_labels_test, err_threshold)
        
        # predictions = model.predict_sequence_full(x_test, self.__configs['data_sequence_len'])
        # predictions = model.predict_point_by_point(x_test)
        ##### TEST ######
        
        
        ##### RESULT #####
        result_in_json = {
            'error_train': error_train,
            'accuracy_train' : accuracy_train,
            'error_test' : error_test,
            'accuracy_test' : accuracy_test,
            'y_pred' : pred_y_test.tolist(),
            'y_true' : np.swapaxes(y_test,0,1).tolist(),
            'data_sequence_len' : self.__configs['data_sequence_len']
        }
        
        ##### RESULT #####
        
        return result_in_json

        
    
    

    

