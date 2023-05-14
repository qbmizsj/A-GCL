import numpy as np
from sklearn.utils import shuffle
import torch
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader

import matplotlib.pyplot as plt
import os
import matplotlib
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

from scipy import interp

import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import time

def get_emb_y(loader, encoder, device, dtype='numpy', is_rand_label=False):
	# train_emb, train_y
	x, y = encoder.get_embeddings(loader, device, is_rand_label)

	if dtype == 'numpy':
		return x,y
	elif dtype == 'torch':
		return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
	else:
		raise NotImplementedError

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def sensitivity(y_pred, y_true):
	CM = confusion_matrix(y_true, y_pred) 

	tn_sum = CM[0, 0] # True Negative
	fp_sum = CM[0, 1] # False Positive

	tp_sum = CM[1, 1] # True Positive
	fn_sum = CM[1, 0] # False Negative
	Condition_negative = tp_sum + fn_sum + 1e-6
	sensitivity = tp_sum / Condition_negative

	return sensitivity

def specificity(y_pred, y_true):
	CM = confusion_matrix(y_true, y_pred) 

	tn_sum = CM[0, 0] # True Negative
	fp_sum = CM[0, 1] # False Positive

	tp_sum = CM[1, 1] # True Positive
	fn_sum = CM[1, 0] # False Negative

	Condition_negative = tn_sum + fp_sum + 1e-6
	Specificity = tn_sum / Condition_negative

	return Specificity


class EmbeddingEvaluation():
	def __init__(self, base_classifier, evaluator, task_type, num_tasks, device, params_dict=None, param_search=True,is_rand_label=False):
		self.is_rand_label = is_rand_label
		self.base_classifier = base_classifier
		self.evaluator = evaluator
		self.eval_metric = evaluator.eval_metric
		self.task_type = task_type
		self.num_tasks = num_tasks
		self.device = device
		self.param_search = param_search
		self.params_dict = params_dict
		if self.eval_metric == 'rmse':
			self.gscv_scoring_name = 'neg_root_mean_squared_error'
		elif self.eval_metric == 'mae':
			self.gscv_scoring_name = 'neg_mean_absolute_error'
		elif self.eval_metric == 'rocauc':
			self.gscv_scoring_name = 'roc_auc'
		elif self.eval_metric == 'accuracy':
			self.gscv_scoring_name = 'accuracy'
		else:
			raise ValueError('Undefined grid search scoring for metric %s ' % self.eval_metric)

		self.classifier = None
	def scorer(self, y_true, y_raw):

		input_dict = {"y_true": y_true, "y_pred": y_raw}
		score = self.evaluator.eval(input_dict)[self.eval_metric]
		return score

	def ee_binary_classification(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):
		if self.param_search:
			params_dict = {'C': [0.001, 0.01,0.1,1,10,100,1000]}
			self.classifier = make_pipeline(StandardScaler(),
			                                GridSearchCV(self.base_classifier, params_dict, cv=5, scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
			                                )
		else:
			self.classifier = make_pipeline(StandardScaler(), self.base_classifier)

		if np.isnan(train_emb).any():
			print("Has NaNs ... ignoring them")
			train_emb = np.nan_to_num(train_emb)
		
		if np.isnan(val_emb).any():
			print("Has NaNs ... ignoring them")
			val_emb = np.nan_to_num(val_emb)
		if np.isnan(test_emb).any():
			print("Has NaNs ... ignoring them")
			test_emb = np.nan_to_num(test_emb)
			
		self.classifier.fit(train_emb, np.squeeze(train_y))

		if self.eval_metric == 'accuracy':
			train_raw = self.classifier.predict(train_emb)
			val_raw = self.classifier.predict(val_emb)
			test_raw = self.classifier.predict(test_emb)
		else:
			train_raw = self.classifier.predict_proba(train_emb)[:, 1]
			val_raw = self.classifier.predict_proba(val_emb)[:, 1]
			test_raw = self.classifier.predict_proba(test_emb)[:, 1]

		return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1), np.expand_dims(test_raw, axis=1)

	def ee_multioutput_binary_classification(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):

		params_dict = {
			'multioutputclassifier__estimator__C': [1e-1, 1e0, 1e1, 1e2]}
		self.classifier = make_pipeline(StandardScaler(), MultiOutputClassifier(
			self.base_classifier, n_jobs=-1))
		
		if np.isnan(train_y).any():
			print("Has NaNs ... ignoring them")
			train_y = np.nan_to_num(train_y)
		self.classifier.fit(train_emb, train_y)

		train_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(train_emb)])
		val_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(val_emb)])
		test_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(test_emb)])

		return train_raw, val_raw, test_raw

	def ee_regression(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):
		if self.param_search:
			params_dict = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]}
# 			params_dict = {'alpha': [500, 50, 5, 0.5, 0.05, 0.005, 0.0005]}
			self.classifier = GridSearchCV(self.base_classifier, params_dict, cv=5,
			                          scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
		else:
			self.classifier = self.base_classifier

		self.classifier.fit(train_emb, np.squeeze(train_y))

		train_raw = self.classifier.predict(train_emb)
		val_raw = self.classifier.predict(val_emb)
		test_raw = self.classifier.predict(test_emb)

		return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1), np.expand_dims(test_raw, axis=1)

	def embedding_evaluation(self, encoder, train_loader, valid_loader, test_loader, flag):
		encoder.eval()
		if flag:
			val_start = time.time()
		train_emb, train_y = get_emb_y(train_loader, encoder, self.device, is_rand_label=self.is_rand_label)
		val_emb, val_y = get_emb_y(valid_loader, encoder, self.device, is_rand_label=self.is_rand_label)
		test_emb, test_y = get_emb_y(test_loader, encoder, self.device, is_rand_label=self.is_rand_label)
		if flag:
			val_end = time.time()
			running_time = val_end-val_start
			print('validation time cost : %.5f sec' %running_time)

		if 'classification' in self.task_type:

			if self.num_tasks == 1:
				train_raw, val_raw, test_raw = self.ee_binary_classification(train_emb, train_y, val_emb, val_y, test_emb,
				                                                        test_y)
			elif self.num_tasks > 1:
				train_raw, val_raw, test_raw = self.ee_multioutput_binary_classification(train_emb, train_y, val_emb, val_y,
				                                                                    test_emb, test_y)
			else:
				raise NotImplementedError
		else:
			if self.num_tasks == 1:
				train_raw, val_raw, test_raw = self.ee_regression(train_emb, train_y, val_emb, val_y, test_emb, test_y)
			else:
				raise NotImplementedError
		

		train_score = self.scorer(train_y, train_raw)
		val_score = self.scorer(val_y, val_raw)
		test_score = self.scorer(test_y, test_raw)

		train_sen_score = sensitivity(train_raw, train_y)
		val_sen_score = sensitivity(val_raw, val_y)
		test_sen_score = sensitivity(test_raw, test_y)

		train_spe_score = specificity(train_raw, train_y)
		val_spe_score = specificity(val_raw, val_y)
		test_spe_score = specificity(test_raw, test_y)

		train_f1_score = f1_score(train_y, train_raw)
		val_f1_score = f1_score(val_y, val_raw)
		test_f1_score = f1_score(test_y, test_raw)


		return train_score, val_score, test_score, train_f1_score, val_f1_score, test_f1_score, train_sen_score, val_sen_score, test_sen_score, train_spe_score, val_spe_score, test_spe_score, running_time																			

	def kf_embedding_evaluation(self, encoder, dataset, folds=10, batch_size=128, flag=False):
		kf_train = []
		kf_val = []
		kf_test = []
		kf_train_f1 = []
		kf_val_f1 = []
		kf_test_f1 = []
		kf_train_sen = []
		kf_val_sen = []
		kf_test_sen = []
		kf_train_spe = []
		kf_val_spe = []
		kf_test_spe = []
		running_times = []
		
		kf = KFold(n_splits=folds, shuffle=True, random_state=None)
		for k_id, (train_val_index, test_index) in enumerate(kf.split(dataset)):
			test_id.append(test_index)

			test_dataset = [dataset[int(i)] for i in list(test_index)]
			train_index, val_index = train_test_split(train_val_index, test_size=0.2, random_state=None)

			train_dataset = [dataset[int(i)] for i in list(train_index)]
			val_dataset = [dataset[int(i)] for i in list(val_index)]

			train_loader = DataLoader(train_dataset, batch_size=batch_size)
			valid_loader = DataLoader(val_dataset, batch_size=batch_size)
			test_loader = DataLoader(test_dataset, batch_size=batch_size)

			# embedding_evaluation -> get_emb_y -> encoder.get_embeddings -> forward
			train_score, val_score, test_score, train_f1, val_f1, test_f1, train_sen, val_sen, test_sen, train_spe, val_spe, test_spe, fpr, tpr, running_time= self.embedding_evaluation(encoder, 
	
			running_times.append(running_time)
	
			kf_train_f1.append(train_f1)
			kf_val_f1.append(val_f1)
			kf_test_f1.append(test_f1)
	
			kf_train_spe.append(train_spe)
			kf_val_spe.append(val_spe)
			kf_test_spe.append(test_spe)

			kf_train.append(train_score)
			kf_val.append(val_score)
			kf_test.append(test_score)

			kf_train_sen.append(train_sen)
			kf_val_sen.append(val_sen)
			kf_test_sen.append(test_sen)

		mean_time = np.array(running_times).mean()
		print("mean validation time %.5f:\n"% mean_time)

		kf_train_ms = [np.array(kf_train).mean(), np.array(kf_train).std(), np.array(kf_train_f1).mean(),
						np.array(kf_train_f1).std(),
						np.array(kf_train_sen).mean(), np.array(kf_train_sen).std(), np.array(kf_train_spe).mean(),
						np.array(kf_train_spe).std()]
		kf_val_ms = [np.array(kf_val).mean(), np.array(kf_val).std(), np.array(kf_val_f1).mean(),
						np.array(kf_val_f1).std(),
						np.array(kf_val_sen).mean(), np.array(kf_val_sen).std(), np.array(kf_val_spe).mean(),
						np.array(kf_val_spe).std()]
		kf_test_ms = [np.array(kf_test).mean(), np.array(kf_test).std(), np.array(kf_test_f1).mean(),
						np.array(kf_test_f1).std(),
						np.array(kf_test_sen).mean(), np.array(kf_test_sen).std(), np.array(kf_test_spe).mean(),
						np.array(kf_test_spe).std()]

		return kf_train_ms, kf_val_ms, kf_test_ms

