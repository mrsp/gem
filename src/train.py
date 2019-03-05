#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gem import GeM
from gem_tools import GeM_tools
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
import sys

import yaml


def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)




if __name__ == "__main__":

	config = load_config('../config/gem_params.yaml')
	print(config)

	path = config['gem_train_path']
	gt = GeM_tools(comp_filtering=config['gem_cf'], freq=config['gem_imu_freq'], a=config['gem_cf_a'], gt_comparison=config['gem_gt_comparison'], mu=1.0)
	brf = np.zeros(3)
	brt = np.zeros(3)
	blf = np.zeros(3)
	blt = np.zeros(3)
	blf[2] = -19.1734
	brf[2] = -17.1439


	
	
	gt.input_data(path,blf,blt,brf,brt)

	g = GeM()
	data_train = gt.data_train
	g.fit(data_train, config['gem_dim_reduction'], config['gem_clustering'])


	if(config['gem_plot_results']):
		if(config['gem_gt_comparison']):
			gt.genGroundTruthStatistics(g.reduced_data_train)
			gt.plot_results(g.reduced_data_train, gt.phase, gt.mean, gt.covariance, 'Ground-Truth Labels')


		if(config['gem_clustering'] == "kmeans"):
			gt.plot_results(g.reduced_data_train, g.predicted_labels_train, g.kmeans.cluster_centers_, None, 'Clustering with K-means')
		elif(config['gem_clustering'] == "gmm"):
			gt.plot_results(g.reduced_data_train, g.predicted_labels_train, g.gmm.means_, g.gmm.covariances_, 'Clustering with Gaussian Mixture Models')
		else:
			print("Unsupported Result Plotting")


		gt.plot_latent_space(g)
		cnf_matrix = confusion_matrix(gt.phase,  g.predicted_labels_train)
		np.set_printoptions(precision=2)
		class_names = ['DS','LSS','RSS']
		gt.plot_confusion_matrix(cnf_matrix, class_names, 'Confusion matrix')


	if(config['gem_save']):
		pickle.dump(g, open('gem_train.save','wb'))
		pickle.dump(gt, open('gem_train_tools.save','wb'))


	print('Training Finished')
