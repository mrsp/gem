#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gem import GeM
from gem_tools import GeM_tools
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
import sys




if __name__ == "__main__":


	path = str(sys.argv[1])
	gt = GeM_tools(comp_filtering=True, freq=500, a=0.995, gt_comparison=True, mu=1.0)
	brf = np.zeros(3)
	brt = np.zeros(3)
	blf = np.zeros(3)
	blt = np.zeros(3)
	blf[2] = -19.1734
	brf[2] = -17.1439


	if path == " ":
		path ="/home/user/GEM_test_valkyrie/"
	
	gt.input_data(path,blf,blt,brf,brt)

	g = GeM()
	data_train = gt.data_train
	g.fit(data_train, 'autoencoders', 'gmm')





	gt.genGroundTruthStatistics(g.reduced_data_train)
	gt.plot_results(g.reduced_data_train, gt.phase, gt.mean, gt.covariance, 'Ground-Truth Labels')


	#gt.plot_results(g.reduced_data_train, g.predicted_labels_train, g.kmeans.cluster_centers_, None, 'Clustering with K-means')


	gt.plot_results(g.reduced_data_train, g.predicted_labels_train, g.gmm.means_, g.gmm.covariances_, 'Clustering with Gaussian Mixture Models')


	gt.plot_latent_space(g)
	cnf_matrix = confusion_matrix(gt.phase,  g.predicted_labels_train)
	np.set_printoptions(precision=2)
	class_names = ['DS','LSS','RSS']
	gt.plot_confusion_matrix(cnf_matrix, class_names, 'GMM Confusion matrix')



	pickle.dump(g, open('gem_train.save','wb'))
	pickle.dump(gt, open('gem_train_tools.save','wb'))

	print('Training Finished')





