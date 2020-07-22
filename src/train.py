#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 * GeM - Gait-phase Estimation Module
 *
 * Copyright 2018-2020 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Foundation for Research and Technology Hellas (FORTH) 
 *	     nor the names of its contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
'''

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

	config = load_config(sys.argv[1])
	path = config['gem_train_path']
	gt = GeM_tools(gt_comparison=config['gem_gt_comparison'], gem2=config['gem2'], useLabels = config['useLabels'])

	gt.input_data(path)

	g = GeM()
	g.setFrames(config['gem_lfoot_frame'], config['gem_rfoot_frame'])
	g.setDimReduction(config['gem_dim'])
	data_train = gt.data_train
	data_labels = gt.data_label
	g.fit(data_train, config['gem_dim_reduction'], config['gem_clustering'], data_labels)
	plt.plot(data_labels[:,6])
	plt.plot(g.reduced_data_train[:,0].T*data_labels[:,0] + g.reduced_data_train[:,1].T*data_labels[:,3])
	plt.show()
	plt.plot(data_labels[:,7])
	plt.plot(g.reduced_data_train[:,0].T*data_labels[:,1] + g.reduced_data_train[:,1].T*data_labels[:,4])
	plt.show()
	plt.plot(data_labels[:,8])
	plt.plot(g.reduced_data_train[:,0].T*data_labels[:,2] + g.reduced_data_train[:,1].T*data_labels[:,5])
	plt.show()



	if(config['gem_plot_results']):
		if(config['gem_gt_comparison']):
			gt.genGroundTruthStatistics(g.reduced_data_train)
			gt.plot_results(g.reduced_data_train, gt.phase, gt.mean, gt.covariance, 'Ground-Truth Labels')
			gt.plot_latent_space(g)
			cnf_matrix = confusion_matrix(gt.phase,  g.predicted_labels_train)
			np.set_printoptions(precision=2)
			class_names = ['DS','LSS','RSS']
			gt.plot_confusion_matrix(cnf_matrix, class_names, 'Confusion matrix')

		if(config['gem_clustering'] == "kmeans"):
			gt.plot_results(g.reduced_data_train, g.predicted_labels_train, g.kmeans.cluster_centers_, None, 'Clustering with K-means')
		elif(config['gem_clustering'] == "gmm"):
			gt.plot_results(g.reduced_data_train, g.predicted_labels_train, g.gmm.means_, g.gmm.covariances_, 'Clustering with Gaussian Mixture Models')
		else:
			print("Unsupported Result Plotting")






	if(config['gem_save']):
		pickle.dump(g, open(path+'/gem_train.save','wb'))
		pickle.dump(gt, open(path+'/gem_train_tools.save','wb'))


	print('Training Finished')
