#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
 * GeM - Gait-phase Estimation Module
 *
 * Copyright 2018-2019 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
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
 *	 nor the names of its contributors may be used to endorse or promote products derived from
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
	gt = GeM_tools(comp_filtering=config['gem_cf'], freq=config['gem_freq'], a=config['gem_cf_a'], gt_comparison=config['gem_gt_comparison'])
	brf = config['gem_rfoot_force_bias']
	brt = config['gem_rfoot_torque_bias']
	blf = config['gem_lfoot_force_bias']
	blt = config['gem_lfoot_torque_bias']
	
	gt.input_data(path,blf,blt,brf,brt)

	g = GeM()
	g.setFrames(config['gem_lfoot_frame'], config['gem_rfoot_frame'])
	g.setDimReduction(config['gem_dim'])
	data_train = gt.data_train
	g.fit(data_train, config['gem_dim_reduction'], config['gem_clustering'])


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
