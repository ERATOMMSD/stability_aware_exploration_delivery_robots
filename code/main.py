#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy

from helper import *
#Either import helper file or run:
from jmetal.core.solution import Solution
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.core.quality_indicator import HyperVolume

import matplotlib.colors as mcolors

from sklearn import metrics
#from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize
from scipy.spatial import distance_matrix

import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
#STDV-BASED and
#STDV-DIST-BASED
parser.add_argument('--data_dir', default='data', type=str,help="path to Pareto Front data")
parser.add_argument('--results_dir', default='results', type=str,help="path to Pareto Front data")
parser.add_argument("--K",default=0.5,type=float,help="parameter for K-overlapping of standar deviations")
parser.add_argument("--eps",default=0.01,type=float,help="epsilon parameter for STDV-DIST-BASED method")
args = parser.parse_args()
DATA_DIR=os.path.abspath(".").replace('\\','/') + '/' + args.data_dir
RES_DIR=os.path.join(args.results_dir)

if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

with open(RES_DIR+'/results.txt', 'w') as f:
    f.write('Pareto Front Pruning Results\n\n')


#Data imports and formatting
sd = args.K
eps = args.eps

dr_bound_g, ur_bound_g, nr_bound_g = None, None, None

rec_ind = pd.read_csv(DATA_DIR)

img_path = RES_DIR+"/plots/"
hv_name = "K_" + str(sd) +  "_epsilon_" + str(eps) + ":"
#filter invalid objective values:
rec_ind=rec_ind.loc[(rec_ind['utilization_rate'] >= 0)\
                              & (rec_ind['delivery_rate'] >= 0) &\
                                  (rec_ind['num_risks'] < 10000)]

#obtain variances across requests for each objective value
drate_str = [s for s in rec_ind.columns if ('delivery_rate' in s and s!='delivery_rate')]
urate_str = [s for s in rec_ind.columns if ('utilization_rate' in s and s!='utilization_rate')]
nrisk_str = [s for s in rec_ind.columns if ('num_risks' in s and s!='num_risks')]
drate_var = np.std(rec_ind[drate_str], axis=1)**2
urate_var = np.std(rec_ind[urate_str], axis=1)**2
nrisk_var = np.std(rec_ind[nrisk_str], axis=1)**2

#obtain pareto front based on the original objectives
Xdf = rec_ind[['delivery_rate', 'utilization_rate', 'num_risks']]
dr_bound, ur_bound, nr_bound=np.min(Xdf['delivery_rate']), np.min(Xdf['utilization_rate']), np.max(Xdf['num_risks'])
if dr_bound_g is None or dr_bound_g>dr_bound:
    dr_bound_g=dr_bound
if ur_bound_g is None or ur_bound_g>ur_bound:
    ur_bound_g=ur_bound
if nr_bound_g is None or nr_bound_g>nr_bound:
    nr_bound_g=nr_bound
X=get_solutions_as_numpy_array(rec_ind)
obj = get_non_dom_from_numpy(X)

#compute an overall variance as the sum of normalized variances for each objective feature
overall_var = drate_var/drate_var.abs().max()\
                + urate_var/urate_var.abs().max()\
                + nrisk_var/nrisk_var.abs().max()

#add individual and overall variances across requests to each solution
Xdf['drate_var'] = drate_var
Xdf['urate_var'] = urate_var
Xdf['nrisk_var'] = nrisk_var
Xdf['overall_var'] = overall_var

X_vars = np.hstack((drate_var.to_numpy().reshape(-1,1),
                urate_var.to_numpy().reshape(-1,1),
                nrisk_var.to_numpy().reshape(-1,1)))
X_aug= np.hstack((X,X_vars))

#obtain "augmented" pareto front when considering the variance of each objective as another objective
obj_aug = get_non_dom_from_numpy(X_aug)

#get indices for original and augmented pareto fronts
pareto_front_ind = [np.where(np.all(np.equal(X, obj[i]),
                                    axis=1))[0][0] for i in range(obj.shape[0])]

pareto_front_ind_aug = [np.where(np.all(np.equal(X_aug, obj_aug[i]),
                                    axis=1))[0][0] for i in range(obj_aug.shape[0])]

#use only solutions in the original pareto front to obtain those with non-dominated variances
X_var_p = X_vars[pareto_front_ind]
obj_w_var = get_non_dom_from_numpy(X_var_p)

#get indices for this new post-processed pareto front
pareto_front_ind_var = [np.where(np.all(np.equal(X_var_p, obj_w_var[i]),
                                    axis=1))[0][0] for i in range(obj_w_var.shape[0])]

pareto_front_ind_var=list(np.array(pareto_front_ind)[pareto_front_ind_var])

#get all three versions of the pareto front as np arrays
pareto_front = Xdf.iloc[pareto_front_ind].to_numpy()
pareto_front_aug = Xdf.iloc[pareto_front_ind_aug].to_numpy()
pareto_front_var = Xdf.iloc[pareto_front_ind_var].to_numpy()

pareto_front_scaled=normalize(pareto_front, axis = 0)
norm_variances = [(np.std(normalize(rec_ind[drate_str], axis=1), axis=1)**2)[pareto_front_ind],
                  (np.std(normalize(rec_ind[urate_str],axis=1), axis=1)**2)[pareto_front_ind],
                  (np.std(normalize(rec_ind[nrisk_str], axis=1), axis=1)**2)[pareto_front_ind]]
for c in range(3):
    pareto_front_scaled[:,c+3] = norm_variances[c]

"""
STDV_BASED
"""
#NEXT ALGO: INITIAL NAIVE CLUSTERING
q=sd#5e-1
db = STDV_BASED(data=pareto_front_scaled, metric=metric_maker(varmethod=2, q=q))
labels=get_labels(pareto_front_scaled, db)
title='STDV_BASED'
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#print("Estimated number of clusters (" + title + "): %d" % n_clusters_)
ax=plt.axes(projection="3d")
for l in set(labels):
    inds=np.where(labels==l)[0]

    ax.scatter(pareto_front[inds,0], pareto_front[inds,1], pareto_front[inds,2],
               label='cluster ' + str(l))

plt.xlabel('del. rate')
plt.ylabel('util. rate')
ax.set_zlabel('num. risks')
plt.title('Clustering (' + title + ')')
plt.legend(bbox_to_anchor=(0,0.5), loc="center right")
dir = img_path + 'STDV_BASED/'
if not os.path.exists(dir):
    os.makedirs(dir)
plt.savefig(dir + title.replace(' ', '_') + '.png')

ax=plt.axes(projection="3d")
rr=[]
for l in set(labels):
    inds=np.where(labels==l)[0]
    rep_ind = inds[np.argmin(pareto_front[inds,-1])]
    ax.scatter(pareto_front[rep_ind,0], pareto_front[rep_ind,1], pareto_front[rep_ind,2],
               label='cluster ' + str(l))
    rr.append(rep_ind)
stdv_front=pareto_front[rr,:3]

plt.xlabel('del. rate')
plt.ylabel('util. rate')
ax.set_zlabel('num. risks')
plt.title('Stability based rep. for Clustering (' + title + ')')
plt.legend(bbox_to_anchor=(0,0.5), loc="center right")
dir = img_path + 'STDV_BASED/'
plt.savefig(dir + title.replace(' ', '_') + '_reps.png')

"""
STDV_DIST_BASED
"""
q=sd
db = STDV_DIST_BASED(data=pareto_front_scaled, eps=eps, min_samples=1,
                        metric=metric_maker(varmethod=2, q=q))
labels=db.labels_
title='STDV_DIST_BASED'

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#print("Estimated number of clusters (" + title + "): %d" % n_clusters_)

ax=plt.axes(projection="3d")
for l in set(labels):
    inds=np.where(labels==l)[0]

    ax.scatter(pareto_front[inds,0], pareto_front[inds,1], pareto_front[inds,2],
               label='cluster ' + str(l))

plt.xlabel('del. rate')
plt.ylabel('util. rate')
ax.set_zlabel('num. risks')
plt.title('Clustering (' + title + ')')
plt.legend(bbox_to_anchor=(0,0.5), loc="center right")
dir = img_path + 'STDV_DIST_BASED_epsilon_' + str(eps) + "/"
if not os.path.exists(dir):
    os.makedirs(dir)
plt.savefig(dir + title.replace(' ', '_') + '.png')

ax=plt.axes(projection="3d")
rr=[]
for l in set(labels):
    inds=np.where(labels==l)[0]
    rep_ind = inds[np.argmin(pareto_front[inds,-1])]
    ax.scatter(pareto_front[rep_ind,0], pareto_front[rep_ind,1], pareto_front[rep_ind,2],
               label='cluster ' + str(l))
    rr.append(rep_ind)

stdv_dist_front=pareto_front[rr,:3]
plt.xlabel('del. rate')
plt.ylabel('util. rate')
ax.set_zlabel('num. risks')
plt.title('Stability based rep. for Clustering (' + title + ')')
plt.legend(bbox_to_anchor=(0,0.5), loc="center right")
dir = img_path + 'STDV_DIST_BASED_epsilon_' + str(eps) + "/"
plt.savefig(dir + title.replace(' ', '_') + '_reps.png')

#Comparing pruned fronts
hv_lines = ['\n' + hv_name]
for name, front in zip(['original', 'STDV_BASED', 'STDV_DIST_BASED_epsilon_' + str(eps)],
                        [pareto_front, stdv_front, stdv_dist_front]):
    if name=='original':
        pareto_front_mod=np.concatenate((-front[:,:2], front[:,2].reshape(-1,1)), axis=1)
        hv=HyperVolume(reference_point=np.array([0,0,1e5])).compute(pareto_front_mod)
        hvo=hv
        phrase= name + '_hv_' + str(hv) + "_points_" + str(len(front))
        hv_lines.append(phrase)

    else:
        pareto_front_mod=np.concatenate((-front[:,:2], front[:,2].reshape(-1,1)), axis=1)
        hv=HyperVolume(reference_point=np.array([0,0,1e5])).compute(pareto_front_mod)
        phrase= name +'_hv_' + str(hv) + "_points_" + str(len(front))
        hv_lines.append(phrase)

with open(RES_DIR+'/results.txt', 'a') as f:
    f.write('\n'.join(hv_lines))
