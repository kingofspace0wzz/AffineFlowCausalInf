## Exploratory analysis and some causal discovery/etc on interventional fMRI data
#
#
#
#

import numpy as np 
import pandas as pd
import os 
import pylab as plt; plt.ion()
import seaborn as sns
from sklearn.preprocessing import scale

os.chdir('/Users/ricardo/Documents/Projects/AffineFlowCausalInf/Application/data')

# load some data in to see whats going on

labels = np.array( pd.read_table('labels.txt') ).squeeze()

# 
# from Robs notes, looks like:
#
# Run1 Cingulate Gyrus anterior division
# Run2 No notes (maybe no stimulation)
# Run3 Cingulate Gyrus anterior division
# Run4 Heschl's Gyrus 
# Run5 Central Opercular Cortex 


# lets load in data for 

dat = [np.array(pd.read_csv(f"Run{i}.txt", header=0)) for i in range(1,6)]

scale_dat = [scale(d, with_mean=True) for d in dat]

intervention_regions = ['Cingulate Gyrus, anterior division', None, 
						'Cingulate Gyrus, anterior division', "Heschl's Gyrus (includes H1 and H2)",
						'Central Opercular Cortex']


interventional_id = [np.argmax(labels==i) if i is not None else None for i in intervention_regions]

rois = np.unique([roi for roi in interventional_id if roi is not None])
rois = np.array([28,44]) # just take 2 ROIs for now

# plot resting state data
int_dset = 2 # interventional dataset
df_12 = pd.DataFrame(np.vstack((scale_dat[1][:, rois], 
								scale_dat[3][:, rois],
								scale_dat[2][:, rois])))
df_12.columns = labels[rois]
df_12['type'] = ['rest'] * 253 + ['intvene cingulate'] * 253 + ['intvene Heschl'] * 253


sns.pairplot(df_12, kind='scatter', diag_kind='kde', hue='type', markers=["o", "s", "D"]) 

# perform causal discovery using the resting dataset
os.chdir('../../')
from models.affine_flow_cd import BivariateFlowLR

train_dat = scale_dat[1][:, rois]

np.random.seed(1)
nlayer = 5
n_hidden = 1 # 2
mod = BivariateFlowLR(n_layers=[nlayer], n_hidden=[n_hidden], split=.8, opt_method='scheduling')
p, dir = mod.predict_proba(data=train_dat) # predicts 0 -> 1, no need to reorder

mod.fit_to_sem(train_dat, n_layers=nlayer, n_hidden=n_hidden)

intervene_dat = scale_dat[2][:, rois]
xvals = intervene_dat[:,0]

int_pred = np.array([mod.predict_intervention(x0_val=x, n_samples=500)[1] for x in xvals])


## compare with GPs and Linear Regression

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X=train_dat[:,0].reshape((-1,1)), y=train_dat[:,1])


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
kernel = 1.0 * RBF(1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=25, n_restarts_optimizer=10)

gp.fit(X=train_dat[:,0].reshape((-1,1)), y=train_dat[:,1])


results_df = pd.DataFrame({'carefl': np.abs((int_pred)-(intervene_dat[:,1])),
						   'linear_regression': np.abs( (lm.predict(xvals.reshape((-1,1)))) - (intervene_dat[:,1])),
						   'gaussian_process': np.abs( (gp.predict(xvals.reshape((-1,1)))) - (intervene_dat[:,1]))})


results_df.median(axis=0)


