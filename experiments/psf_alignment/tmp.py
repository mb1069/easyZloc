#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import tqdm

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV


def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


param_grid = {
    "n_components": range(1, 3),
    "covariance_type": ["tied"],
}

def get_n_components(data):
    data = data[:, np.newaxis]
    
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
    )
    grid_search.fit(data)
    df = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "param_covariance_type", "mean_test_score"]
    ]
    df['mean_test_score'] = -df['mean_test_score']
    
    n_components = df.sort_values(by="mean_test_score").head(1)['param_n_components'].to_list()[0]
    score_diff = df['mean_test_score'].max() - df['mean_test_score'].min()
    return n_components, score_diff

def run_experiment(params):
    sep, sd, sample_size, rep = params
    p1 = np.random.normal(0, sd, size=sample_size//2)
    p2 = np.random.normal(sep, sd, size=sample_size//2)
    data = np.concatenate((p1, p2))
    n_components, score_diff = get_n_components(data)
    
    return *params, n_components, score_diff


seps = [50]
stdevs = np.linspace(2.5, 50, 50)
sample_sizes = np.logspace(1, 2, 50).astype(int)
repeats = np.arange(100)
total = len(seps) * len(stdevs) * len(sample_sizes) * len(repeats)

res = []

params = list(itertools.product(seps, stdevs, sample_sizes, repeats))

# from multiprocessing import Pool
# with Pool(1) as p:
#     res = list(tqdm.tqdm(p.imap(run_experiment, params), total=len(params)))

res = list(tqdm.tqdm(map(run_experiment, params), total=len(params)))
# for sep, stdev, s_size in tqdm(, total=total):
#     print(sep, stdev, s_size)
#     n_components, score_diff = run_experiment(*param)
#     res.append([sep, stdev, s_size, n_components, score_diff])


# In[151]:


df = pd.DataFrame(res, columns=['seperation', 'stdev', 'sample_size', 'rep', 'bic_n_components', 'bic_score_diff'])


# In[152]:


df.to_csv('./hist_experiments.csv')


