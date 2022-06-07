#%%
import os
import numpy as np
import json
from cvxopt import matrix, solvers

# Implementation if function is accessed remotely:
def RemotePredict(rating:dict):
    # Find out what rank should be used:
    with open('./OptimalRank.json','r') as f:
        OptimalRank = json.load(f)
    inputrank   = min(len(rating),len(OptimalRank))
    optimalrank = OptimalRank[inputrank]
    # Open rank depending data:
    with open(f'./Rank{optimalrank}/yIndex.json','r') as f:
        yIndex = json.load(f)
    Y = np.load(f'./Rank{optimalrank}/FitArray.npy')
    _, predictions = Predict(Y,yIndex,rating)
    return predictions
    
# Direct implementation of predict function with all components provided:
def Predict(Y:np.ndarray, yIndex:dict, rating:dict):
    Y = Y.astype(np.float64)
    # Aggregating data:
    Ys = np.hstack([Y[:,yIndex[r]].reshape(-1,1) for r in rating])
    Rs = np.array([v for v in rating.values()]).reshape(1,-1)
    # Objective function:
    P = Ys@Ys.T
    q = -Ys@Rs.T
    # Constraints:
    G = np.vstack([Y.T,-Y.T])
    h = np.vstack([np.ones([Y.shape[1],1])*5, -np.ones([Y.shape[1],1]) ]) 
    # Perform opimization:
    sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    x = np.array(sol['x']).T
    # Fill predictions for all movies:
    predictions = {u:r for u, r in zip(yIndex,(x@Y).reshape(-1))}
    # Overwrite rated movies:
    for k,v in rating.items():
        predictions[k] = v
    return x, predictions
    