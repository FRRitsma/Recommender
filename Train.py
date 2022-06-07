#%%
import numpy as np
import shutil
import json
import random
import os
import gc
from LoadData import *
from SubFunctions import *

def SaveResults(savename, Y, xi, yi, rank, overwrite = True):
    savename = savename + 'Rank' + str(rank)
    if os.path.isdir(savename) and not overwrite:
        raise Exception(f"Save location {savename} allready exists")
    if os.path.isdir(savename):
        shutil.rmtree(savename)
    os.makedirs(savename)
    np.save(os.path.join(savename, "FitArray.npy"),Y)
    with open(os.path.join(savename, 'xIndex.json'), 'w') as f:
        json.dump(xi, f)
    with open(os.path.join(savename, 'yIndex.json'), 'w') as f:
        json.dump(yi, f)    

# UserSettings:
NoMovies = int(1.5e4)
NoUsers  = int(2e5)
name = 'BigFit2'

#%% Collect ratings for the N most popular movies:
SelectedMovies  = list(MovieFrequencies.keys())[0:NoMovies]
UserRatings     = GetMovieRatings(SelectedMovies,Lookup)

# Randomly select a given number of users:
UniqueUsers = set()
for v in UserRatings.values():
    UniqueUsers = UniqueUsers | set(v.keys())
UniqueUsers = set(UniqueUsers)
if len(UniqueUsers) < NoUsers:
    raise Exception(f"{NoUsers} users were requested, only {len(UniqueUsers)} were found")

UniqueUsers = list(UniqueUsers)
random.shuffle(UniqueUsers)
UniqueUsers = UniqueUsers[0:NoUsers]
UniqueUsers = set(UniqueUsers)

xi = {j:i for i,j in enumerate(UniqueUsers)}
yi = {j:i for i,j in enumerate(UserRatings)}

Array = np.zeros([len(UniqueUsers), NoMovies], dtype = np.uint8)
for i0, (movie, v) in enumerate(UserRatings.items()):
    print(f"Movie: {i0+1}/{len(UserRatings)}", end = '\r')
    for user in set(v.keys()) & UniqueUsers:
        Array[xi[user],yi[movie]] = int(v[user])

#%% Free up memory:
del UserRatings
del UniqueUsers
gc.collect()

#%% Functions for fitting data:
def FitX(X,Y,Array):
    for i in range(X.shape[0]):
        mask = Array[i,:] != 0
        X[i,:] = Array[i,mask]@np.linalg.pinv(Y[:,mask])

def FitY(X,Y,Array):
    for i in range(Y.shape[1]): 
        mask = Array[:,i] != 0
        Y[:,i] = np.linalg.pinv(X[mask,:])@Array[mask,i]

def Normalize(X,Y):
    norm = np.sum(abs(Y),axis=1)/Y.shape[1] 
    X = X*norm
    Y = Y/norm.reshape(-1,1)
    return X, Y

def Error(X,Y,Array):
    return np.sum(abs((X@Y)[Array!=0] - Array[Array!=0]))/np.sum(Array!=0)

# Fit arrays: 
dtp = np.float32
X   = np.empty([Array.shape[0],0]).astype(dtp)
Y   = np.empty([0,Array.shape[1]]).astype(dtp)
for rank in range(2,16):
    # User previous iteration as a hotstart:
    X   = np.hstack([(np.random.rand(Array.shape[0], rank - X.shape[1])-.5).astype(dtp), X]) 
    Y   = np.vstack([(np.random.rand(rank - Y.shape[0], Array.shape[1])-.5).astype(dtp), Y]) 
    # Clean up rank deficient rows/columns:
    while np.any(np.sum(Array != 0, axis = 1)<rank) or np.any(np.sum(Array != 0, axis = 0)<rank):
        KeepMask    = np.sum(Array != 0, axis = 1)>=rank
        Array       = Array[KeepMask,:]
        X           = X[KeepMask,:]
        xi          = {k:v for (k,v), b in zip(xi.items(), KeepMask) if b}
        KeepMask    = np.sum(Array != 0, axis = 0)>=rank
        Array       = Array[:,KeepMask]
        Y           = Y[:,KeepMask]
        yi          = {k:v for (k,v), b in zip(yi.items(), KeepMask) if b}
    # Fitting arrays to data:
    for iter in range(200):
        error = Error(X,Y,Array)
        print(f"Rank {rank}, Error: {error}",end='\r')
        FitX(X,Y,Array)
        FitY(X,Y,Array)
        X, Y = Normalize(X,Y)
    SaveResults(name, Y, xi, yi, rank)
    print('\n', end = '\r')
    gc.collect()

