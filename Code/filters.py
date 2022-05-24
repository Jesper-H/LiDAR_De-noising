import numpy as np
from sklearn.neighbors import KDTree
from itertools import permutations
from numpy import linalg as LA

class iterate_filter(object):
    def __init__(self, algoritm, times:int):
        self.times = times
        self.algoritm = algoritm
        
    def __call__(self, P, **kwargs):
        mask  = np.array([False for _ in range(len(P))])
        index = np.array([i for i in range(len(P))])
        for _ in range(self.times):
            m = self.algoritm(P[~mask], **kwargs)
            ind = index[~mask]
            mask[ind[m]] = True
            
        return mask
        

def DSOR(P, Sfactor:float=0.0008, r:float=0.05, k:int=5, leafSize:int=10):
    """
    Implementation of Dynamic Statistical Outlier Removal algorithm
    
    Inputs:
    P = Point cloud dataset ((x,y,z).....)
    Kneighbour = minimum number of nearest neighbors
    Sfactor = multiplication factor for standard deviation
    r = multiplication factor for range
    
    Returns:
    Mask where 1 means outlier
    """
    tree = KDTree(P[:, :3], leaf_size=leafSize)  #Create a KD tree
    dist, i = tree.query(P[:, :3], k=k+1)
    mean_dist = dist.mean(axis=1)

    #calculate:
    mean = np.mean(mean_dist)
    std = np.std(mean_dist)
    threshold_g = mean + (std * Sfactor)
    
    distance = LA.norm(P[:,:3],axis=1)
    threshold = threshold_g * r * distance
    mask = mean_dist >= threshold
    return mask


def DROR(P, b=0.05, alpha=1, kmin=2, SRmin=5, rp=2):
    """
    Dynamic Radius Outlier Removal
    
    Inputs:
    P : the raw point cloud containing all points p
    b : multiplication factor
    alpha : horizontal angular resolution of the lidar
    kmin : minimum number of neighbors
    SRmin : minimum search radius
    rp : the range from the sensor to the point p
    
    Returns:
    Mask where 1 means outlier
    """
    tree = KDTree(P[:, :3])  # leaf_size=5
    rp = P[:,:2]*P[:,:2]
    rp = rp.sum(axis=1)**.5

    search_radius = np.ndarray((P.shape[0],))
    for index, radius in enumerate(rp):
        rp = SRmin if radius < SRmin else b * (radius * alpha)
        search_radius[index] = rp
        
    number_of_neighbours = tree.query_radius(P, r=search_radius, count_only=True) # this line eats cpu time
    mask = [n < kmin for n in number_of_neighbours]
            
    return np.asarray(mask, dtype=np.bool)


def EE(X, contamination:float = 0.1):
    "Returns mask where 1 mean outlier"
    from sklearn.covariance import EllipticEnvelope
    ee = EllipticEnvelope(contamination=contamination)
    yhat = ee.fit_predict(X)
    mask = yhat == -1
    return mask

def OCS(X, contamination:float = 0.1,  kernel:str='rbf'):
    "Returns mask where 1 mean outlier"
    from sklearn.svm import OneClassSVM
    ocs = OneClassSVM(nu=contamination)
    yhat = ocs.fit_predict(X)
    mask = yhat == -1
    return mask

def LOF(X, contamination:float='auto', n_neighbors:int=5, metric:str='l1'):
    "Returns mask where 1 mean outlier"
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors, metric=metric)
    yhat = lof.fit_predict(X)
    mask = yhat == -1
    return mask

def IF(X, contamination:float = 0.1, n_estimators:int=100, max_features:float=1.0):
    "Returns mask where 1 mean outlier"
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=contamination, n_estimators=n_estimators, max_features=max_features)
    yhat = iso.fit_predict(X)
    mask = yhat == -1
    return mask
