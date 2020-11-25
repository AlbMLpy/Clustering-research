import numpy as np
import pandas as pd

from sklearn.neighbors import kneighbors_graph as kg
from sklearn.datasets import make_circles, make_moons, load_sample_image 
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg


class Error(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, info):
        self.info = info

class LocalSpectral:
    """
        This class is used to solve Spectral clustering task 
        with constraints on localization.
        
        Parameters.
        ----------
        A : numpy.ndarray
            Adjacency matrix of graph nodes;
    """
    def __init__(self, A):
        self.A = A
        self.diag = A.sum(axis=1)
        self.vol_graph = self.diag.sum()
        self.laplacian = csr_matrix(np.diag(self.diag) - A)
        self.s = None    
   

    def _solve(self, s, k, tol, err, n_iter, shift):
        b = self.diag * s
        gamma = -self.vol_graph
        A = self.laplacian - gamma*diags(self.diag)
        y, _ = cg(A, b)
        check = (s.dot(self.diag * y))**2
        num_iter = 0
        while np.abs(check - k) > err:
            num_iter += 1
            gamma += shift * self.vol_graph
            A = self.laplacian - gamma*diags(self.diag)
            y, _ = cg(A, b, tol=tol)
            check = (s.dot(self.diag * y))**2
            if num_iter == n_iter:
                print("Ended iterations.")
                break
        self.x = y 
        self.gamma = gamma
    
    
    def _volume(self, s):
        return self.diag[s].sum()
    
    
    def _conductance(self, s):
        vol = self._volume(s)
        denom = vol * (self.vol_graph - vol)
        summ = 0
        for i in s:
            for j in range(self.A[i, :].size):
                if self.A[i, j] > 0.0:
                    if j not in s:
                        summ += 1
                    
        return (self.vol_graph * summ) / denom 
    
    
    def _sweep_cut(self, k):
        p_d = pd.Series(self.x).sort_values(ascending=False)
        supp_length = np.sum(self.x > 0.0)
        a = self.x.dot(self.laplacian.dot(self.x))
        b = self.x.dot(self.diag * self.x)
        phi = a / b
        s = []
        for j in range(1, supp_length + 1):
            s = list(p_d.iloc[list(range(0, j))].index)
            cond = self._conductance(s)
            cond_help = 1000
            if cond**2 <= 8*phi and cond < cond_help and self._volume(s) < k:
                cond_help = cond
                self.s = s
                
        if self.s == None:
            raise Error("\nFound empty cluster, change parameters\n")
    
    
    def process(self, u, k=15, tol=0.00001, err=0.001, n_iter = 10000, shift=0.001):
        """
            This method produce list of nodes that solve Spectral Graph Partitioning task.
        
            Parameters.
            ----------
            u : list
                List of initial local nodes;
            
            k : int
                Volume that we want our local set to have;
            
            tol : double
                Tolerance that is used to solve system of linear equations by
                Conjugate Gradients method;
                
            err : double
                We want to find 'x'- local partitioning vector s.t: abs((x^T*D*s)^2 - k) < err
            
            n_iter : int
                Number of iterations we allow to find 'x' - local partitioning vector.
            
            shift : double
                Shift that we use to approximate 'x'.
        
            Returns.
            ----------
            out : list
                List of nodes according initial data that fit objective;
                
        """
        s = np.zeros(self.A.shape[0])
        s[u] = 1.0
        self._solve(s, k=1/k, tol=tol, err=err, n_iter=n_iter, shift=shift)
        self._sweep_cut(k)
        return self.s


# Produce random points around centers:
def rand_around_centers(centers, number=50):
    """
        Produce random points around centers.
        
        Parameters.
        ----------
        centers : numpy.ndarray
            This array includes x, y coordinates of centers;
        
        number : int
            What number of points do you want scattered around centers;
        
        Returns.
        ----------
        out : numpy.ndarray, numpy.ndarray
            The first array is filled with x, y coordinates of a point;
            The second array is filled with labels tailored to a point;    
    """
    result = []
    labels = []
    for c, j in zip(centers, range(centers.shape[0])):
        result.append(c)
        labels.append(j)
        for i in range(number):
            result.append(
                [c[0] + 0.9*np.random.rand(), c[1] + 0.8*np.random.rand()]
            )
            labels.append(j)
    return np.array(result), np.array(labels)       


# Draw data like scatter plot, plus local coloring
def draw2d(data, c=None, s=None):
    """
        Plot scatter data.
        
        Parameters.
        ----------
        data : numpy.ndarray
            This array includes x, y coordinates of data points;
        
        c : array_like
            Labels array to emphasize color of points;
        
        s : array_like
            Local clustering points, their position in data;
            If s=None then function does not plot extra color points;
            Else highlights them with red color.
    """
    plt.scatter(data[:, 0], data[:, 1], c=c)
    if s != None:
        plt.scatter(data[s][:, 0], data[s][:, 1], c='red')
        

def plot_results(data, labels, local_labels, name_global, names_local, file):
    """
        It is just convenient function to plot the results.
        
        Parameters.
        ----------
        data : numpy.ndarray
            Array that contains x, y coordinates of points;
            
        labels : numpy.ndarray
            Array that contains labels for all the points in "data";
        
        local_labels : numpy.ndarray
            Array that contains such row values of points
            that lie in a local cluster;
            
        name_global : string
            Name used for a plot in the whole;
        
        names_local : list of strings
            Names used locally in plots;
        
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'{name_global}', fontsize=16)

    #1
    axs[0, 0].set_title(f'{names_local[0]}')
    axs[0, 0].scatter(data[:, 0], data[:, 1], c=labels)
    axs[0, 0].scatter(data[local_labels[0]][:, 0], data[local_labels[0]][:, 1], c='red')


    #2
    axs[0, 1].set_title(f'{names_local[1]}')
    axs[0, 1].scatter(data[:, 0], data[:, 1], c=labels)
    axs[0, 1].scatter(data[local_labels[1]][:, 0], data[local_labels[1]][:, 1], c='red')


    #3
    axs[1, 0].set_title(f'{names_local[2]}')
    axs[1, 0].scatter(data[:, 0], data[:, 1], c=labels)
    axs[1, 0].scatter(data[local_labels[2]][:, 0], data[local_labels[2]][:, 1], c='red')


    #4
    axs[1, 1].set_title(f'{names_local[3]}')
    axs[1, 1].scatter(data[:, 0], data[:, 1], c=labels)
    axs[1, 1].scatter(data[local_labels[3]][:, 0], data[local_labels[3]][:, 1], c='red')        
    fig.savefig(file)
    
def metric(local, target, label):
    """
        This metric counts ratio between true label number of points in local set
        and all the points in the local cluster.
       
        Parameters.
        ----------
        local : list
            List of points in a local cluster, its row numbers;
        
        target : numpy.ndarray
            Array that contains labels of all the points in initial data;
        
        label : int
            Label we used for initial points;
               
        Returns.
        ----------
        out : double
            
            
    """
    result = target[local]
    return (result==label).sum() / result.size 


def process(data, nn, u, k, tol, err, n_iter, shift):
    """
        Extra wrapper to use this program more properly.
        
        Parameters.
        ----------
        data : numpy.ndarray
            Array representing data points, x and coordinates;
        
        nn : int
            The number of nearest neighbours to make Adjacency matrix; 
            
        u : list
            List of initial local nodes;
            
        k : int
            Volume that we want our local set to have;
            
        tol : double
            Tolerance that is used to solve system of linear equations by
            Conjugate Gradients method;
                
        err : double
            We want to find 'x'- local partitioning vector s.t: abs((x^T*D*s)^2 - k) < err
            
        n_iter : int
            Number of iterations we allow to find 'x' - local partitioning vector.
            
        shift : double
            Shift that we use to approximate 'x'.
        
        Returns.
        ----------
        out : list
            List of nodes according initial data that fit objective;
        
    """
    A = kg(data, nn).toarray()
    if (A - A.T).sum() != 0:
        raise Error("\nNon-symmetric matrix\n")
    
    model = LocalSpectral(A)
    s = model.process(u=u, k=k, tol=tol, err=err, n_iter=n_iter, shift=shift)
    return s