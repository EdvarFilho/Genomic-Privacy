import numpy as np
from scipy.stats import chi2

def gradient_descent(init, steps, grad, desired_sum, proj=lambda x: x, min_value=1):
    xs = [init]
    for step in steps:
        xs_next = proj(xs[-1] - step * grad(xs[-1]), desired_sum, min_value)
        xs = [xs_next]
    return xs

def l2_gradient(b, x):
    return 2*(x - b)

def proj(x, desired_sum, min_value=1):
    """Projection of x onto the subspace"""
    current_sum = np.sum(x)
    factor = (current_sum - desired_sum)/len(x)
    projection = np.around(np.clip(x - factor, min_value, None))     
    
    return projection

def min_l2_norm(arr, desired_sum, num_steps=10, min_value=1):
    x0 = arr.copy()
    alpha = 0.1
    gradient = lambda y: l2_gradient(arr, y)
    xs = gradient_descent(x0, [alpha]*num_steps, gradient, desired_sum, proj, min_value)
    return xs[-1].astype(int)

def gerarTabelas(dataset):
    tabsCont = {}
    for i in range(0, dataset.shape[1] - 1):
        casos0, casos1, casos2, controles0, controles1, controles2 = 0, 0, 0, 0, 0, 0
        for j in range(0, dataset.shape[0]):
            if(dataset[j,i] == 0 and dataset[j,dataset.shape[1]-1]=='Caso'):
                casos0 += 1
            elif(dataset[j,i] == 1 and dataset[j,dataset.shape[1]-1]=='Caso'):
                casos1 += 1
            elif(dataset[j,i] == 2 and dataset[j,dataset.shape[1]-1]=='Caso'):
                casos2 += 1
            elif(dataset[j,i] == 0 and dataset[j,dataset.shape[1]-1]=='Controle'):
                controles0 += 1
            elif(dataset[j,i] == 1 and dataset[j,dataset.shape[1]-1]=='Controle'):
                controles1 += 1
            elif(dataset[j,i] == 2 and dataset[j,dataset.shape[1]-1]=='Controle'):
                controles2 += 1
        tabsCont[i] = [[casos0, casos1, casos2],[controles0, controles1, controles2]]
    return tabsCont

def mre(arr_pred, arr_true):
    error = np.mean(np.abs((arr_pred - arr_true)/arr_true))
    return error

def chi(tabs):
    chisq = []
    for i in range(0, len(tabs)):
        tab = np.array(tabs[i])
        m = np.sum(tab, axis = 1)
        s = np.sum(tab, axis = 0)
        n = np.sum(m)
        E = np.zeros(tab.shape)
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                if(n == 0):
                    E[i,j] = 0
                else:
                    E[i,j] = s[j]*m[i]/n
        qui = 0
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                if(E[i,j] == 0):
                    qui = qui + 0
                else:
                    qui = qui + (E[i,j]-tab[i,j])**2/E[i,j]
        chisq.append(qui)
    return chisq

def pValue(chiOriginal, df):
    ps = []
    for i in range(0, len(chiOriginal)):
        p = chi2.sf(chiOriginal[i], df)
        ps.append(p)
    return ps