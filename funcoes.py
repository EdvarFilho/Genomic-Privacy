import numpy as np
from scipy.stats import chi2
from scipy.special import comb

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
    e = []
    for i in range(len(arr_true)):
        if(arr_true[i]==0):
            e.append(0)
        else:
            e.append(np.abs((arr_pred[i] - arr_true[i])/arr_true[i]))
    error = np.mean(np.array(e))
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

def fisher(tabs):
    fis = []
    for i in range(0, len(tabs)):
        tab = np.array(tabs[i])
        s = np.sum(tab, axis = 0)
        n = np.sum(s)
        num = comb(s[0], tab[0,0], exact=True)*comb(s[1], tab[0,1], exact=True)*comb(s[2], tab[0,2], exact=True)
        den = comb(n, (tab[0,0]+tab[0,1]+tab[0,2]), exact=True)
        p = num/den
        fis.append(p)
    return fis

def cochran(tabs):
    coc = []
    for i in range(0, len(tabs)):
        tab = np.array(tabs[i])
        s = np.sum(tab, axis = 0)
        n = np.sum(s)
        num = n * ((2 * s[0] + s[1] - 2 * (2 * tab[0,0] + tab[0,1]))**2)
        den = 4 * n * s[0] + n * s[1] - ((2 * s[0] + s[1])**2)
        if(den == 0):
            coc.append(0)
        else:
            T = num/den
            coc.append(T)
    return coc

def pValue(chiOriginal, df):
    ps = []
    for i in range(0, len(chiOriginal)):
        p = chi2.sf(chiOriginal[i], df)
        ps.append(p)
    return ps

def KL(data, pri_data):
    data /= np.sum(data)
    pri_data /= np.sum(pri_data)
    kl = 0
    for i in range(len(pri_data)):
        if(data[i]==0):
            ld = 0
        else:
            ld = np.log(data[i])
        if(pri_data[i]==0):
            lp = 0
        else:
            lp = np.log(pri_data[i])
        kl += data[i] * ld - data[i] * lp
    return kl