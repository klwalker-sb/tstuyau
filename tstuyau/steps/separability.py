import math
import numpy as np
import pandas as pd
from ..handler import logger

def sep_mstat(p,q):
    ## parametric separability index (M) (Kaufman and Remer, 1994, good if >1)
    sep = abs(np.mean(p) - np.mean(q)) / (np.std(p)+np.std(q))
    logger.debug(f'separability (M) = {sep}')

    return round(sep,2)

def sep_bhatt(p,q):
    ## Bhattacharyya coefficient
    bc=np.sum(np.sqrt(p*q))
    ## Bhattacharyya distance
    b=-np.log(bc)
    logger.debug(f'Bhattacharyya distance = {b}')

    return round(b,3)

def sep_jm(p,q):
    ## Jeffries-Matusita distance
    b = sep_bhatt(p,q)
    jm= 2*(1 - math.exp(-b)) 
    logger.debug(f'Jeffries-Matusita distance = {jm}')

    return int(jm)

def sep_jenshan(p,q):
    ## Jensen-Shannon divergence
    m=(p+q)/2
    js=0.5*np.sum(p*np.log(p/m))+0.5*np.sum(q*np.log(q/m))
    logger.debug(f'Jensen-Shannon divergence = {js}')

    return int(js)
    
def sep_fdiverg(p,q):
    ## f divergence
    def f(t):
        return t*np.log(t)
    f1=np.sum(q*f(p/q))
    logger.debug(f'f divergence = {f1}')

    return int(f1)


def get_separability(df,col1,col2=None,classes=None,stat='allstats'):
    '''
    returns separability measures for two classes or for pre-post observations of an event (e.g. burning)
    <col1> and <col2> are each columns of <df> with measures to compare. e.g. pre/post. if col2 == None, <classes> is used to filter <col1>  
    '''
    #df.reset_index(drop=True, inplace=True) 
    if col2:
        p = df[col1]
        q = df[col2]
    else:
        p = df[col1 == classes[0]][col1]
        q = df[col1 == classes[1]][col1]

    mstat = sep_mstat(p,q)
    f1 = sep_fdiverg(p,q)
    bhatt = sep_bhatt(p,q)
    jm = sep_jm(p,q)
    jen = sep_jenshan(p,q)
    
    if stat.startswith('all'):
        return mstat,f1,bhatt,jm,jen
    else:
        return stat
        
'''
def separabilityTS(tsbdf):

    tsbdf.reset_index(drop=True, inplace=True) 
    p = tsbdf[-1.0]
    M = []
    for nd in range(1,15):
        q = TSBdf[nd]
        
        ##parametric separability index (M) (Kaufman and Remer, 1994, good if >1)
        sep = abs(np.mean(p) - np.mean(q)) / (np.std(p)+np.std(q))
        M.append(sep)
       
    mdf = pd.DataFrame(M)
    mdf.index += 1 
    print(mdf)
    return mdf
'''

        
        