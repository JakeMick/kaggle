import scipy as sp

def logloss(act, pred):
    """ Vectorised computation of logloss """
    #cap in official Kaggle implementation, 
    #per forums/t/1576/r-code-for-logloss
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    #compute logloss function (vectorised)
    ll = sum(   act*sp.log(pred) + 
                sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return abs(ll)
