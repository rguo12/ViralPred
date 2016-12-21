from sklearn.linear_model import LinearRegression,BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import csv
def MAE(y_pred,y_true):
    return

def APE(y_pred,y_true):
    pe = 0
    for i in xrange(0,len(y_pred)):
        pe += abs(y_pred[i] - y_true[i])/float(y_true[i])
    ape = pe/float(len(y_true))
    return ape

def RMSE(y_pred,y_true):
    e = 0
    for i in xrange(0,len(y_pred)):
        e += np.square(y_pred[i] - y_true[i])
    if e < 0:
        print e
    rmse = np.sqrt(e/float(len(y_true)))

    return rmse

def RMSLE(y_pred,y_true):
    e = 0
    for i in xrange(0,len(y_pred)):

        if y_pred[i] <0:
            print y_pred[i],i
        e += np.square(np.log1p(y_pred[i]) - np.log1p(y_true[i]+0.0001))

    rmse = np.sqrt(e/float(len(y_true)))

    return rmse

def TOPK_ACC_TAU(y_pred,y_true,K=147):
    #pick top K out of y_pred
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    #indexes for sorted cascade sizes
    sorted_k_pred = y_pred.argsort()[:][::-1]
    sorted_k_true = y_true.argsort()[:][::-1]

    tau,p = stats.kendalltau(sorted_k_pred,sorted_k_true)

    top_k_pred = sorted_k_pred[-K:]
    top_k_true = sorted_k_true[-K:]

    coverage = len([x for x in top_k_pred if x in top_k_true])/float(K)

    return coverage,tau

if __name__ == '__main__':
    superspreader_df = pd.read_csv('E:\ViralPred\\weng_network\\superspreader_design_matrix_50.csv',header=None)
    reg_of = open('E:\\ViralPred\\weng_network\\reg_res\\kshell.csv','ab')

    reg_wrt = csv.writer(reg_of)
    print superspreader_df.head()

    kshell = superspreader_df[2].values
    X = []
    for x in kshell:
        if np.isnan(x):
            X.append([1])
        else:
            X.append([x])
    #X = [[x] for x in kshell]
    X = np.asarray(X)
    y = superspreader_df[1].values

    print len(y)

    kf = KFold(len(y),n_folds=10,random_state=42)

    y_true_all = []
    y_pred_all = []

    topkacc_vector = np.zeros(10)
    tau_vector = np.zeros(10)
    j = 0

    for tr_ind,ts_ind in kf:
        #lr = LinearRegression()
        lr = LinearRegression()
        #lr = SVR(cache_size=20000,kernel='rbf',C=0.0001)
        #lr = BayesianRidge()
        #lr = ExtraTreesRegressor(n_estimators=10,n_jobs=-1,bootstrap=True)
        lr.fit(X[tr_ind],y[tr_ind])
        y_pred = lr.predict(X[ts_ind])

        y_true = y[ts_ind]

        y_pred_all.extend(y_pred)
        y_true_all.extend(y_true)

        topkacc,tau = TOPK_ACC_TAU(y_pred,y_true)
        topkacc_vector[j] = topkacc
        tau_vector[j] = tau
        j += 1

    ape = APE(y_pred_all,y_true_all)
    rmse = RMSE(y_pred_all,y_true_all)
    rmlse = RMSLE(y_pred_all,y_true_all)
    #topkacc,tau = TOPK_ACC_TAU(y_pred_all,y_true_all)
    topkacc = np.mean(topkacc_vector)
    tau = np.mean(tau_vector)

    reg_wrt.writerow([ape,rmse,rmlse,topkacc,tau])
