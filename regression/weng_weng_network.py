from sklearn.linear_model import LinearRegression,BayesianRidge,Ridge,ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import csv

from sqlalchemy import create_engine

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
        if y_pred[i] <0:
            #print y_pred[i],i
            y_pred[i] = 0
        e += np.square(y_pred[i] - y_true[i])
    if e < 0:
        print e
    rmse = np.sqrt(e/float(len(y_true)))

    return rmse

def RMSLE(y_pred,y_true):
    e = 0
    for i in xrange(0,len(y_pred)):

        if y_pred[i] <0:
            #print y_pred[i],i
            y_pred[i] = 0
        e += np.square(np.log1p(y_pred[i]) - np.log1p(y_true[i]))

    rmse = np.sqrt(e/float(len(y_true)))

    return rmse

def TOPK_ACC_TAU(y_pred,y_true,K=146):
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
    asonam_df = pd.read_csv('E:\ViralPred\\weng_network\\weng_features_1.3.csv',index_col=0,header=None)
    print asonam_df.head()
    #asonam_df = asonam_df.head(20000)

    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/Twitter_Indiana')
    conn = engine.connect()

    q = 'SELECT * FROM fs_df_50'
    q_res = conn.execute(q)
    fs_df = pd.DataFrame(q_res.fetchall())
    fs_df = fs_df.set_index(1)
    print fs_df.head()
    print len(fs_df),len(asonam_df)
    fs_df = fs_df.loc[asonam_df.index.values]
    fs_list = fs_df[3].values.tolist()
    fs_list = [x for x in fs_list if x >= 50]

    #kshell = superspreader_df[2].values
    X = asonam_df.values
    #X = np.asarray(X)
    y = np.asarray(fs_list)

    print len(y)

    kf = KFold(len(y),n_folds=10,random_state=10)

    y_true_all = []
    y_pred_all = []

    #topkacc_list = []
    #tau_list = []
    topkacc_vector = np.zeros(10)
    tau_vector = np.zeros(10)
    j = 0

    reg_of = open('E:\\ViralPred\\weng_network\\reg_res\\weng.csv','ab')
    reg_wrt = csv.writer(reg_of)
    for tr_ind,ts_ind in kf:
        #lr = LinearRegression(n_jobs=-1)
        #lr = Ridge()
        #lr = ElasticNet()
        lr = SVR(cache_size=20000,kernel='rbf',C=0.0001)
        #lr = BayesianRidge()
        #lr = ExtraTreesRegressor(n_estimators=100,n_jobs=-1,bootstrap=True,max_features='sqrt')
        lr.fit(X[tr_ind],y[tr_ind])
        y_pred = lr.predict(X[ts_ind])

        y_true = y[ts_ind]

        y_pred_all.extend(y_pred)
        y_true_all.extend(y_true)
        topkacc,tau = TOPK_ACC_TAU(y_pred,y_true,K=29)
        topkacc_vector[j] = topkacc
        tau_vector[j] = tau
        j += 1

    ape = APE(y_pred_all,y_true_all)
    rmse = RMSE(y_pred_all,y_true_all)
    rmlse = RMSLE(y_pred_all,y_true_all)
    topkacc = np.mean(topkacc_vector)
    tau = np.mean(tau_vector)

    row = [ape,rmse,rmlse,topkacc,tau]
    reg_wrt.writerow(row)
