from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold,KFold
from sklearn.metrics import confusion_matrix,roc_curve,auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interp
import csv

from sqlalchemy import create_engine

TH0 = 106.0
TH1 = 226.0
TH2 = 587.0

def performance(cm):
    c1_precision = float(cm[0][0])/(cm[0][0]+cm[1][0])
    c1_recall = float(cm[0][0])/(cm[0][0]+cm[0][1])
    c2_precision = float(cm[1][1])/(cm[1][1]+cm[0][1])
    c2_recall = float(cm[1][1])/(cm[1][1]+cm[1][0])
    c1_f1 = 2*c1_precision*c1_recall/(c1_precision+c1_recall)
    if c2_recall == 0 or c2_precision == 0:
        c2_f1 = 0.0
    else:
        c2_f1 = 2*c2_precision*c2_recall/(c2_precision+c2_recall)

    weighted_prec = c1_precision*(np.sum(cm[0]))/(np.sum(cm))+c2_precision*(np.sum(cm[1]))/(np.sum(cm))
    weighted_rec = c1_recall*(np.sum(cm[0]))/(np.sum(cm)) + c2_recall*(np.sum(cm[1]))/(np.sum(cm))
    weighted_f1 = c1_f1*(np.sum(cm[0]))/(np.sum(cm)) + c2_f1*(np.sum(cm[1]))/(np.sum(cm))

    return c1_precision,c1_recall,c1_f1,c2_precision,c2_recall,c2_f1,weighted_prec,weighted_rec,weighted_f1


if __name__ == '__main__':
    #asonam_df = pd.read_csv('E:\ViralPred\\weng_network\\asonam_features_lambda1800.csv',index_col=0,header=None)
    #print asonam_df.head()
    #asonam_df = asonam_df.head(20000)

    #kshell_df = pd.read_csv('E:\ViralPred\\weng_network\\superspreader_design_matrix_50.csv',header=None,index_col=0)
    #kshell_df = pd.read_csv('E:\ViralPred\\weng_network\\evcent_design_matrix_50.csv',header=None,index_col=0)
    res_df = pd.read_csv('E:\\ViralPred\\weng_network\\seismic_pred_weng_30k.txt',header=None)
    print res_df.head()
    print type(res_df[0].values[0])
    res_list = []
    for x in res_df[2].values:
        #res_list.append(int(x.replace('[','').replace(']','').split('.')[0]))
        res_list.append(x)
    #.astype(np.float).tolist()

    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/Twitter_Indiana')
    conn = engine.connect()

    q = 'SELECT * FROM fs_df_50'
    q_res = conn.execute(q)
    fs_df = pd.DataFrame(q_res.fetchall())
    print fs_df.head()
    fs_list = fs_df[3].values.astype('int').tolist()
    fs_list_50 = [x for x in fs_list if x >= 50]

    TH = TH2
    clf = 'RF'

    y_true = []
    y_pred = []
    #topkacc_vector = np.zeros(10)
    #tau_vector = np.zeros(10)
    #j = 0
    for i in xrange(0,len(fs_list_50)):
        #if res_list[i] == 'inf' or res_list[i] < 0:
            #continue
        if fs_list_50[i] >= TH:
            y_true.append(1)
        else:
            y_true.append(0)

        if res_list[i] >= TH:
            y_pred.append(1)
        else:
            y_pred.append(0)

    print len(y_true)

    for i in xrange(0,1):
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        cm = np.zeros((2,2))
        #for tr_ind,ts_ind in kf:
        #    if clf == 'SVM':
        #        lr = SVC(cache_size=20000,kernel='rbf',C=0.0001,probability=True)
        #    if clf == 'LR':
        #        lr = LogisticRegression(C=0.0001,n_jobs=-1)
        #    if clf == 'RF':
        #        lr = RandomForestClassifier(n_estimators=100,criterion='gini',max_features='sqrt',n_jobs=-1)

        #    lr.fit(X[tr_ind],y[tr_ind])
        #    y_pred = lr.predict(X[ts_ind])
        #    y_true = y[ts_ind]

        cm = confusion_matrix(y_true,y_pred)


        print cm

        c1_precision,c1_recall,c1_f1,c2_precision,\
        c2_recall,c2_f1,weighted_prec,weighted_rec,weighted_f1 = performance(cm)

        of = open('E:\\ViralPred\\weng_network\\clf_res\\seismic_TH_'+str(int(TH))+'_'+clf+'.csv','ab')
        wrt = csv.writer(of)
        wrt.writerow([c1_precision,c1_recall,c1_f1,c2_precision,
                      c2_recall,c2_f1,weighted_prec,weighted_rec,weighted_f1])

        of.close()





