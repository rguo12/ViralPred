#need to exclude inf
import pandas as pd
from sqlalchemy import create_engine

import numpy as np
from scipy import stats
import decimal
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.metrics import confusion_matrix

def performance(cm):
    c1_precision = float(cm[0][0])/(cm[0][0]+cm[1][0])
    c1_recall = float(cm[0][0])/(cm[0][0]+cm[0][1])
    c2_precision = float(cm[1][1])/(cm[1][1]+cm[0][1])
    c2_recall = float(cm[1][1])/(cm[1][1]+cm[1][0])
    c1_f1 = 2*c1_precision*c1_recall/(c1_precision+c1_recall)
    c2_f1 = 2*c2_precision*c2_recall/(c2_precision+c2_recall)

    weighted_prec = c1_precision*(np.sum(cm[0]))/(np.sum(cm))+c2_precision*(np.sum(cm[1]))/(np.sum(cm))
    weighted_rec = c1_recall*(np.sum(cm[0]))/(np.sum(cm)) + c2_recall*(np.sum(cm[1]))/(np.sum(cm))
    weighted_f1 = c1_f1*(np.sum(cm[0]))/(np.sum(cm)) + c2_f1*(np.sum(cm[1]))/(np.sum(cm))

    return c1_precision,c1_recall,c1_f1,c2_precision,c2_recall,c2_f1,weighted_prec,weighted_rec,weighted_f1



if __name__ == '__main__':
    #p = 'E:\\ViralPred\\weibo_network\\'
    #p = 'E:\\ViralPred\\weng_network\\'
    DS = 'Weibo'
    clf = 'RF'
    #f_name = 'rpp_NT36001percent10iters.csv'
    f_name = 'rpp_NT360010percent10iters.csv'
    if DS == 'Twitter':
        TH0 = 106.0
        TH1 = 226.0
        TH2 = 587.0

        p = 'E:\\ViralPred\\weng_network\\'
        db_name = 'Twitter_Indiana'
        q = 'SELECT * FROM fs_df_50'
        fs_col_no = 3

    if DS == 'Weibo':
        TH0 = 152.0
        TH1 = 325.0
        TH2 = 688.0

        p = 'E:\\ViralPred\\weibo_network\\'
        db_name = 'Aminer'
        q = 'SELECT * FROM fs_df'
        fs_col_no = 2

    TH = TH0

    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/'+db_name)
    conn = engine.connect()


    q_res = conn.execute(q)
    fs_df = pd.DataFrame(q_res.fetchall())
    fs_df = fs_df.set_index(1)
    print fs_df.head()

    #fs_list_50 = [x for x in fs_list if x >= 50]

    res_df = pd.read_csv(p+f_name,header=None)
    print res_df.head()
    print type(res_df[0].values[0])

    names = res_df[0].values.astype('int')
    res_matrix = res_df.values

    #divide names into ten iterations
    name_lists = [[] for x in xrange(0,10)]
    result_lists = [[] for x in xrange(0,10)]
    iter_cnt = 0
    last_ind = 0
    for i in xrange(0,len(names)):
        name = names[i]
        current_ind = int(name)
        if current_ind >= last_ind:
            name_lists[iter_cnt].append(name)
            result_lists[iter_cnt].append(res_matrix[i][1:])
        else:
            iter_cnt += 1
        last_ind = current_ind

    name_lists = np.asarray(name_lists)

    #print result_lists.shape

    clf_of = open(p+'clf_res\\'+f_name+'_TH_'+str(int(TH))+'_'+clf+'.csv','ab')
    clf_wrt = csv.writer(clf_of)

    loop_cnt = 0
    for name_list in name_lists: # each iter is a iter on the clf process
        if len(name_list) == 0:
            break
        fs_df_loop = fs_df.loc[name_list]
        fs_list = fs_df_loop[fs_col_no].values.astype('int').tolist()
        y_max = max(fs_list)
        result_array = np.asarray(result_lists[loop_cnt])

        X = []

        for result_row in result_array:
            x_row = []
            for x in result_row:
                if x < 50 or np.isnan(x):
                    x = 50
                if np.isinf(x) or x > y_max:
                    x = y_max
                x_row.append(x)
            X.append(x_row)

        y = []

        for y_true in fs_list:
            if y_true >= TH:
                y.append(1)
            else:
                y.append(0)
        X = np.asarray(X)
        y = np.asarray(y)

        kf = StratifiedKFold(y,n_folds=10,random_state=0,shuffle=True)
        cm = np.zeros((2,2))
        for tr_ind,ts_ind in kf:
            if clf == 'SVM':
                lr = SVC(cache_size=20000,kernel='rbf',C=1,probability=True)
            if clf == 'LR':
                lr = LogisticRegression(C=0.0001,n_jobs=-1)
            if clf == 'RF':
                lr = RandomForestClassifier(n_estimators=100,criterion='gini',max_features='sqrt',n_jobs=-1)

            lr.fit(X[tr_ind],y[tr_ind])
            y_pred = lr.predict(X[ts_ind])
            y_true = y[ts_ind]

            cm_one_fold = confusion_matrix(y_true,y_pred)
            if cm_one_fold.shape == (2,2):
                cm = np.add(cm,cm_one_fold)
            else:
                cm[0][0] += cm_one_fold

            #Y_pred.extend(y_pred)
            #Y_true.extend(y_true)

        #ape = APE(Y_pred,Y_true)
        #rmse = RMSE(Y_pred,Y_true)
        #rmlse = RMSLE(Y_pred,Y_true)
        #topkacc,tau = TOPK_ACC_TAU(Y_pred,Y_true,K=int(len(Y_true)/10.0))

        #print ape,rmse,rmlse,topkacc,tau

        loop_cnt += 1

        print cm

        c1_precision,c1_recall,c1_f1,c2_precision,\
        c2_recall,c2_f1,weighted_prec,weighted_rec,weighted_f1 = performance(cm)


        clf_wrt.writerow([c1_precision,c1_recall,c1_f1,c2_precision,
                      c2_recall,c2_f1,weighted_prec,weighted_rec,weighted_f1])
