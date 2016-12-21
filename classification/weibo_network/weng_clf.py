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
from time import time

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
    c2_f1 = 2*c2_precision*c2_recall/(c2_precision+c2_recall)

    weighted_prec = c1_precision*(np.sum(cm[0]))/(np.sum(cm))+c2_precision*(np.sum(cm[1]))/(np.sum(cm))
    weighted_rec = c1_recall*(np.sum(cm[0]))/(np.sum(cm)) + c2_recall*(np.sum(cm[1]))/(np.sum(cm))
    weighted_f1 = c1_f1*(np.sum(cm[0]))/(np.sum(cm)) + c2_f1*(np.sum(cm[1]))/(np.sum(cm))

    return c1_precision,c1_recall,c1_f1,c2_precision,c2_recall,c2_f1,weighted_prec,weighted_rec,weighted_f1


if __name__ == '__main__':
    #p = 'E:\\ViralPred\\weibo_network\\'
    #p = 'E:\\ViralPred\\weng_network\\'
    DS = 'Twitter'
    clf = 'SVM'
    #f_name = 'rpp_NT36001percent10iters.csv'
    #f_name = 'rpp_NT36001percent10iters.csv'

    if DS == 'Twitter':
        TH0 = 106.0
        TH1 = 226.0
        TH2 = 587.0

        p = 'E:\\ViralPred\\weng_network\\'
        db_name = 'Twitter_Indiana'
        q = 'SELECT * FROM fs_df_50'
        f_name = 'weng_features_1.3.csv'
        fs_col_no = 3

    if DS == 'Weibo':
        TH0 = 152.0
        TH1 = 325.0
        TH2 = 688.0

        p = 'E:\\ViralPred\\weibo_network\\'
        db_name = 'Aminer'
        q = 'SELECT * FROM fs_df'
        f_name = 'weng_features_1.0.csv'
        fs_col_no = 2

    TH = TH0
    asonam_df = pd.read_csv(p+f_name,index_col=0,header=None)
    #print asonam_df.head()
    #asonam_df = asonam_df.head(20000)

    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/'+db_name)
    conn = engine.connect()

    #q = 'SELECT * FROM fs_df_50'
    q_res = conn.execute(q)
    fs_df = pd.DataFrame(q_res.fetchall())
    fs_df = fs_df.set_index(1)
    print fs_df.head()
    print asonam_df.head()
    print len(fs_df),len(asonam_df)
    fs_df = fs_df.loc[asonam_df.index.values]
    fs_list = fs_df[fs_col_no].values
    fs_list = [x for x in fs_list if int(x) >= 50]

    #TH = TH0
    #clf = 'RF'

    X = asonam_df.values
    y = []
    for fs in fs_list:
        if fs >= TH:
            y.append(1)
        else:
            y.append(0)

    print 'we have %d samples' %len(y)

    y = np.asarray(y)
    time_list = []
    for i in xrange(0,10):
        t_0 = time()
        kf = StratifiedKFold(y,n_folds=10,random_state=0,shuffle=True)
        j = 0
        cm = np.zeros((2,2))
        for tr_ind,ts_ind in kf:
            if clf == 'SVM':
                lr = SVC(cache_size=200000,kernel='rbf',C=1)
            if clf == 'LSVM':
                lr = LinearSVC()
            if clf == 'LR':
                lr = LogisticRegression(C=0.0001,n_jobs=-1)
            if clf == 'RF':
                lr = RandomForestClassifier(n_estimators=100,criterion='gini',max_features='sqrt',n_jobs=1)

            lr.fit(X[tr_ind],y[tr_ind])
            y_pred = lr.predict(X[ts_ind])
            y_true = y[ts_ind]

            cm_one_fold = confusion_matrix(y_true,y_pred)
            if cm_one_fold.shape == (2,2):
                cm = np.add(cm,cm_one_fold)
            else:
                cm[0][0] += cm_one_fold

            #ROC curve
            '''
            if type(lr) == type(LinearSVC()):
                y_pred_proba = lr._predict_proba_lr(X[ts_ind])
            else:
                y_pred_proba = lr.predict_proba(X[ts_ind])
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
            #fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (j, roc_auc))

            j += 1
            '''

        '''
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        mean_tpr /= float(j)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--',
                 label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.legend(loc="lower right")
        plt.show()
        '''
        print lr
        print cm
        time_list.append(time()-t_0)
        c1_precision,c1_recall,c1_f1,c2_precision,\
        c2_recall,c2_f1,weighted_prec,weighted_rec,weighted_f1 = performance(cm)

        of = open(p+'clf_res\\weng_TH_'+str(int(TH))+'_'+clf+'C1.csv','ab')
        wrt = csv.writer(of)
        wrt.writerow([c1_precision,c1_recall,c1_f1,c2_precision,
                      c2_recall,c2_f1,weighted_prec,weighted_rec,weighted_f1])

        of.close()
    time_of = open(p+'run_time\\weng_clf_'+str(clf)+'_train_testC1.csv','ab')

    time_wrt = csv.writer(time_of)
    time_wrt.writerow([np.mean(time_list),np.median(time_list),np.std(time_list)])

    time_of.close()





