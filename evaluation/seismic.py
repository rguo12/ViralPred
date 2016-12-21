#need to exclude inf
import pandas as pd
from sqlalchemy import create_engine
from regression.ten_fold import APE,RMSLE,TOPK_ACC_TAU
import numpy as np
from scipy import stats
import csv
def APE(y_pred,y_true):
    pe = 0
    for i in xrange(0,len(y_pred)):
        pe += abs(max(y_pred[i],50.0) - y_true[i])/float(y_true[i])
    ape = pe/float(len(y_true))
    return ape

def RMSE(y_pred,y_true):
    e = 0
    for i in xrange(0,len(y_pred)):
        e += np.square(max(y_pred[i],50.0) - y_true[i])
    if e < 0:
        print e
    rmse = np.sqrt(e/float(len(y_true)))

    return rmse

def RMSLE(y_pred,y_true):
    e = 0
    for i in xrange(0,len(y_pred)):

        if y_pred[i] <0:
            print y_pred[i],i
        e += np.square(np.log1p(max(y_pred[i],50.0)) - np.log1p(y_true[i]+0.0001))

    rmlse = np.sqrt(e/float(len(y_true)))

    return rmlse

def TOPK_ACC_TAU(y_pred,y_true,K=100):
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

    #return rmse



if __name__ == '__main__':

    DS = 'Weibo'

    if DS == 'Weibo':
        p = 'E:\\ViralPred\\weibo_network\\'
        db_name = 'Aminer'
        q = 'SELECT * FROM fs_df'
        f_name = 'seismic_pred_weibo_300.txt'
        fs_col_no = 2
    if DS == 'Twitter':
        p = 'E:\\ViralPred\\twitter_network\\'
        db_name = 'Twitter_Indiana'
        q = 'SELECT * FROM fs_df_50'
        f_name = 'seismic_pred_weng_30k.txt'
        fs_col_no = 3


    #res_df = pd.read_csv(p+f_name,header=None)


    '''
    print res_df.head()
    print type(res_df[0].values[0])
    res_list = []
    for x in res_df[2].values:
        #res_list.append(int(x.replace('[','').replace(']','').split('.')[0]))
        res_list.append(x)
    #.astype(np.float).tolist()
    '''

    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/'+db_name)
    conn = engine.connect()

    q_res = conn.execute(q)
    fs_df = pd.DataFrame(q_res.fetchall())
    print fs_df.head()
    fs_list = fs_df[fs_col_no].values.astype('int').tolist()
    fs_list_50 = [x for x in fs_list if x >= 50]

    y_max = np.max(fs_list_50)

    res_f = open(p+f_name,'rb')

    y_true = []

    reg_of = open(p+'reg_res\\seismic.csv','ab')
    reg_wrt = csv.writer(reg_of)

    last_y_pred = np.zeros((99257,5))
    for m in xrange(0,10):
        ln_cnt = 0
        y_pred = []
        for line in res_f:
            line_raw = line[:-1].split(',')
            row = []
            for ele in line_raw:
                ele = float(ele)
                if np.isnan(ele) or ele< 50:
                    ele = 50
                elif np.isinf(ele) or ele > y_max:
                    ele = y_max
                row.append(ele)
            y_pred.append(row)
            ln_cnt += 1
            if ln_cnt == 99257:
                break

        y_pred = np.asarray(y_pred)
        for k in xrange(0,1):
            col_pred = y_pred[:,k]
            y_true = fs_list_50

            ape = APE(col_pred,y_true)
            rmse = RMSE(col_pred,y_true)
            rmlse = RMSLE(col_pred,y_true)
            topkacc,tau = TOPK_ACC_TAU(col_pred,y_true,K=len(col_pred)/10)

            print [ape,rmse,rmlse,topkacc,tau]


            reg_wrt.writerow([ape,rmse,rmlse,topkacc,tau])

        print (last_y_pred == y_pred).all()
        last_y_pred = y_pred







    #topkacc_vector = np.zeros(10)
    #tau_vector = np.zeros(10)
    #j = 0
    '''


    for i in xrange(0,len(fs_list_50)):
        #if res_list[i] == 'inf' or res_list[i] < 0:
            #continue
        y_true.append(fs_list_50[i])
        y_pred_raw = res_list[i]
        #y_pred.append(res_list[i])


        #if np.isnan(y_pred_raw) or y_pred_raw < 50:


    print len(y_true)
    '''
