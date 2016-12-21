#need to exclude inf
import pandas as pd
from sqlalchemy import create_engine
from regression.ten_fold import APE,RMSLE,TOPK_ACC_TAU
import numpy as np
from scipy import stats
import decimal
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
        y_p = decimal.Decimal(max(y_pred[i],50.0))
        y_t = decimal.Decimal(y_true[i])
        e += (y_p-y_t)*(y_p-y_t)
    if e < 0:
        print e
    rmse = np.sqrt(e/decimal.Decimal(len(y_true)))

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
    #clf = 'RF'
    #f_name = 'rpp_NT36001percent10iters.csv'
    f_name = 'rpp_NT360010percent10iters.csv'
    if DS == 'Twitter':

        p = 'E:\\ViralPred\\weng_network\\'
        db_name = 'Twitter_Indiana'
        q = 'SELECT * FROM fs_df_50'

        fs_col_no = 3

    if DS == 'Weibo':

        p = 'E:\\ViralPred\\weibo_network\\'
        db_name = 'Aminer'
        q = 'SELECT * FROM fs_df'
        fs_col_no = 2
    #p = 'E:\\ViralPred\\weibo_network\\'
    #p = 'E:\\ViralPred\\weng_network\\'

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

    reg_of = open(p+'reg_res\\'+f_name+'.csv','ab')
    reg_wrt = csv.writer(reg_of)

    loop_cnt = 0
    for name_list in name_lists:
        fs_df_loop = fs_df.loc[name_list]
        fs_list = fs_df_loop[fs_col_no].values.astype('int').tolist()
        y_max = max(fs_list)
        result_array = np.asarray(result_lists[loop_cnt])
        Y_pred = []
        Y_true = []
        for col_no in xrange(0,1):
            y_pred_init = result_array[:,col_no]
            y_true = fs_list
            y_pred = []
            for y in y_pred_init:
                if np.isnan(y) or y < 50:
                    y = 50
                elif np.isinf(y) or y > y_max:
                    y = y_max
                y_pred.append(y)

            ape_one_iter = APE(y_pred,y_true)
            rmse_one_iter = RMSE(y_pred,y_true)
            rmsle_one_iter = RMSLE(y_pred,y_true)
            topkacc_one_iter,tau_one_iter = TOPK_ACC_TAU(y_pred,y_true,K=int(len(y_true)/10.0))

            print ape_one_iter,rmse_one_iter,rmsle_one_iter,topkacc_one_iter,tau_one_iter
            reg_wrt.writerow([ape_one_iter,rmse_one_iter,rmsle_one_iter,topkacc_one_iter,tau_one_iter])
            #Y_pred.extend(y_pred)
            #Y_true.extend(y_true)

        #ape = APE(Y_pred,Y_true)
        #rmse = RMSE(Y_pred,Y_true)
        #rmlse = RMSLE(Y_pred,Y_true)
        #topkacc,tau = TOPK_ACC_TAU(Y_pred,Y_true,K=int(len(Y_true)/10.0))

        #print ape,rmse,rmlse,topkacc,tau


        loop_cnt += 1
