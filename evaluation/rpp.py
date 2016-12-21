#need to exclude inf
import pandas as pd
from sqlalchemy import create_engine
from regression.ten_fold import APE,RMSLE,TOPK_ACC_TAU
import numpy as np
from scipy import stats
import decimal
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

    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/Twitter_Indiana')
    conn = engine.connect()

    q = 'SELECT * FROM fs_df_50'
    q_res = conn.execute(q)
    fs_df = pd.DataFrame(q_res.fetchall())
    fs_df = fs_df.set_index(1)
    print fs_df.head()

    #fs_list_50 = [x for x in fs_list if x >= 50]

    res_df = pd.read_csv('E:\\ViralPred\\weng_network\\rpp_NT36001percent10iters.csv',header=None)
    print res_df.head()
    print type(res_df[0].values[0])


    names = res_df[0].values

    fs_df = fs_df.loc[names]
    fs_list = fs_df[3].values.astype('int').tolist()
    print np.max(fs_list)
    #col_no = 2
    for col_no in xrange(1,6):
        res_list = []
        col_numbers_for_this_loop = [col_no+5*j1 for j1 in xrange(0,5)]
        five_col_mat = res_df.iloc[:,col_numbers_for_this_loop].values.tolist()
        print five_col_mat[0]
        for row in five_col_mat:
            x = np.median(row)

            if np.isnan(x):
                x = 50.0
            elif x > np.max(fs_list):
                x = np.max(fs_list)
            res_list.append(x)


        y_true = []
        y_pred = []
        #topkacc_vector = np.zeros(10)
        #tau_vector = np.zeros(10)
        #j = 0
        for i in xrange(0,len(fs_list)):
            #if res_list[i] == 'inf' or res_list[i] < 0:
                #continue
            y_true.append(fs_list[i])
            y_pred.append(res_list[i])

        print len(y_true)

        ape = APE(y_pred,y_true)
        rmse = RMSE(y_pred,y_true)
        rmlse = RMSLE(y_pred,y_true)
        topkacc,tau = TOPK_ACC_TAU(y_pred,y_true,K=100)

        print ape,rmse,rmlse,topkacc,tau
    '''
    for x in res_df[col_no].values:
        #res_list.append(int(x.replace('[','').replace(']','').split('.')[0]))
        if np.isnan(x):
            x = 50.0
        elif x > np.max(fs_list):
            x = np.max(fs_list)
        res_list.append(x)
    #.astype(np.float).tolist()
    '''