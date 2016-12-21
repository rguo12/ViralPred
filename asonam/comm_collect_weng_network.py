import cPickle
from sqlalchemy import create_engine
from urllib import quote_plus as urlquote
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import entropy
import csv
from time import time

THRESHOLD = 1800

def load_outnei_d():
    f = open('E:\\ViralPred\\weng_network\\outnei_d.pkl','rb')
    outnei_d = cPickle.load(f)
    return outnei_d

def load_louvain():
    louvain_df = pd.read_csv('E:\\ViralPred\\weng_network\\Louvain.txt',header=None)
    return louvain_df

def cm_based_measures(adp_cms,fnt_cms,na_cms):
    #number of cms

    set_adp_cms = set(adp_cms)
    set_fnt_cms = set(fnt_cms)
    set_na_cms = set(na_cms)

    num_adp_cms = len(set_adp_cms)
    num_fnt_cms = len(set_fnt_cms)
    num_na_cms = len(set_na_cms)

    adp_cms_ent,adp_cms_gini = ent(adp_cms)
    fnt_cms_ent,fnt_cms_gini = ent(fnt_cms)
    na_cms_ent,na_cms_gini = ent(na_cms)

    af_ol = overlap(set_adp_cms,set_fnt_cms)
    ana_ol = overlap(set_adp_cms,set_na_cms)
    fna_ol = overlap(set_fnt_cms,set_na_cms)

    na_not_a_cms = [x for x in na_cms if x not in set_adp_cms]
    f_not_a_cms = [x for x in fnt_cms if x not in set_adp_cms]
    f_not_ana_cms = [x for x in f_not_a_cms if x not in set_na_cms]

    na_not_a_ent,na_not_a_gini = ent(na_not_a_cms)
    f_not_a_ent,f_not_a_gini = ent(f_not_a_cms)
    f_not_ana_ent,f_not_ana_gini = ent(f_not_ana_cms)

    cm_feature_list = [num_adp_cms,num_fnt_cms,num_na_cms,
            adp_cms_ent,fnt_cms_ent,na_cms_ent,
            adp_cms_gini,fnt_cms_gini,na_cms_gini,
            af_ol,ana_ol,fna_ol,
            na_not_a_ent,f_not_a_ent,f_not_ana_ent,
            na_not_a_gini,f_not_a_gini,f_not_ana_gini]
    print cm_feature_list
    return cm_feature_list

def ent(cms):
    if len(set(cms)) == 1:
        return 0.0,0.0
    pk = np.divide(Counter(cms).values(),float(len(cms)))
    gini = 1 - np.sum(np.square(pk))
    return entropy(pk,base=2),gini

def overlap(set_cms1,set_cms2):
    return len(set_cms1.intersection(set_cms2))


def feature_extraction(cascade,outnei_d,louvain_df,N=50):

    times = cascade[3].values[:N]
    times = np.subtract(times,times[0])
    adps = cascade[2].values.astype('int').tolist()[:N]
    #outneis = []
    #for adp in adps:
        #outneis.extend(outnei_d[adp])
    #outneis = list(set(outneis))
    adp_cms = louvain_df.loc[adps][0].values.tolist()
    #outneis_cms = louvain_df.loc[outneis]

    obs_time = times[-1]
    cut_off_time = obs_time - THRESHOLD

    #fnt_adps = []
    #na_adps = []
    na_outneis = []
    fnt_outneis = []

    for i in xrange(len(adps)):
        if times[i] < cut_off_time:
            na_outneis.extend(outnei_d[adps[i]])
        else:
            fnt_outneis.extend(outnei_d[adps[i]])

    set_na_outneis = set(na_outneis)
    na_outneis = list(set_na_outneis)
    fnt_outneis = [x for x in fnt_outneis if x not in set_na_outneis]
    fnt_outneis = list(set(fnt_outneis))
    #na_outneis = [outnei_d[na_adp] for na_adp in na_adps]
    #fnt_outneis = [outnei_d[fnt_adp] for fnt_adp in fnt_adps]

    na_cms = louvain_df.loc[na_outneis][0].values.tolist()
    fnt_cms = louvain_df.loc[fnt_outneis][0].values.tolist()
    avg_time = np.mean(times)
    tot_time = obs_time

    #set_adp_cms = set(adp_cms)
    #set_na_cms = set(na_cms)

    cnas = len(na_outneis)
    fnts = len(fnt_outneis)

    feature_list = []

    feature_list.extend(cm_based_measures(adp_cms,fnt_cms,na_cms))
    feature_list.extend([cnas,fnts,avg_time,tot_time])

    return feature_list

if __name__ == '__main__':
    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/Twitter_Indiana')
    conn = engine.connect()

    '''
    q = "WITH ORDERED AS(SELECT \"0\",   \"1\",   \"2\", \
      ROW_NUMBER() OVER (PARTITION BY \"0\" ORDER BY \"2\" ASC) \
       AS rn FROM rt_df_50)SELECT \"0\",\"1\",\"2\" FROM ORDERED WHERE rn <= 50;"
    '''

    q = "WITH ORDERED AS(SELECT \"line_index\",\"hashtag\",\"uid\",\"time\" ,\
     ROW_NUMBER() OVER (PARTITION BY \"line_index\" ORDER BY \"time\" ASC) \
       AS rn FROM rt_df_50)SELECT \"line_index\",\"hashtag\",\"uid\",\"time\" FROM ORDERED WHERE rn <= 50;"

    res = conn.execute(q)
    rt_df = pd.DataFrame(res.fetchall())

    cascades = rt_df.groupby(0)

    outnei_d = load_outnei_d()
    louvain_df = load_louvain()

    #of = open('E:\\ViralPred\\weng_network\\asonam_features_lambda'+str(THRESHOLD)+'.csv','wb')
    #wrt = csv.writer(of)

    for name,cascade in cascades:

        if len(cascade) < 50:
            continue
        row = [name]
        t_0 = time()
        for N in [10,20,30,40,50]:
            row.extend(feature_extraction(cascade,outnei_d,louvain_df,N=N))
        #wrt.writerow(row)
        delta_t = time()-t_0
    #of.close()











