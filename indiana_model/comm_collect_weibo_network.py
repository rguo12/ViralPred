import cPickle
from sqlalchemy import create_engine
from urllib import quote_plus as urlquote
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import entropy
import csv
import igraph
from time import time
from random import random

def load_louvain():
    louvain_df = pd.read_csv('E:\\ViralPred\\weibo_network\\Louvain.txt',header=None)
    return louvain_df
'''
def load_interactions():

    mention_dict = {}

    with open('E:\\ViralPred\\weng_network\\timeline_tag_men.anony.dat','rb') as mf:
        for line in mf:
            ls= line[:-1].split(' ')
            mention_dict[ls[0]] = []
            for i in xrange(1,len(ls)):
                eles = ls[i].split(',')
                time = int(eles[0])
                user1 = int(eles[1])
                user2 = int(eles[2])
                tuple = [time,user1,user2]
                mention_dict[ls[0]].append(tuple)
        #mf.close()

    retweet_dict = {}
    with open('E:\\ViralPred\\weng_network\\timeline_tag_men.anony.dat','rb') as rf:
        for line in rf:
            ls= line[:-1].split(' ')
            retweet_dict[ls[0]] = []
            for i in xrange(1,len(ls)):
                eles = ls[i].split(',')
                time = int(eles[0])
                user1 = int(eles[1])
                user2 = int(eles[2])
                tuple = [time,user1,user2]
                retweet_dict[ls[0]].append(tuple)
        #rf.close()

    return mention_dict,retweet_dict
'''
def load_graph(max_uid_index_in_cascades):
    g = igraph.Graph()
    g = g.as_directed()

    #g.AddNode(100000)
    #map_id_df = pd.read_csv('')
    edges = []
    max_uid_index = 0
    with open('E:\\ViralPred\\weibo_network\\edgelist.txt','rb') as f:
        ln_cnt = 0
        for line in f:
            ls = line[:-1].split('\t')
            uid1 = int(ls[0])
            uid2 = int(ls[1])
            if max(uid1,uid2) > max_uid_index:
                max_uid_index = max(uid1,uid2)
            edges.append((uid1,uid2))
            ln_cnt += 1
            #if ln_cnt > 10000:
            #    break

        f.close()

    #print largest_node_index
    g.add_vertices(max(max_uid_index,max_uid_index_in_cascades)+1)
    g.add_edges(edges)

    print g.is_directed()
    print igraph.summary(g)

    return g


def load_outnei_d():
    f = open('E:\\ViralPred\\weibo_network\\outnei_d.pkl','rb')
    outnei_d = cPickle.load(f)
    return outnei_d

def cm_based_measures(adp_cms):
    set_adp_cms = set(adp_cms)
    num_adp_cms = len(set_adp_cms)
    adp_cms_ent,adp_cms_gini = ent(adp_cms)
    cm_feature_list = [num_adp_cms,adp_cms_ent]

    print cm_feature_list
    return cm_feature_list

def ent(cms):
    if len(set(cms)) <= 1:
        return 0.0,0.0
    pk = np.divide(Counter(cms).values(),float(len(cms)))
    gini = 1 - np.sum(np.square(pk))
    return entropy(pk,base=2),gini

def distance_features(sg,adps):
    #need to debug this function
    shortest_dists = []

    for i in xrange(0,len(adps)-1):
        #last_adp = adps[i]
        #adp = adps[i+1]
        sps = sg.shortest_paths_dijkstra(source=i,target=i+1)[0][0]
        if np.isinf(sps) or np.isnan(sps):
            continue

        #if len(sps) > 0:
        #    shortest_dist = np.min(sps)
        #    shortest_dists.append(shortest_dist)
        shortest_dists.append(sps)

    if len(shortest_dists) != 0:
        avg_dist = np.mean(shortest_dists)
        std_dist = np.std(shortest_dists)
    else:
        avg_dist = 50.0
        std_dist = 0.0

    cv_dist = float(std_dist)/float(avg_dist)

    diameter = sg.diameter(directed=False)

    return avg_dist,cv_dist,diameter

def basic_network_features(outnei_d,adps):
    first_neighbors = []
    for adp in adps:
        if adp in outnei_d:
            first_neighbors.extend(outnei_d[adp])
    #print first_neighbors
    first_neighbor_set = set(first_neighbors)
    first_neighbor_set = first_neighbor_set - set(adps)
    first_neighbor_size = len(first_neighbor_set)

    second_neighbors = []

    for fn in first_neighbor_set:
        if fn in outnei_d:
            second_neighbors.extend(outnei_d[fn])
    second_neighbor_set = set(second_neighbors) - first_neighbor_set
    second_neighbor_size = len(second_neighbor_set)

    return first_neighbor_size,second_neighbor_size
'''
def inner_cm_interaction(mentions,retweets,louvain_df,obs_time):

    inner = 0
    inter = 0

    for m in mentions:
        if m[0] <= obs_time:
            if louvain_df.loc[m[1]][0] == louvain_df.loc[m[2]][0]:
                inner += 1
            else:
                inter += 1
        else:
            break

    for r in retweets:
        if r[0] <= obs_time:
            if louvain_df.loc[r[1]][0] == louvain_df.loc[r[2]][0]:
                inner += 1
            else:
                inter += 1
    if inner+inter == 0:
        return 0
    else:
        inner_ratio = float(inner)/(float(inner+inter))

    return inner_ratio
'''
def feature_extraction(cascade,g,outnei_d,louvain_df,N=50):
    print cascade.head()
    times = cascade[2].values[:N]
    obs_time = times[-1] #for pick interaction pairs
    times = np.subtract(times,times[0])

    step_times = []
    for m in xrange(0,len(times)-1):
        step_times.append(times[m+1] - times[m])

    adps = cascade[1].values.astype('int').tolist()[:N]
    sg = g.subgraph(adps,implementation="create_from_scratch")
    adp_cms = louvain_df.loc[adps][0].values.tolist()

    avg_steptime = np.mean(step_times)
    std_steptime = np.std(step_times)
    cv_steptime = float(std_steptime)/float(avg_steptime)


    avg_dist,cv_dist,diameter = distance_features(sg,adps)

    #inner_ratio = inner_cm_interaction(mentions,retweets,louvain_df,obs_time)

    feature_list = []

    first_neisize,second_neisize = basic_network_features(outnei_d,adps)

    feature_list.extend([first_neisize,second_neisize])
    feature_list.extend([avg_dist,cv_dist,diameter])
    feature_list.extend(cm_based_measures(adp_cms))
    #feature_list.append(inner_ratio)
    feature_list.extend([avg_steptime,cv_steptime])

    print feature_list
    return feature_list

if __name__ == '__main__':
    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/Aminer')
    conn = engine.connect()

    '''
    q = "WITH ORDERED AS(SELECT \"line_index\",\"hashtag\",\"uid\",\"time\" ,\
     ROW_NUMBER() OVER (PARTITION BY \"line_index\" ORDER BY \"time\" ASC) \
       AS rn FROM rt_df_50)SELECT \"line_index\",\"hashtag\",\"uid\",\"time\" FROM ORDERED WHERE rn <= 50;"
    '''

    q = "WITH ORDERED AS(SELECT \"0\",   \"1\",   \"2\", \
      ROW_NUMBER() OVER (PARTITION BY \"0\" ORDER BY \"2\" ASC) \
       AS rn FROM rt_df)SELECT \"0\",\"1\",\"2\" FROM ORDERED WHERE rn <= 50;"

    #0 is rtmid, 1 is uid, 2 is time

    res = conn.execute(q)
    rt_df = pd.DataFrame(res.fetchall())
    print rt_df.head()

    max_uid_index = np.max(rt_df[1].values.astype('int'))

    cascades = rt_df.groupby(0)

    #outnei_d = load_outnei_d()
    louvain_df = load_louvain()

    #mention_dict,retweet_dict = load_interactions()

    #of = open('E:\\ViralPred\\weibo_network\\weng_features_1.0.csv','wb')
    #wrt = csv.writer(of)

    g = load_graph(max_uid_index)
    outnei_d = load_outnei_d()

    time_of = open('E:\\ViralPred\\weibo_network\\run_time\\weng_model.csv','ab')
    time_wrt = csv.writer(time_of)
    time_list = []

    for i in xrange(0,10):
        t_0 = time()
        for name,cascade in cascades:
            if random() < 0.99: # 1% samples only, that's 993 samples
                continue

            #t_0 = time()
            row = [name]

            if len(cascade) >= 50:
                #hashtag = cascade[1].values[0]
                for N in [50]:
                    '''
                    if hashtag in mention_dict:
                        mentions = mention_dict[hashtag]
                    else:
                        mentions = []

                    if hashtag in retweet_dict:
                        retweets = retweet_dict[hashtag]
                    else:
                        retweets = []
                    '''
                    row.extend(feature_extraction(cascade,g,outnei_d,louvain_df,N=N))
                #wrt.writerow(row)
                #print row
            #print 'time for this loop is %f' %(time()-t_0)

        #time()-t_0
        time_list.append(time()-t_0)
    #of.close()

    time_wrt.writerow([np.mean(time_list),np.median(time_list),np.std(time_list)])









