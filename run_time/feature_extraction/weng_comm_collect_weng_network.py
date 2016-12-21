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
import random

def load_louvain():
    louvain_df = pd.read_csv('E:\\ViralPred\\weng_network\\Louvain.txt',header=None)
    return louvain_df

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
    with open('E:\\ViralPred\\weng_network\\timeline_tag_rt.anony.dat','rb') as rf:
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

def load_graph(max_uid_index):
    g = igraph.Graph()
    #g = g.as_directed()

    #g.AddNode(100000)
    #map_id_df = pd.read_csv('')
    edges = []
    with open('E:\\ViralPred\\weng_network\\follower_gcc.anony.dat','rb') as f:
        #next(f)
        cnt =0
        distinct_nodes = {}

        for line in f:
            ls = line[:-1].split(' ')
            uid1 = int(ls[0])
            uid2 = int(ls[1])
            if uid1 not in distinct_nodes:
                distinct_nodes[uid1] = 1
            if uid2 not in distinct_nodes:
                distinct_nodes[uid2] = 1
            edges.append((uid1,uid2))

        f.close()
    largest_node_index = max(distinct_nodes.keys())
    print largest_node_index
    g.add_vertices(max(largest_node_index+1,max_uid_index))
    g.add_edges(edges)

    print g.is_directed()
    print igraph.summary(g)

    return g

def cm_based_measures(adp_cms):
    set_adp_cms = set(adp_cms)

    num_adp_cms = len(set_adp_cms)

    adp_cms_ent,adp_cms_gini = ent(adp_cms)
    cm_feature_list = [num_adp_cms,
            adp_cms_ent]

    print cm_feature_list
    return cm_feature_list

def ent(cms):
    if len(set(cms)) == 1:
        return 0.0,0.0
    pk = np.divide(Counter(cms).values(),float(len(cms)))
    gini = 1 - np.sum(np.square(pk))
    return entropy(pk,base=2),gini

def distance_features(g,sg,adps):
    #need to debug this function
    shortest_dists = []

    for i in xrange(0,len(adps)-1):
        last_adp = adps[i]
        adp = adps[i+1]
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

def basic_network_features(g,adps):
    first_neighbors = []
    for adp in adps:
        first_neighbors.extend(g.neighbors(adp))
    #print first_neighbors
    first_neighbor_set = set(first_neighbors)
    first_neighbor_size = len(first_neighbor_set)
    second_neighbors = []

    for fn in first_neighbor_set:
        second_neighbors.extend(g.neighbors(fn))
    second_neighbor_set = set(second_neighbors)
    second_neighbor_size = len(second_neighbor_set)

    return first_neighbor_size,second_neighbor_size

def inner_cm_interaction(mentions,retweets,louvain_vector,obs_time):

    inner = 0
    inter = 0

    for m in mentions:
        if m[0] <= obs_time:
            if m[1] >= len(louvain_vector) or m[2] >= len(louvain_vector):
                inter += 1
            elif louvain_vector[m[1]] != louvain_vector[m[2]]:
                inter += 1
            else:
                inner += 1
        else:
            break

    for r in retweets:
        if r[0] <= obs_time:
            if r[1] >= len(louvain_vector) or r[2] >= len(louvain_vector):
                inter += 1
            elif louvain_vector[r[1]] != louvain_vector[r[2]]:
                inter += 1
            else:
                inner += 1

    if inner+inter == 0:
        return 0
    else:
        inner_ratio = float(inner)/(float(inner+inter))

    return inner_ratio

def feature_extraction(cascade,g,louvain_df,louvain_vector,mentions,retweets,N=50):
    print cascade.head()
    times = cascade[3].values[:N]
    obs_time = times[-1] #for pick interaction pairs
    times = np.subtract(times,times[0])

    step_times = []
    for m in xrange(0,len(times)-1):
        step_times.append(times[m+1] - times[m])

    adps = cascade[2].values.astype('int').tolist()[:N]
    sg = g.subgraph(adps,implementation="create_from_scratch")
    adp_cms = louvain_df.loc[adps][0].values.tolist()

    avg_steptime = np.mean(step_times)
    std_steptime = np.std(step_times)
    cv_steptime = float(std_steptime)/float(avg_steptime)


    avg_dist,cv_dist,diameter = distance_features(g,sg,adps)

    inner_ratio = inner_cm_interaction(mentions,retweets,louvain_vector,obs_time)

    feature_list = []

    first_neisize,second_neisize = basic_network_features(g,adps)

    feature_list.extend([first_neisize,second_neisize])
    feature_list.extend([avg_dist,cv_dist,diameter])
    feature_list.extend(cm_based_measures(adp_cms))
    feature_list.append(inner_ratio)
    feature_list.extend([avg_steptime,cv_steptime])

    print feature_list
    return feature_list

if __name__ == '__main__':
    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/Twitter_Indiana')
    conn = engine.connect()

    q = "WITH ORDERED AS(SELECT \"line_index\",\"hashtag\",\"uid\",\"time\" ,\
     ROW_NUMBER() OVER (PARTITION BY \"line_index\" ORDER BY \"time\" ASC) \
       AS rn FROM rt_df_50)SELECT \"line_index\",\"hashtag\",\"uid\",\"time\" FROM ORDERED WHERE rn <= 50;"

    res = conn.execute(q)
    rt_df = pd.DataFrame(res.fetchall())

    max_uid_index = np.max(rt_df[2].values)

    cascades = rt_df.groupby(0)

    #outnei_d = load_outnei_d()
    louvain_df = load_louvain()
    louvain_vector = louvain_df[0].values

    mention_dict,retweet_dict = load_interactions()

    #of = open('E:\\ViralPred\\weng_network\\weng_features_1.3.csv','wb')
    #wrt = csv.writer(of)

    time_of = open('E:\\ViralPred\\weng_network\\run_time\\weng_model.csv','ab')
    time_wrt = csv.writer(time_of)
    g = load_graph(max_uid_index+1)

    time_list = []

    for name,cascade in cascades:


        row = [name]
        if len(cascade) >= 50:

            if random.random() < 0.9:
                continue

            hashtag = cascade[1].values[0]
            t_0 = time()
            for N in [50]:
                if hashtag in mention_dict:
                    mentions = mention_dict[hashtag]
                else:
                    mentions = []

                if hashtag in retweet_dict:
                    retweets = retweet_dict[hashtag]
                else:
                    retweets = []
                row.extend(feature_extraction(cascade,g,louvain_df,louvain_vector,mentions,retweets,N=N))
            time_list.append(time()-t_0)
            #time_wrt.writerow(row)
            #print row
            #print 'time for this loop is %f' %(time()-t_0)
    time_mean = np.mean(time_list)
    time_median = np.median(time_list)
    time_std = np.std(time_list)
    time_wrt.writerow([time_mean,time_median,time_std])
    #of.close()











