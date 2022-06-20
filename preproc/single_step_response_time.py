# also computes response time distribution

from sqlalchemy import create_engine
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import cPickle
#For seismic kernel

def load_innei_d():
    f = open('E:\\ViralPred\\weibo_network\\weibo_network.txt','rb')
    next(f)

    innei_d = {}

    for line in f:
        ls = line[:-2].split('\t')
        uid = int(ls[0])
        innei_d[uid] = []
        for i in xrange(1,len(ls)/2):
            followee = int(ls[2*i])
            innei_d[uid].append(followee)

    return innei_d

if __name__ == '__main__':
    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/Aminer')
    conn = engine.connect()
    q = 'SELECT \"0\",\"1\",\"2\" FROM rt_df'
    res = conn.execute(q)

    rt_df = pd.DataFrame(res.fetchall())

    #outnei_f = open('E:\\ViralPred\\weibo_network\\outnei_d.pkl','rb')
    #outnei_d = cPickle.load(outnei_f)

    innei_d = load_innei_d()

    cascades = rt_df.groupby(0)
    res_time_list = []

    for name, cascade in cascades:
        # for each cascade size > 50
        if len(cascade) < 50:
            continue
            
        # active nodes
        adps = cascade[1].values.astype('int').tolist()

        times = cascade[2].values
        
        # compute time relative to the beginning of the cascade
        times = np.subtract(times,times[0])
        times = times.tolist()
        
        #linear search for the first active in_nei
        adps_done = {} # save those which are not active anymore
        done_cnt = 1

        for i in xrange(0,len(adps)):
            adp = adps[i]
            innei = set(innei_d[adp]) # neighbors of i-th active node
            t_i = times[i]
            #previous_adps = adps[:i]
            #innei.intersection(previous_adps)
            
            for j in xrange(0,i):
                # for each node activated before node i
                if adps[i-j] in innei and adps[i-j] not in adps_done:
                    adps_done[adps[i-j]] = 1
                    
                    prev_time = times[i-j]
                    res_time = t_i - prev_time
                    res_time_list.append(res_time)

        '''
        for i in xrange(0,len(adps)):
            potential_innei = adps[i]
            outneis = outnei_d[potential_innei]
            if len(outneis) == 0:
                continue
            print type(outneis[0]),type(adps[0])
            #if done_cnt == len(adps):
            #    break
            for j in xrange(i+1,len(adps)):
                if adps[j] not in outneis:
                    res_time = times[j] - times[i]
                    res_time_list.append(res_time)
                    adps_done[j] = 1
                    done_cnt += 1
        '''
    print np.percentile(res_time_list,90)
    plt.hist(res_time_list,bins=200,normed=True,log=True)
    of = open('E:\\ViralPred\\weibo_network\\single_step_res_time_latest.txt','wb')
    for rt in res_time_list:
        of.write(str(rt)+'\n')
    plt.show()
