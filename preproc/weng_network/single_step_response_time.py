from sqlalchemy import create_engine
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import cPickle
#For seismic kernel

def load_innei_d():
    f = open('E:\\ViralPred\\weng_network\\edgelist_tab.txt','rb')
    next(f)

    innei_d = defaultdict(list)

    for line in f:
        line = line[:-1]
        ls = line.split('\t')
        user1 = int(ls[0])
        user2 = int(ls[1])

        #This may cause duplicate, need to handle when read innei from this dict
        innei_d[user1].append(user2)
        innei_d[user2].append(user1)

    return innei_d

if __name__ == '__main__':
    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/Twitter_Indiana')
    conn = engine.connect()
    q = 'SELECT line_index,hashtag,uid,time FROM rt_df_50'
    res = conn.execute(q)

    rt_df = pd.DataFrame(res.fetchall())
    print rt_df.head()

    #outnei_f = open('E:\\ViralPred\\weibo_network\\outnei_d.pkl','rb')
    #outnei_d = cPickle.load(outnei_f)

    innei_d = load_innei_d()

    cascades = rt_df.groupby(0)
    res_time_list = []

    for name, cascade in cascades:
        if len(cascade) < 50:
            continue
        adps = cascade[2].values.astype('int').tolist()

        times = cascade[3].values
        times = np.subtract(times,times[0])
        times = times.tolist()
        #linear search for the first active in_nei

        adps_done = {}
        done_cnt = 1


        for i in xrange(0,len(adps)):
            adp = adps[i]
            innei = set(innei_d[adp])
            t_i = times[i]
            #previous_adps = adps[:i]
            #innei.intersection(previous_adps)
            for j in xrange(0,i):
                if adps[i-j] in innei and adps[i-j] not in adps_done:
                    adps_done[adps[i-j]] = 1
                    prev_time = times[i-j]
                    res_time = t_i - prev_time
                    res_time_list.append(res_time)
                    break

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
    of = open('E:\\ViralPred\\weng_network\\single_step_res_time_latest.txt','wb')
    for rt in res_time_list:
        of.write(str(rt)+'\n')
    plt.show()