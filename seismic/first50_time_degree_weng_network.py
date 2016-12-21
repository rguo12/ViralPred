from sqlalchemy import create_engine
import pandas as pd
import csv
import numpy as np
def get_fifty_nodes(conn):
    q = "WITH ORDERED AS(SELECT \"line_index\",\"hashtag\",\"uid\",\"time\" ,\
     ROW_NUMBER() OVER (PARTITION BY \"line_index\" ORDER BY \"time\" ASC) \
       AS rn FROM rt_df_50)SELECT \"line_index\",\"hashtag\",\"uid\",\"time\" FROM ORDERED WHERE rn <= 50;"

    res = conn.execute(q)
    fiftynodes_df = pd.DataFrame(res.fetchall())
    #fiftynodes_df = fiftynodes_df.set_index(0)

    return fiftynodes_df

if __name__ == '__main__':
    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/Twitter_Indiana')
    conn = engine.connect()

    out_deg_df = pd.read_csv('E:\\ViralPred\\weng_network\\outdeg.csv',sep=',',header=None,index_col=0)
    print out_deg_df.head()

    fifty_nodes_df = get_fifty_nodes(conn)
    print fifty_nodes_df.head()

    #time since root and out degree for each node
    fifty_nodes_groups = fifty_nodes_df.groupby(0)

    of = open('E:\\ViralPred\\weng_network\\seismic_50.csv','wb')
    wrt = csv.writer(of)

    for rtmind,fifty_nodes_g in fifty_nodes_groups:
        time_since_root = fifty_nodes_g[3].values - fifty_nodes_g[3].values[0]
        out_deg = out_deg_df.loc[fifty_nodes_g[2].values.astype('int')][1].values
        #print time_since_root
        #print out_deg
        #break
        for i in xrange(0,len(time_since_root)):
            if np.isnan(out_deg[i]):
                row = [rtmind,time_since_root[i],0]
            else:
                row = [rtmind,time_since_root[i],out_deg[i]]
            wrt.writerow(row)
    of.close()
