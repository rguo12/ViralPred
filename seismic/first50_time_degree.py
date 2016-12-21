from sqlalchemy import create_engine
import pandas as pd
import csv
def get_fifty_nodes(conn):
    q = "WITH ORDERED AS(SELECT \"0\",   \"1\",   \"2\", \
      ROW_NUMBER() OVER (PARTITION BY \"0\" ORDER BY \"2\" ASC) \
       AS rn FROM rt_df)SELECT \"0\",\"1\",\"2\" FROM ORDERED WHERE rn <= 50;"

    res = conn.execute(q)
    fiftynodes_df = pd.DataFrame(res.fetchall())
    #fiftynodes_df = fiftynodes_df.set_index(0)

    return fiftynodes_df

if __name__ == '__main__':
    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/Aminer')
    conn = engine.connect()

    out_deg_df = pd.read_csv('E:\\ViralPred\\weibo_network\\outdeg.txt',sep=',',header=None,index_col=0)
    print out_deg_df.head()

    fifty_nodes_df = get_fifty_nodes(conn)

    #time since root and out degree for each node
    fifty_nodes_groups = fifty_nodes_df.groupby(0)

    of = open('E:\\ViralPred\\weibo_network\\seismic_50.csv','wb')
    wrt = csv.writer(of)

    for rtmind,fifty_nodes_g in fifty_nodes_groups:
        time_since_root = fifty_nodes_g[2].values - fifty_nodes_g[2].values[0]
        out_deg = out_deg_df.loc[fifty_nodes_g[1].values.astype('int')][1].values
        #print time_since_root
        #print out_deg
        #break
        for i in xrange(0,len(time_since_root)):
            row = [rtmind,time_since_root[i],out_deg[i]]
            wrt.writerow(row)
    of.close()
