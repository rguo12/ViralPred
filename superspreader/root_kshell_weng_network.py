#rtmid(index)/rtuid/kshell/fs

from sqlalchemy import create_engine
import pandas as pd

def load_diffusion_fs(conn):


    q = 'SELECT * FROM fs_df_50 WHERE \"fs\" >= 50'
    res = conn.execute(q)

    fs_df = pd.DataFrame(res.fetchall())
    fs_df = fs_df.set_index(0)
    #print fs_df.head()

    return fs_df

def get_root_nodes(conn):
    '''
    q = "WITH ORDERED AS(SELECT \"0\",   \"1\",   \"2\", \
      ROW_NUMBER() OVER (PARTITION BY \"0\" ORDER BY \"2\" ASC) \
       AS rn FROM rt_df)SELECT \"0\",\"1\",\"2\" FROM ORDERED WHERE rn = 1;"
    '''
    q = "WITH ORDERED AS(SELECT \"line_index\",\"hashtag\",\"uid\",\"time\" ,\
     ROW_NUMBER() OVER (PARTITION BY \"line_index\" ORDER BY \"time\" ASC) \
       AS rn FROM rt_df_50)SELECT \"line_index\",\"hashtag\",\"uid\",\"time\" FROM ORDERED WHERE rn = 1;"
    res = conn.execute(q)

    rtuid_df = pd.DataFrame(res.fetchall())
    rtuid_df = rtuid_df.set_index(0)

    #print rtuid_df.head()
    return rtuid_df

if __name__ == '__main__':

    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/Twitter_Indiana')
    conn = engine.connect()

    fs_df = load_diffusion_fs(conn)
    rtuid_df = get_root_nodes(conn)

    print fs_df.head()
    print rtuid_df.head()

    rtuid_df = rtuid_df.loc[fs_df[1].values]
    kshell_df = pd.read_csv('E:\\ViralPred\\weng_network\\kshell.csv',index_col=0,header=None,sep=',')

    rtuids = rtuid_df[2].astype('int')
    #print rtuids[:5]
    rtuid_kshell = kshell_df.loc[rtuids][1]

    #print rtuid_kshell[:5]
    fs_df[4] = pd.Series(rtuid_kshell.values,index=fs_df.index)
    print len(fs_df[1])
    print fs_df.head()
    fs_df[[1,3,4]].to_csv('E:\\ViralPred\\weng_network\\superspreader_design_matrix_50.csv',header=None,index=False)
    #print design_matrix.head()


