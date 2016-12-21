from sqlalchemy import create_engine
import pandas as pd
from urllib import quote_plus as url_quote

def load_diffusion():
    f = open('E:\\ViralPred\\diffusion\\diffusion\\repost_data.txt','rb')

    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/Aminer')
    conn = engine.connect()

    rt_array = []
    fs_array = []

    cas_cnt = 0

    for line in f:

        ls = line[:-1].split('\t')
        #print line[:-1]
        if ls[0] == str(cas_cnt):
            #went into new cascade
            cas_cnt += 1

            cas_id = int(ls[0])
            cas_size = int(ls[1])
            cas_row = [cas_id,cas_size]
            fs_array.append(cas_row)
        else:
            #nodes and timestamps
            uid = ls[1]
            timestamp = int(ls[0])

            row = [cas_id,uid,timestamp]
            rt_array.append(row)

    fs_df = pd.DataFrame(fs_array)
    rt_df = pd.DataFrame(rt_array)

    fs_df.to_sql('fs_df',conn)
    rt_df.to_sql('rt_df',conn)

if __name__ == '__main__':
    pass
    #load_diffusion()