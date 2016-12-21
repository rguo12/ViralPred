import pandas as pd
from sqlalchemy import create_engine
import csv
TH = 0
if __name__ == '__main__':
    meme_file = open('E:\ViralPred\\weng_network\\timeline_tag.anony.dat','rb')
    line_index = 0

    all_rows = []
    fs_all_rows = []

    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/Twitter_Indiana')
    conn = engine.connect()

    diffusion_f = open('E:\\ViralPred\\weng_network\\diffusion.csv','wb')
    diffusion_wrt = csv.writer(diffusion_f)


    for line in meme_file:
        ls = line[:-1].split(' ')
        uid_d = {}
        #print ls
        #break
        if len(ls) - 1 >= TH:
            rows = []
            for i in xrange(1,len(ls)):
                ele = ls[i].split(',')
                t = int(ele[0])
                uid = int(ele[1])
                if uid not in uid_d:
                    uid_d[uid] = 1
                    row = [line_index,ls[0],uid,t]
                    diffusion_wrt.writerow(row)
                    #rows.append(row)
        #if len(uid_d) >= TH:
            #all_rows.extend(rows)
            #fs_row = [line_index,ls[0],len(uid_d)]
            #fs_all_rows.append(fs_row)
            #print ls[0]
        line_index += 1
        #if line_index >= 50000:
        #    break

    #print len(all_rows)
    #df = pd.DataFrame(all_rows,columns=['line_index','hashtag','uid','time'])
    #df.to_sql('rt_df_50',conn)


    #fs_df = pd.DataFrame(fs_all_rows,columns=['line_index','hashtag','fs'])
    #fs_df.to_sql('fs_df_50',conn)

    #df.to_csv('E:\\')
