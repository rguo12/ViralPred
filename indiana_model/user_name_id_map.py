#collect user profile as a array
import pandas as pd
import csv
if __name__ == '__main__':
    with open('E:\\ViralPred\\weibo_network\\weibocontents\\user_profile2.txt','rb') as f1:
        ln_cnt = 0
        profile_array = [[None]*14]
        for line in f1:
            if ln_cnt >= 15:
                row_index = ln_cnt/15 - 1
                col_index = ln_cnt % 15
                row = profile_array[row_index]
                if col_index == 14:
                    profile_array.append([None]*14)
                elif col_index in {0,1,2,4,6,7,11,12}:
                    row[col_index] = int(line[:-1].replace('\r',''))
                #elif col_index == 5:
                    #Chinese handling
                    #print line[:-1]
                else:
                    row[col_index] = line[:-1].replace('\r','')
                    # unicode(line[:-1].replace('\r',''),'utf-8')
                #print row
            ln_cnt += 1
            #if ln_cnt == 150:
                #break

    profile_df = pd.DataFrame(profile_array)
    print profile_df.head()
    profile_df.to_csv('E:\\ViralPred\\weibo_network\\weibocontents\\profile_df_2.csv')




