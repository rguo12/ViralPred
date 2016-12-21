import pandas as pd
from collections import defaultdict
if __name__ == '__main__':

    all_rts = []
    all_mens = []

    #read profile matrix(name-->uid mapping)
    profile_df1 = pd.read_csv('E:\\ViralPred\\weibo_network\\weibocontents\\profile_df.csv'
                              ,index_col='0')
    profile_df2 = pd.read_csv('E:\\ViralPred\\weibo_network\\weibocontents\\profile_df_2.csv',
                              index_col='0')

    print profile_df1.head()

    uids = profile_df1['1'].values.tolist()
    names = profile_df1['8'].values.tolist()

    uids.extend(profile_df2['1'].values.tolist())
    names.extend(profile_df2['8'].values.tolist())

    name_uid_d = defaultdict(str)
    for i in xrange(0,len(uids)):
        #if name_uid_d[names[i]] != '':
        #    print 'duplicate user name!'
        name_uid_d[names[i]] = uids[i]

    with open('E:\\ViralPred\\weibo_network\\weibocontents\\Retweet_Content.txt','rb') as f:
        original_uid = -1
        rter_uid = -1
        last_line_type = -1 # 0: original msg line 1: rt msg line 2:retweet line 3: mention line
        rt_flag = 0 #if there is a rt line in the (last) retweet
        for line in f:
            line = line[:-2]
            ls = line.split(' ')
            #print ls
            if len(ls) == 4 and ':' in ls[2] and '-' in ls[2]:
                #new tweet
                original_uid = int(ls[1])
                last_line_type = 0
            elif len(ls) == 3 and ':' in ls[1] and '-' in ls[1]:
                last_rter_uid = rter_uid
                if rt_flag==0:
                    all_rts.append([original_uid,last_rter_uid])
                rter_uid = ls[0]
                last_line_type = 1
                rt_flag = 0
            if ls[0] == 'retweet':
                user_names = set(ls[1:])
                for name in user_names:
                    #map name to uid and save the pair
                    if name in name_uid_d:
                        uid = name_uid_d[name]
                        pair = [uid,rter_uid]
                        all_rts.append(pair)
                last_line_type = 2
                rt_flag = 1
            if ls[0] == '@':
                mention_names = set(ls[1:])
                for name in mention_names:
                    if name in name_uid_d:
                        uid = name_uid_d[name]
                        pair = [rter_uid,uid]
                        all_mens.append(pair)
                last_line_type = 3
            #if len(ls) == 1 and ls[0] == '':
            #    if last_line_type == 1:
            #        all_rts.append([original_uid,rter_uid])
            #    last_line_type += 1

    of_rt = open('E:\\ViralPred\\weibo_network\\weibocontents\\all_rts.txt','wb')
    for rt in all_rts:
        of_rt.write(str(rt[0])+'\t'+str(rt[1])+'\n')

    of_men = open('E:\\ViralPred\\weibo_network\\weibocontents\\all_mens.txt','wb')
    for men in all_mens:
        of_rt.write(str(men[0])+'\t'+str(men[1])+'\n')





