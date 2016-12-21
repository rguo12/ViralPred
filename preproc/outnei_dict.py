from collections import defaultdict
import cPickle
if __name__ == '__main__':
    f = open('E:\\ViralPred\\weng_network\\edgelist_tab.txt','rb')
    outnei_d = defaultdict(list)
    for line in f:
        ls = line[:-1].split('\t')
        followee = int(ls[0])
        follower = int(ls[1])
        outnei_d[followee].append(follower)

    of = open('E:\\ViralPred\\weng_network\\outnei_d.pkl','wb')
    cPickle.dump(outnei_d,of,protocol=2)
    #print outnei_d
