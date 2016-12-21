#import snap
import pandas as pd
import igraph
import csv
if __name__ == '__main__':
    g = igraph.Graph()
    g = g.as_directed()
    g.add_vertices(1787443)
    #g.AddNode(100000)
    #map_id_df = pd.read_csv('')
    edges = []
    with open('E:\ViralPred\\weibo_network\\weibo_network.txt','rb') as f:
        next(f)

        cnt =0

        for line in f:
            ls = line[:-2].split('\t')
            #print ls
            #follower =
            #for i in xrange(1,len(ls)/2):
            follower = int(ls[0])
            for i in xrange(1,len(ls)/2):
                followee = int(ls[2*i])
                edges.append([followee,follower])

        f.close()
    g.add_edges(edges)

    edgelist_f = open('E:\ViralPred\\weibo_network\\edgelist.txt','wb')

    for e in edges:
        edgelist_f.write(str(e[0])+'\t'+str(e[1])+'\n')

    edgelist_f.close()

    '''
    k_shell = g.shell_index(mode='OUT')
    print type(k_shell),len(k_shell)

    kshell_wrt = csv.writer(open('E:\ViralPred\\weibo_network\\kshell.csv','wb'))

    for i in xrange(len(k_shell)):
        kshell_wrt.writerow([i,k_shell[i]])
    '''