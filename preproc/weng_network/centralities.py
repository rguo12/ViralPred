#import snap
import pandas as pd
import igraph
import csv

from time import time

if __name__ == '__main__':
    g = igraph.Graph()
    #g = g.as_directed()

    #g.AddNode(100000)
    #map_id_df = pd.read_csv('')
    edges = []
    t_0 = time()
    with open('E:\\ViralPred\\weng_network\\follower_gcc.anony.dat','rb') as f:
        #next(f)
        cnt =0
        distinct_nodes = {}

        for line in f:
            ls = line[:-1].split(' ')
            #print ls
            #break
            #print ls
            #follower =
            #for i in xrange(1,len(ls)/2):
            uid1 = int(ls[0])
            uid2 = int(ls[1])
            if uid1 not in distinct_nodes:
                distinct_nodes[uid1] = 1
            if uid2 not in distinct_nodes:
                distinct_nodes[uid2] = 1
            edges.append((uid1,uid2))

        f.close()
    largest_node_index = max(distinct_nodes.keys())
    print largest_node_index
    g.add_vertices(largest_node_index+1)
    g.add_edges(edges)


    #print g.is_directed()
    print igraph.summary(g)
    print 'loading graph takes %f' %(time()-t_0)

    t_0 = time()
    evcent = g.evcent()
    print 'evcent computation takes %f' %(time()-t_0)

    #print type(evcent),len(evcent)
    t_0 = time()
    k_shell = g.shell_index()
    print 'kshell computation takes %f' %(time()-t_0)

    t_0 = time()
    outdeg = g.outdegree()
    print 'outdeg computation takes %f' %(time()-t_0)
    #print type(k_shell),len(k_shell)
    #print type(outdeg),len(outdeg)

    t_0 = time()
    pagerank = g.pagerank()
    print 'pagerank computation takes %f' %(time()-t_0)

    #evcent_wrt = csv.writer(open('E:\ViralPred\\weng_network\\virality2013\\evcent.csv','wb'))
    #kshell_wrt = csv.writer(open('E:\ViralPred\\weng_network\\virality2013\\kshell.csv','wb'))
    pr_wrt = csv.writer(open('E:\ViralPred\\weng_network\\pagerank.csv','wb'))
    '''
    outdeg_wrt = csv.writer(open('E:\ViralPred\\weng_network\\outdeg.csv','wb'))

    #for i in xrange(len(evcent)):
    #    evcent_wrt.writerow([i,evcent[i]])

    for i in xrange(len(outdeg)):
        #kshell_wrt.writerow([i,k_shell[i]])
        outdeg_wrt.writerow([i,outdeg[i]])
    '''

    for i in xrange(len(pagerank)):
        pr_wrt.writerow([i,pagerank[i]])