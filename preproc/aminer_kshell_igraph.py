# This file computes some centralities using the igraph package

#import snap
import pandas as pd
import igraph
import csv
from time import time

if __name__ == '__main__':

    t_0 = time()

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

    pr_wrt = csv.writer(open('E:\ViralPred\\weibo_network\\pagerank.csv','wb'))
    for i in xrange(len(pagerank)):
        pr_wrt.writerow([i,pagerank[i]])
    #evcent = g.evcent()
    #print type(evcent),len(evcent)
    #k_shell = g.shell_index(mode='OUT')
    #print type(k_shell),len(k_shell)

    #evcent_wrt = csv.writer(open('E:\ViralPred\\weibo_network\\evcent.csv','wb'))
    #kshell_wrt = csv.writer(open('E:\ViralPred\\weibo_network\\kshell.csv','wb'))

    #for i in xrange(len(evcent)):
    #    evcent_wrt.writerow([i,evcent[i]])
