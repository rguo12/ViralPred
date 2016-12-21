import pandas as pd
import numpy as np
PATH = 'E:\\ViralPred\\weibo_network\\'
if __name__ == '__main__':
    evcent_df = pd.read_csv(PATH+'evcent.csv',header=None)
    print evcent_df.head()
    print 'average ec %f' %np.mean(evcent_df[1].values)

    outdeg_df = pd.read_csv(PATH+'outdeg.txt',header=None)
    print outdeg_df.head()
    print 'average outdeg %f' %np.mean(outdeg_df[1].values)

    pr_df = pd.read_csv(PATH+'pr.csv',header=None)
    print pr_df.head()
    print 'average pagerank %.10f' %np.mean(pr_df[1].values)

    kshell_df = pd.read_csv(PATH+'kshell.csv',header=None)
    print kshell_df.head()
    print 'average kshell %.10f' %np.mean(kshell_df[1].values)