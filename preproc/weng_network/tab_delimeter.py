if __name__ == '__main__':
    f = open('E:\ViralPred\\weng_network\\virality2013\\follower_gcc.anony.dat','rb')
    of = open('E:\ViralPred\\weng_network\\virality2013\\edgelist_tab.txt','wb')
    for line in f:
        ls = line[:-1].split(' ')
        new_line = ls[0] + '\t' + ls[1] + '\n'
        of.write(new_line)
