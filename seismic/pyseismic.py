import numpy as np
from scipy.stats import chi2
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import csv

t_cutoff = 300
c = np.power(0.1,4.0)*7.332
theta = 0.282

def memory_pdf(t,theta,cutoff,c):
    #here t is a scalar
    if t < cutoff:
        return c
    else:
        #return float(c)*np.exp((np.log(t) - np.log(cutoff))*(-(1+theta)))
        log_diff = np.log(t) - np.log(cutoff)
        #print log_diff
        log_diff = -(1+theta)*log_diff
        #print log_diff
        diff = np.exp(log_diff)*c
        #print diff
        return diff
        #return c*np.exp((np.log(t) - np.log(cutoff))*(-(1+theta)))

def memory_ccdf(T,theta,cutoff,c):
    #here T is a vector
    ccdf = np.zeros(len(T))
    '''
    for i in xrange(len(T)):
        t = T[i]
        if t < 0:
            ccdf[i] = 1
        elif t <= cutoff:
            ccdf[i] = 1 - c*t
        else:
            ccdf[i] = c*(cutoff**(1+theta))/theta*np.power(t,-theta)
            #the authors made a mistake here, the last element should be np.power(t,theta)
    '''

    T = np.asarray(T)

    index1 = np.where(T>cutoff)
    index2 = np.where(T<=cutoff)

    ccdf[index1] = np.subtract(np.ones(len(index1)),c*T[index1])
    ccdf[index2] = np.multiply(np.power(T[index2],-theta),c*cutoff**(1+theta)/theta)

    return ccdf

"""
t1: a vector of 
"""


def linear_kernel(t1,t2,ptime,slope,c):
    return c*(t2-ptime*slope*t2+slope*np.power(t2,2.0)/2.0) - c*(t1-ptime*slope*t1+slope*np.power(t1,2.0)/2.0)

def power_kernel_base(t,ptime,share_time,slope,theta,cutoff,c):
    return c*np.power(cutoff,(1+theta))*np.power((t-share_time),-theta)*(share_time*slope-theta+(theta-1)*ptime*slope-theta*slope*t+1)/((theta-1)*theta)

def power_kernel(t1,t2,ptime,share_time,slope,theta,cutoff,c):
    return power_kernel_base(t2,ptime,share_time,slope,theta,cutoff,c) - power_kernel_base(t1,ptime,share_time,slope,theta,cutoff,c)

def integral_memory_kernel(ptime,share_time,slope,window,theta,cutoff,c):
    #ptime is a scalar
    #share_time is a vector
    #window should be larger than cutoff
    integral = np.zeros(len(share_time))
    for i in xrange(0,len(share_time)):
        share_t = share_time[i]
        lk = 0
        pk = 0
        if ptime <= share_t:
            integral[i] = 0
        if ptime > share_t and ptime <= share_t + cutoff:
            lk = linear_kernel(share_t,ptime,ptime,slope,c)
        if ptime > share_t + cutoff and ptime <= share_t + window:
            lk = linear_kernel(share_t,share_t+cutoff,ptime,slope,c)
            pk = power_kernel(share_t+cutoff, ptime, ptime, share_t, slope,theta,cutoff,c)
        if ptime > share_t + window and ptime <= share_t + window + cutoff:
            lk = linear_kernel(ptime-window,share_t+cutoff,ptime,slope,c)
            pk = power_kernel(share_t+cutoff, ptime, ptime, share_t, slope,theta,cutoff,c)
        if ptime > share_t + window + cutoff:
            pk = power_kernel(ptime-window,ptime,ptime,share_t,slope,theta,cutoff,c)
        #print lk,pk
        integral[i] = lk+pk

    return integral

def get_infectious(share_time,degree,ptimes,max_window=2*60*6000,min_window=3000,min_count=25):
    share_time = sorted(share_time)

    slopes = np.divide(1.0,np.divide(ptimes,2.0))
    for i in xrange(len(slopes)):
        slope = slopes[i]
        if slope < 1.0/max_window:
            slopes[i] = 1.0/max_window
        elif slope > 1.0/min_window:
            slopes[i] = 1.0/min_window

    windows = np.divide(ptimes,2.0)
    for i in xrange(0,len(windows)):
        window = windows[i]
        if window > max_window:
            windows[i] = max_window
        elif window < min_window:
            windows[i] = min_window

    for j in xrange(0,len(ptimes)):
        ind = []
        for k in xrange(0,len(share_time)):
            share_t = share_time[k]
            if share_t >= ptimes[j]-windows[j] and share_t < ptimes[j]:
                ind.append(k)

        if (len(ind) < min_count):
            ind2 = [k for k in xrange(0,len(share_time)) if share_time[k] < ptimes[j]]
            lcv = len(ind2)
            ind = ind2[max((lcv-min_count-1),0):lcv-1]
            x = ptimes[j]
            if ind[0] < 0 or ind[0] >= len(share_time):
                print ind[0]
            y = share_time[ind[0]]
            slopes[j] = 1.0/(x-y)
            windows[j] = ptimes[j] - share_time[ind[0]]

    MI = np.zeros((len(share_time),len(ptimes)))
    for j in xrange(0,len(ptimes)):
        MI[:,j] = degree*integral_memory_kernel(ptimes[j],share_time,slopes[j],windows[j],theta,t_cutoff,c)
    infectiousness_seq = np.zeros(len(ptimes))
    p_low_seq = np.zeros(len(ptimes))
    p_up_seq = np.zeros(len(ptimes))
    share_time = share_time[1:]

    for j in xrange(0,len(ptimes)):
        share_time_tri = [t for t in share_time if t > ptimes[j] - windows[j] and t < ptimes[j]]
        #print share_time_tri
        rt_count_weighted = np.sum(np.add(np.multiply(slopes[j],np.subtract(share_time_tri,ptimes[j])),1))
        #print rt_count_weighted
        I = np.sum(MI[:,j])
        #print I
        rt_num = len(share_time_tri)
        if rt_count_weighted == 0:
            continue
        else:
            infectiousness_seq[j] = rt_count_weighted/float(I)
            p_low_seq[j] = infectiousness_seq[j]*chi2.ppf(0.05,2.0*rt_num) / (2.0*rt_num)
            p_up_seq[j] = infectiousness_seq[j]*chi2.ppf(0.95,2.0*rt_num) / (2.0*rt_num)
    infectiousness = pd.DataFrame(np.asarray([infectiousness_seq,p_up_seq,p_low_seq]).T,columns=['infectiousness','p_up','p_low'])

    return infectiousness


def pred_cascade(ptimes,infectiousness,share_time,degree,n_star=[100],alpha_t=0.5,gamma_t=0.2):
    if len(n_star) == 1:
        n_star = [n_star[0] for x in xrange(len(ptimes))]

    #features = np.zeros((len(ptimes),3))
    prediction = np.zeros((len(ptimes),1))

    for i in xrange(0,len(ptimes)):
        ind_i = [j for j in xrange(len(share_time)) if share_time[j] <= ptimes[i]]
        share_time_now = [share_time[j] for j in ind_i]
        nf_now = [degree[j] for j in ind_i]
        #print nf_now
        rt0 = len(ind_i) - 1
        #print memory_ccdf(np.subtract(ptimes[i],share_time_now),0.28255,300,6.0*np.power(10.0,-4.0))
        #print np.subtract(ptimes[i],share_time_now)
        rt1 = alpha_t*infectiousness[i]*np.dot(nf_now,memory_ccdf(np.subtract(ptimes[i],share_time_now),theta,t_cutoff,c))
        #print rt1
        product = infectiousness[i]*n_star[i]
        product = product * gamma_t
        prediction[i] = rt0 + rt1/(1.0-product)
        #np.add(rt0,np.divide(rt1,(1 - infectiousness[i]*n_star[i]*gamma_t)))
        #features[i,:] = np.asarray([rt0,rt1,infectiousness[i]])
        #if (infectiousness[i]>1.0/n_star[i]):
        #    prediction[i] = np.inf

    return prediction


if __name__ == '__main__':
    df = pd.read_csv('E:\\ViralPred\\weibo_network\\seismic_50.csv',header=None)
    print df.head()
    gb = df.groupby(0)
    fifty_time = []
    cnt = 0

    of1 = open('E:\\ViralPred\\weibo_network\\seismic_pred_weibo_300.txt','wb')
    of_wrt = csv.writer(of1)

    time_of = open('E:\\ViralPred\\weibo_network\\run_time\\seismic_prediction.csv','wb')
    time_wrt = csv.writer(time_of)
    time_list = []
    for i in xrange(0,10):
        t_0 = time()
        for name,g in gb:
            #fifty_time.append()
            if len(g) >= 50:
                rtmid = name
                end_time = np.max(g[1].values)
                #print end_time
                pred_times = np.linspace(2,10,5)*end_time
                inf_df = get_infectious(g[1].values.tolist(),g[2].values.tolist(),pred_times)
                pred = pred_cascade(pred_times, \
                                    inf_df['infectiousness'].values.tolist(), \
                                    g[1].values.tolist(),g[2].values.tolist(),\
                                    n_star=[231.3381],\
                                    alpha_t=1,gamma_t=1)
                print pred
                cnt += 1

                of_wrt.writerow([pred[k][0] for k in xrange(0,5)])
                #if cnt > 10:

        time_list.append(time()-t_0)

    time_wrt.writerow([np.mean(time_list),np.median(time_list),np.std(time_list)])
    time_of.close()
    #print np.median(np.asarray(fifty_time))
    #plt.hist(fifty_time,bins=50)
    #plt.show()