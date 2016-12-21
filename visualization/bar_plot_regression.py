import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * round(y,1))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'



if __name__ == '__main__':

    n_groups = 5
    #TH = 0
    #DS = 'Twitter' \
    DS = 'Weibo'
    p = 'C:\\Users\\rguo12\\Google Drive\\ASONAM_2016\\Presentation\\'
    #Twitter

    if DS == 'Twitter':
        error_matrix = [[0.563865059,1199.608001,1.159043869,0.147945205],
                        [0.53906775,1813.022301,1.060571486,0.024137931],
                        [2.77295974,1892.404473,1.389350014,0.088356164],
                        [0.585146447,1826.829331,1.401704596,0],
                        [1.558220418,2543.853842,1.369333297,0.158779766]] #RPP result with 10% sample

        fig_names = ['reg_ape.pdf','reg_rmse.pdf','reg_rmlse.pdf','top_cov.pdf']
        y_lim = [3,3200,2,0.3]
        n_cols = [1,2,2,2]
        #p = 'C:\\Users\\rguo12\\Google Drive\\ASONAM_2016\\IEEEtran\\figures\\'
    if DS == 'Weibo':

        error_matrix = [[0.710645624,763.0432453,0.98321183,0.110030242],
                        [0.7181849702540517,763.84996794238805,0.99192664010702181,0.094666666666666663],
                        [1.725625422,830.5671691,1.115390722,0.143145161],
                        [0.60635218243792721,894.99258234881108,1.2966828912168815,0.0],
                        [1.791131793,2412.977879,1.410811986,0.304190162]] #RPP result with 10% sample

        fig_names = ['reg_ape_weibo.pdf','reg_rmse_weibo.pdf','reg_rmlse_weibo.pdf','top_cov_weibo.pdf']
        y_lim = [2.,2400,2.1,0.35]
        n_cols = [1,2,2,2]
        #p = 'C:\\Users\\rguo12\\Google Drive\\ASONAM_2016\\IEEEtran\\figures\\'

    error_matrix = np.asarray(error_matrix)
    error_matrix = np.transpose(error_matrix)
    #ape_vector = error_matrix[:,0]
    #rmse_vector = error_matrix[:,1]
    #rmlse_vector = error_matrix[:,2]
    #top10_cover = error_matrix[:,3]

    j = 0
    k = 0


    for error_vector in error_matrix:

        font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 38}

        matplotlib.rc('font', **font)
        fig, ax = plt.subplots(figsize=(16,9))
        plt.subplots_adjust(left=0.2, bottom=None, right=0.95, top=0.95, wspace=None, hspace=None)
        if j == 0:
            formatter = FuncFormatter(to_percent)
            plt.gca().yaxis.set_major_formatter(formatter)
        j += 1

        index = np.arange(n_groups)
        bar_width = 0.3

        opacity = 1.0
        error_config = {'ecolor': '0.0'}

        labels = ['A(SVR)','B(SVR)','C(DTR)',
              'SEISMIC','RPP']

        hatchs = ['-', '+', 'x', '\\', '.']

        colors = ['#FDFF33','#2ECC71','r','#A6ACAF','#34495E']

        rectss = []
        index = np.arange(n_groups)
        for i in xrange(0,5):

            plt.bar(index[i]*bar_width+1.5,error_vector[i] , bar_width,
                             alpha=opacity,
                             color=colors[i],
                             label=labels[i],hatch=hatchs[i] )

            #plt.ylim(0,y_lim[i])
        if len(y_lim) > 0:
            plt.ylim(0,y_lim[k])
        plt.xticks(index*bar_width+2, ['','','','',''])  #change into p,r,f1

        #font_P = FontProperties()
        #font_P.set_size('small')
        plt.legend(ncol=n_cols[k],prop={'size':33},loc="upper left", bbox_to_anchor=(-0.02, 1.03))

        fig_name = fig_names[k]
        k += 1
        plt.savefig(p+fig_name,format='pdf')





        plt.show()

