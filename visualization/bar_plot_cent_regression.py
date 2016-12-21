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
    #Twitter
    if DS == 'Twitter':
        error_matrix = [[2.772794422,1818.63851,1.376180374,0.091780822],
                        [2.735499275,1793.709215,1.378700274,0.096575342],
                        [2.751147451,1801.338236,1.373446704,0.104109589],
                        [2.77295974,1892.404473,1.389350014,0.088356164]]

        fig_names = ['cent_reg_ape.eps','cent_reg_rmse.eps','cent_reg_rmlse.eps','cent_top_cov.eps']
        y_lim = [3.8,2500,1.8,0.15]
        n_cols = [2,2,2,2]
        p = 'C:\\Users\\rguo12\\Google Drive\\ASONAM_2016\\IEEEtran\\figures\\'
    if DS == 'Weibo':

        error_matrix = [[1.700127193,778.1060607,1.094372596,0.156444444],
                        [1.895334682,745.6476577,1.158875464,0.103931452],
                        [1.89316148,746.321355,1.158414543,0.106149194],
                        [1.725625422,830.5671691,1.115390722,0.143145161]] #RPP result with 10% sample

        fig_names = ['cent_reg_ape_weibo.eps','cent_reg_rmse_weibo.eps','cent_reg_rmlse_weibo.eps','cent_top_cov_weibo.eps']
        y_lim = [2.5,1100,1.6,0.2]
        n_cols = [2,2,2,2]
        p = 'C:\\Users\\rguo12\\Google Drive\\ASONAM_2016\\IEEEtran\\figures\\'

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
        plt.subplots_adjust(left=0.2, bottom=None, right=0.85, top=0.95, wspace=None, hspace=None)
        if j == 0:
            formatter = FuncFormatter(to_percent)
            plt.gca().yaxis.set_major_formatter(formatter)
        j += 1

        index = np.arange(n_groups)
        bar_width = 0.4

        opacity = 1.0
        error_config = {'ecolor': '0.0'}

        labels = ['Out-degree','Pagerank','Shell Number','Eigenvector']

        hatchs = ['-', '+', 'x', '\\']

        rectss = []
        index = np.arange(n_groups)
        for i in xrange(0,4):

            plt.bar(index[i]*bar_width+1.75,error_vector[i] , bar_width,
                             alpha=opacity,
                             color='white',
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
        plt.savefig(p+fig_name,format='eps')





        plt.show()

