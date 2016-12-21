import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

if __name__ == '__main__':

    n_groups = 3
    TH = 2
    DS = 'Weibo'
    fig_p = 'C:\\Users\\rguo12\\Google Drive\\ASONAM_2016\\IEEEtran\\figures\\'

    #size
    '''
    means = [[0.65075281,0.527,0.582287765],[0.458641634,0.4455,0.451929128],[0.418249055,0.456,0.436283862],
             [0.477305247,0.3105,0.376093623],[0.280338835,0.40025,0.329705207],[0.269808599,0.4085,0.324948281],
             [0.1918181,0.364321608,0.251291933],[0.215707786,0.216,0.215797026]]
    stds = [[0.019749366,0.017669811,0.017149921],[0.006831933,0.00793062,0.005754246],[0.009761821,0.008096639,0.008380395],
            [0.023398431,0.026594172,0.026486749],[0.008781269,0.014461837,0.010592532],[0.005770578,0.012258784,0.00754864],
            [0.00415832,0.009838643,0.005327155],[0.022552399,0.029702226,0.026040789]]

    #time
    '''
    if DS == 'Twitter':
        if TH == 0:
            means = [[0.504366155,0.457587761,0.479839576],[0.502079662,0.972817921,0.662326793],
                     [0.504425421,0.527865046,0.515854442],
                     [0.518991835,0.199699495,0.288419807]]

            stds = [[0,0,0],[0,0,0],
                    [0,0,0],
                    [0,0,0]]

        if TH == 1:
            means = [[0.268456376,0.010949904,0.021041557],[0.333333333,0.00109499,0.00218281],
                     [0.201699272,0.001888858,0.003742572],
                     [0.280952381,0.064604435,0.105052304]
                     ]

            stds = [[0,0,0],[0,0,0],
                    [0,0,0],
                    [0,0,0]
                    ]

        if TH == 2:
            means = [[0.222222222,0.006830601,0.01325381],[0,0,0],
                     [0.041666667,0.000136612,0.000272294],
                     [0.144615385,0.032103825,0.05254332]
                     ]

            stds = [[0,0,0],[0,0,0],
                    [0,0,0],
                    [0,0,0]
                    ]

    if DS == 'Weibo':
        if TH == 0:
            means = [[0.619047619,0.609113241,0.614040251],[0.502416199,0.990194104,0.666603409],
                     [0.563543196,0.378664626,0.452965617],
                     [0.620370946,0.602770619,0.611444153]]

            stds = [[0,0,0],[0,0,0],
                    [0,0,0],
                    [0,0,0]]

        if TH == 1:
            means = [[0.487745758,0.145948342,0.22466892],[0.225806452,0.000282065,0.000563426],
                     [0.475,0.001531208,0.003052577],
                     [0.431813735,0.177861949,0.25194783]
                     ]

            stds = [[0,0,0],[0,0,0],
                    [0,0,0],
                    [0,0,0]
                    ]

        if TH == 2:
            means = [[0.318906606,0.042240772,0.074600355],[0,0,0],
                     [0,0,0],
                     [0.232055749,0.066981796,0.103956919]
                     ]

            stds = [[0,0,0],[0,0,0],
                    [0,0,0],
                    [0,0,0]
                    ]

    fig, ax = plt.subplots(figsize=(16,9))
    plt.subplots_adjust(left=None, bottom=0.08, right=None, top=0.98, wspace=None, hspace=None)

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 38}
    matplotlib.rc('font', **font)

    index = np.arange(n_groups)
    bar_width = 0.17

    opacity = 1.0
    error_config = {'ecolor': '0.0','capthick':1.5,'elinewidth':2.5}
    '''
    labels = ['Louvain Am','Infomap Am','SLM Am',
              'Louvain Bm','Infomap Bm','SLM Bm',
              'Cm','Dm']
    '''
    labels = ['Out-degree','Pagerank','Shell Number','Eigenvector']

    hatchs = ['-', '+', 'x', '\\', '.']

    rectss = []

    for i in xrange(0,len(labels)):

        rects = plt.bar(index+bar_width*i,means[i] , bar_width,
                     alpha=opacity,
                     color='white',
                     yerr=stds[i],
                     error_kw=error_config,
                     label=labels[i],hatch=hatchs[i] )
        #rects.set_hatch(hatchs[i])
    #plt.set_hatch(hatchs)

    '''
    rects1 = plt.bar(index, means_men, bar_width,
                     alpha=opacity,
                     color='b',
                     yerr=std_men,
                     error_kw=error_config,
                     label='Louvain')

    rects2 = plt.bar(index + bar_width, means_women, bar_width,
                     alpha=opacity,
                     color='r',
                     yerr=std_women,
                     error_kw=error_config,
                     label='Infomap')

    rects3 = plt.bar(index + 2*bar_width, means_women, bar_width,
                     alpha=opacity,
                     color='r',
                     yerr=std_women,
                     error_kw=error_config,
                     label='SLM')
    '''


    #should have 6+1+1 for each community detection we have 2 method(mine and Weng's) and the baseline and
    #the centrality measures

    #plt.xlabel('Metric')
    #plt.ylabel('Percentage')
    #plt.title(r'Classification Results: $TH=500$')
    plt.xticks(index + 2.5*bar_width, ('Precision', 'Recall', 'F1 Score'))  #change into p,r,f1

    #font_P = FontProperties()
    #font_P.set_size('small')
    plt.legend(ncol=1,prop={'size':35},loc='upper right', bbox_to_anchor=(1.02, 1.03))



    plt.ylim(0,1.1)

    #plt.tight_layout()

    #plt.tight_layout()
    #fig_name = fig_names[k]
    #k += 1
    if DS == 'Weibo':
        fig_name = 'cent_clf_'+str(TH)+'_weibo.eps'
    if DS == 'Twitter':
        fig_name = 'cent_clf_'+str(TH)+'.eps'
    plt.savefig(fig_p+fig_name)
    plt.show()