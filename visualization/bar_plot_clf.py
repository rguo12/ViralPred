import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

if __name__ == '__main__':

    n_groups = 3
    TH = 2
    DS = 'Twitter'
    #fig_p = 'C:\\Users\\rguo12\\Google Drive\\ASONAM_2016\\IEEEtran\\figures\\'
    fig_p = 'C:\\Users\\rguo12\\Google Drive\\ASONAM_2016\\Presentation\\comparison_ppt\\'

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
            means = [[0.86631475,0.892487365,0.879205018],[0.815112069,0.8602537,0.837069795],
                     [0.518991835,0.199699495,0.288419807],
                     [0.576033835,0.57531758,0.575673873],[0.515185015,0.487725896,0.501013128]
                     ]

            stds = [[0.001410216,0.001478815,0.000900978],[0.002631753,0.003339485,0.002073042],
                    [0,0,0],
                    [0.001472587,0.002441292,0.001742604],[0.021963579,0.028094029,0.024723849],
                    ]

        if TH == 1:
            means = [[0.827504977,0.741828634,0.782325852],[0.776769276,0.658430233,0.712703748],
                     [0.280952381,0.064604435,0.105052304],
                     [0.411534083,0.247248837,0.308903431],[0.324686643,0.206150209,0.252129109]
                     ]

            stds = [[0.002351992,0.002338723,0.00189482],[0.006201721,0.007021016,0.005702481],
                    [0,0,0],
                    [0.001717822,0.002311326,0.002144185],[0.027660742,0.018614982,0.021939641],
                    ]

        if TH == 2:
            means = [[0.766672097,0.575887978,0.6577078],[0.749736345,0.41888412,0.537415611],
                     [0.144615385,0.032103825,0.05254332],
                     [0.355158831,0.12431694,0.184160657],[0.172129337,0.061965825,0.090884617]
                     ]

            stds = [[0.004129009,0.006264075,0.004765804],[0.012559037,0.009093132,0.00897367],
                    [0,0,0],
                    [0.005440012,0.002010875,0.002632788],[0.076049198,0.029484968,0.042211404],
                    ]

    if DS == 'Weibo':
        if TH == 0:
            means = [[0.766531588,0.711932184,0.738223448],[0.723544986,0.676316849,0.699133861],
                    [0.620370946,0.602770619,0.611444153],
            [0.499157868,0.480821118,0.489817466],[0.553858674,0.555109739,0.55447395]]
            stds = [[0.000538601,0.000666881,0.000400223],[0.000940611,0.000614825,0.000729363],
                    [0,0,0],
            [0.000959623,0.001254553,0.001004709],[0.007330582,0.010446763,0.008686567]]
        if TH == 1:
            means = [[0.696332073,0.397554096,0.506137705],[0.66811294,0.35675545,0.465137578],
                    [0.431813735,0.7861949,0.25194783],
            [0.247393314,0.013373897,0.025375482],[0.308202346,0.173979797,0.222396904]]
            stds = [[0.00148241,0.001775464,0.001577474],[0.001004065,0.000806168,0.000595086],
                    [0,0,0],
            [0.008866948,0.000525536,0.000985761],[0.017075521,0.009923057,0.012438693]]
        if TH == 2:
            means = [[0.658103407,0.181011767,0.283922856],[0.630597423,0.177029066,0.276446475],
                    [0.232055749,0.0669896,0.103956919],
            [0.247479737,0.000804586,0.001603912],[0.139684161,0.033793947,0.054378935]]

            stds = [[0.002560054,0.001749577,0.002073347],[0.00387973,0.001279035,0.001655773],
                    [0,0,0],
            [0.043314823,0.000177395,0.000353316],[0.014995603,0.004752802,0.007143278]]


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
    labels = ['A(RF)','B(RF)','C(DT)',
              'SEISMIC','RPP']

    hatchs = ['-', '+', 'x', '\\', '.']

    rectss = []

    colors = ['#FDFF33','#2ECC71','r','#A6ACAF','#34495E']
    for i in xrange(0,len(labels)):

        rects = plt.bar(index+bar_width*i,means[i] , bar_width,
                     alpha=opacity,
                     color=colors[i],
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
    plt.legend(ncol=3,prop={'size':35},loc='upper center', bbox_to_anchor=(0.5, 1.03))



    plt.ylim(0,1.12)

    #plt.tight_layout()

    #plt.tight_layout()
    #fig_name = fig_names[k]
    #k += 1
    if DS == 'Weibo':
        fig_name = 'clf_'+str(TH)+'_weibo.pdf'
    if DS == 'Twitter':
        fig_name = 'clf_'+str(TH)+'.pdf'
    plt.savefig(fig_p+fig_name)
    plt.show()