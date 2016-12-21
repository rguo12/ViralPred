#import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

if __name__ == '__main__':

    n_groups = 1

    DS = 'Twitter'#"Weibo"#
    problem = 'mixed'

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
    '''
    means = [[0.685645871,0.51627907,0.58892894],[0.486995475,0.498837209,0.492832474],[0.405792822,0.545930233,0.465511515],
             [0.432356828,0.308139535,0.359765332],[0.314267427,0.436046512,0.365249267],[0.27516143,0.340697674,0.304431074],
             [0.267669274,0.329761905,0.295454705],[0.42577568,0.193604651,0.266116522]]

    stds = [[0.013479539,0.019145874,0.016328889],[0.008244288,0.006600537,0.007009328],[0.005445222,0.012087183,0.007131946],
            [0.022835403,0.022096423,0.022673378],[0.00836109,0.01025485,0.008676836],[0.005941341,0.005616813,0.005579282],
            [0.010250581,0.008035102,0.009078806],[0.010253064,0.00868857,0.009696511]]
    '''

    if DS == 'Twitter':
        if problem == 'mixed': #clf for feature based and reg only for pp based
            means = [[98.6621],[24.8284],[88.8785],
                     [27.1416],[0.6371],[6.8543],
                     [0.3939],[0.0944],[0.3487],
                     [33.2695],[98.786]]

            stds = [[1.8981],[0.5455],[1.7283],
                    [0.2703],[0.0252],[0.1779],
                    [0.0366],[0.0134],[0.01824],
                    [0.6433],[10.470894751909208]]

    if DS == 'Weibo':
        if problem == 'mixed':
            means = [[1236.8485],[161.4144],[18831.78],
                     [302.966],[3.3968],[7970.436],
                     [0.49070003],[0.909200001],[97.8261],
                     [245.1013],[76711.9428]]
            stds = [[57.9265205],[8.706197577],[572.3123],
                    [5.007447],[0.142156],[193.6028],
                    [0.041226297],[0.032043094],[3.27982],
                    [5.4759],[7567.4208]]
        if problem == 'reg':
            means = [[],[],[18315.25463],
                     [],[],[6532.493000030518],
                     [],[],[6.7149],
                     [245.1013],[]]

            stds = [[],[],[532.0187338],
                    [],[],[],
                    [],[],[],
                    [5.4759],[]]

        if problem == 'clf':

            means = [[1236.8485],[],[],
                     [],[],[],
                     [],[],[],
                     [],[]]

            stds = [[57.9265205],[],[],
                    [],[],[],
                    [],[],[],
                    [],[]]

    fig, ax = plt.subplots(figsize=(16,9))
    plt.subplots_adjust(left=None, bottom=0.03, right=0.95, top=0.93, wspace=None, hspace=None)

    #fig.tight_layout()

    index = np.arange(n_groups)
    bar_width = 0.1

    opacity = 1.0
    error_config = {'ecolor': '0.0'}
    '''
    labels = ['Louvain Am','Infomap Am','SLM Am',
              'Louvain Bm','Infomap Bm','SLM Bm',
              'Cm','Dm']
    '''
    labels = ['A(RF)','A(LR)','A(SVM)',
              'B(RF)','B(LR)','B(SVM)',
              'C(DT)','C(LR)','C(SVM)',
              'SEISMIC','RPP']

    hatchs = ['-', '+', '//', 'x', '\\', '*', 'o', 'O', '.','/','|']

    rectss = []

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 38}

    matplotlib.rc('font', **font)

    colors = ['#FDFF33','#FDFF33','#FDFF33',
              '#2ECC71','#2ECC71','#2ECC71',
              'r','r','r',
              '#A6ACAF','#34495E']

    for i in xrange(0,len(labels)):

        rects = plt.bar(index+bar_width*i+0.05,means[i] , bar_width,
                     alpha=opacity,
                     color=colors[i],
                     yerr=stds[i],
                     error_kw=error_config,
                     label=labels[i],hatch=hatchs[i],
                        log=True)
        #rects.set_hatch(hatchs[i])
    #plt.set_hatch(hatchs)
    plt.ylim(0,10**3.3)

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
    plt.ylabel('Run time (seconds)')
    #plt.title(r'Classification Results: $TH=500$')
    plt.xticks(index*bar_width, [''])  #change into p,r,f1

    #font_P = FontProperties()
    #font_P.set_size('small')
    plt.legend(ncol=3, loc="upper left",prop={'size':30},bbox_to_anchor=(-0.01, 1.1))


    #plt.tight_layout()

    #plt.tight_layout()
    plt.show()