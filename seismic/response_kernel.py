import powerlaw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def pdf_plot(res):
    fig2 = res.plot_pdf(color = 'b', linewidth = 3,label='data')
    res.power_law.plot_pdf(color = 'b', linestyle = '--', ax = fig2,label='fit')

    plt.rc('font',size=45)
    #plt.title(r'PDF: cascade size $\alpha='+str(np.round(res.alpha,4))+'$')
    plt.xlabel('out-degree')
    plt.ylabel('probability')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    #df = pd.read_csv('E:\\ViralPred\\weng_network\\single_step_res_time_latest.txt',header=None)
    df = pd.read_csv('E:\\ViralPred\\weibo_network\\single_step_res_time_latest.txt',header=None)
    print df.head()
    res_time = df[0].values.astype('int').tolist()
    res_rt = powerlaw.Fit(res_time,xmin=300)
    print res_rt.pdf()

    print res_rt.xmin,res_rt.sigma,res_rt.alpha,res_rt.xmax
    pdf_plot(res_rt)

