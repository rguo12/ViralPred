from sqlalchemy import create_engine
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/Aminer')
    conn = engine.connect()
    q = 'SELECT \"0\",\"1\",\"2\" FROM rt_df'
    res = conn.execute(q)

    rt_df = pd.DataFrame(res.fetchall())

    cascades = rt_df.groupby(0)

    #all_times_fr = []

    fiftythtimes = []

    for name,cascade in cascades:
        if len(cascade) >= 50:
            times = cascade[2].values
            times = times - times[0]
            #all_times_fr.extend(times)
            fiftythtimes.append(times[49])

    plt.hist(fiftythtimes,range=(0,5000000),bins=100,log=True,normed=True,cumulative=True)
    print np.percentile(fiftythtimes, 90)
    plt.show()