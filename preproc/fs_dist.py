from sqlalchemy import create_engine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    engine = create_engine('postgresql://postgres:Impossible2@localhost:5432/Twitter_Indiana')
    conn = engine.connect()
    q = 'SELECT * FROM fs_df_50 WHERE \"fs\" >= 50;'
    res = conn.execute(q)

    fs_df = pd.DataFrame(res.fetchall())
    print fs_df.head()
    #fs_df = fs_df[fs_df[2]>=50]
    fs_list = fs_df[3].values.tolist()
    print np.percentile(fs_list,50)
    #plt.hist(fs_df[2].values,bins=500,log=True)
    #plt.show()
