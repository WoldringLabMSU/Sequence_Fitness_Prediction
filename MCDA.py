import pandas as pd
import numpy as np
import topsispy as tp

# Reading Confusion matrix and Preprocessing
df= pd.read_csv('cm_vals_final.csv')
df['FPR'] = df['fp']/(df['fp'] + df['tn'])
df['TPR']=df['tp']/(df['tp'] + df['fn']) 
df['Precision']=df['tp']/(df['tp'] + df['fp']) 
df['NPV']= df['tn']/(df['tn'] + df['fn'])
df['FDR']=df['fp']/(df['fp'] + df['tp'])
df['F1']=  2 * (df['Precision']*df['TPR'])/(df['Precision']+df['TPR'])


# select the numeric columns and group them by every 20 rows ( mean of scores in each 20 random state calculations)
num_cols = df.select_dtypes(include=[float, int]).columns
grouped_num_cols = df[num_cols].groupby(df.index // 20)
mean_df =  grouped_num_cols.mean()
mean_df[['FPR', 'TPR', 'Precision', 'NPV', 'FDR', 'F1']] = mean_df[['FPR', 'FNR', 'TNR', 'TPR', 'Precision', 'NPV', 'FDR', 'F1']].apply(pd.to_numeric)
mean_df= np.array(mean_df)
w=[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]  #relative Importance of the metrics, can be changed within user preferences

# calculate the TOPSIS scores and rankings using the specified criteria
sign = [-1,1,1,1,-1,1] # 1 indicates maximizing and -1 indicates minimization
topsis_results = tp.topsis(a, w, sign)
