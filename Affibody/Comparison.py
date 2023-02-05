
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
#%% Reading the obtained results for different representation and sampling methods
undersampling= pd.read_csv('Pred_Infos_undersampling.csv')
oversampling= pd.read_csv('Pred_Info_oversampling.csv')
physical= pd.read_csv('Pred_Info_oversampling_physical.csv')
ph_undersampling= pd.read_csv('Pred_Info_undersampling_physical.csv')

physical_over=pd.DataFrame()
physical_over['Score']=pd.to_numeric(physical.iloc[4,1::2])
physical_over['Embedding'] = 'Physical_Features'
physical_over['Sampling']= 'R-Over-sampling'


physical_smote= pd.DataFrame()
physical_smote['Score']=pd.to_numeric(physical.iloc[4,2::2])
physical_smote['Embedding'] = 'Physical_Features'
physical_smote['Sampling']= 'SMOTE'

physical_under=pd.DataFrame()
physical_under['Score']=pd.to_numeric(ph_undersampling.iloc[4,1:])
physical_under['Embedding'] = 'Physical_Features'
physical_under['Sampling']=  ' Undersampling'
#%% 
undersampling_oh= list(undersampling.iloc[4,1::3])
undersampling_unirep=list(undersampling.iloc[4,2::3])
undersampling_esm= list(undersampling.iloc[4,3::3])

oversampling_ro_oh=list(oversampling.iloc[4,1::6])
oversampling_ro_unirep=list(oversampling.iloc[4,2::6])
oversampling_ro_esm= list(oversampling.iloc[4,3::6])

oversampling_smo_oh=list(oversampling.iloc[4,4::6])
oversampling_smo_unirep=list(oversampling.iloc[4,5::6])
oversampling_smo_esm= list(oversampling.iloc[4,6::6])


#%% plot 1 making comparison between each embedding performance for undersampling vs oversampling

OH_under= pd.DataFrame()
OH_under['Score']= pd.to_numeric(undersampling.iloc[4,1::3])
OH_under['Embedding'] = 'OneHot'
OH_under['Sampling']= 'Under-sampling'
OH_under_n= OH_under[(np.abs(stats.zscore(OH_under['Score'])) < 3)|(np.abs(stats.zscore(OH_under['Score'])) > -3)]

OH_over= pd.DataFrame()
OH_over['Score']= pd.to_numeric(oversampling.iloc[4,1::6])
OH_over['Embedding'] = 'OneHot'
OH_over['Sampling']= 'R-Over-sampling'
OH_over_n= OH_over[(np.abs(stats.zscore(OH_over['Score'])) < 3)|(np.abs(stats.zscore(OH_over['Score'])) >-3)]


OH_smote= pd.DataFrame()
OH_smote['Score']= pd.to_numeric(oversampling.iloc[4,4::6])
OH_smote['Embedding'] = 'OneHot'
OH_smote['Sampling']= 'SMOTE'
OH_smote_n= OH_smote[(np.abs(stats.zscore(OH_smote['Score'])) < 3)|(np.abs(stats.zscore(OH_smote['Score'])) >-3)]




Unirep_under= pd.DataFrame()
Unirep_under['Score']= pd.to_numeric(undersampling.iloc[4,2::3])
Unirep_under['Embedding'] = 'UniRep'
Unirep_under['Sampling']= 'Under-sampling'
Unirep_under_n= Unirep_under[(np.abs(stats.zscore(Unirep_under['Score'])) < 3)|(np.abs(stats.zscore(Unirep_under['Score'])) > -3)]


Unirep_over= pd.DataFrame()
Unirep_over['Score']= pd.to_numeric(oversampling.iloc[4,2::6])
Unirep_over['Embedding'] = 'UniRep'
Unirep_over['Sampling']= 'R-Over-sampling'
Unirep_over_n= Unirep_over[(np.abs(stats.zscore(Unirep_over['Score'])) < 3)|(np.abs(stats.zscore(Unirep_over['Score']))>-3)]


Unirep_smote= pd.DataFrame()
Unirep_smote['Score']= pd.to_numeric(oversampling.iloc[4,5::6])
Unirep_smote['Embedding'] = 'UniRep'
Unirep_smote['Sampling']= 'SMOTE'
Unirep_smote_n= Unirep_smote[(np.abs(stats.zscore(Unirep_smote['Score'])) < 3)|(np.abs(stats.zscore(Unirep_smote['Score'])) >-3)]




ESM_under= pd.DataFrame()
ESM_under['Score']= pd.to_numeric(undersampling.iloc[4,3::3])
ESM_under['Embedding'] = 'ESM'
ESM_under['Sampling']= 'Under-sampling'
ESM_under_n= ESM_under[(np.abs(stats.zscore(ESM_under['Score'])) < 3)|(np.abs(stats.zscore(ESM_under['Score'])) >-3)]


ESM_over= pd.DataFrame()
ESM_over['Score']= pd.to_numeric(oversampling.iloc[4,3::6])
ESM_over['Embedding'] = 'ESM'
ESM_over['Sampling']= 'R-Over-sampling'
ESM_over=ESM_over.drop(index='RandomOverSampler()UniRep_Embedding6')
ESM_over_n= ESM_over[(np.abs(stats.zscore(ESM_over['Score'])) < 3)|(np.abs(stats.zscore(ESM_over['Score'])) > -3)]


ESM_smote= pd.DataFrame()
ESM_smote['Score']= pd.to_numeric(oversampling.iloc[4,6::6])
ESM_smote['Embedding'] = 'ESM'
ESM_smote['Sampling']= 'SMOTE'
ESM_smote_n= ESM_smote[(np.abs(stats.zscore(ESM_smote['Score'])) < 3)| (np.abs(stats.zscore(ESM_smote['Score'])) >-3)]


physical_over=pd.DataFrame()
physical_over['Score']=pd.to_numeric(physical.iloc[4,1::2])
physical_over['Embedding'] = 'Physical_Features'
physical_over['Sampling']= 'R-Over-sampling'


physical_smote= pd.DataFrame()
physical_smote['Score']=pd.to_numeric(physical.iloc[4,2::2])
physical_smote['Embedding'] = 'Physical_Features'
physical_smote['Sampling']= 'SMOTE'
#%%
df_all_n= pd.concat([OH_under_n, OH_over_n, OH_smote_n, Unirep_under_n, Unirep_over_n, Unirep_smote_n, ESM_under_n, ESM_over_n, ESM_smote_n])
df_all= pd.concat([OH_under, OH_over, OH_smote, Unirep_under, Unirep_over, Unirep_smote, ESM_under, ESM_over, ESM_smote, physical_smote, physical_over])
df_all_nn= df_all[(np.abs(stats.zscore(df_all['Score'])) < 3)| (np.abs(stats.zscore(df_all['Score'])) >-3)]

df_alll= pd.concat([ OH_over, Unirep_over, ESM_under,  physical_smote])

#%% Sampling vs sampling

OH_under_vs_over= stats.ttest_ind(OH_under['Score'], OH_over['Score'])
OH_under_vs_smote= stats.ttest_ind(OH_under['Score'], OH_smote['Score'])
OH_over_vs_smote= stats.ttest_ind(OH_over['Score'], OH_smote['Score'])

Unirep_under_vs_over= stats.ttest_ind(Unirep_under['Score'], Unirep_over['Score'])
Unirep_under_vs_smote= stats.ttest_ind(Unirep_under['Score'], Unirep_smote['Score'])
Unirep_over_vs_smote= stats.ttest_ind(Unirep_over['Score'], Unirep_smote['Score'])



ESM_under_vs_over= stats.ttest_ind(ESM_under['Score'], ESM_over['Score'])
ESM_under_vs_smote= stats.ttest_ind(ESM_under['Score'], ESM_smote['Score'])
ESM_over_vs_smote= stats.ttest_ind(ESM_over['Score'], ESM_smote['Score'])

#%% T test Representation vs representation given same sampling method

Under_OH_Unirep= stats.ttest_ind(OH_under['Score'], Unirep_under['Score'])
Under_OH_ESM=stats.ttest_ind(OH_under['Score'], ESM_under['Score'])
Under_ESM_Unirep=stats.ttest_ind(ESM_under['Score'], Unirep_under['Score'])


Over_OH_Unirep= stats.ttest_ind(OH_over['Score'], Unirep_over['Score'])
Over_OH_ESM=stats.ttest_ind(OH_over['Score'], ESM_over['Score'])
Over_ESM_Unirep=stats.ttest_ind(ESM_over['Score'], Unirep_over['Score'])


smote_OH_Unirep= stats.ttest_ind(OH_smote['Score'], Unirep_smote['Score'])
smote_OH_ESM=stats.ttest_ind(OH_smote['Score'], ESM_smote['Score'])
smote_ESM_Unirep=stats.ttest_ind(ESM_smote['Score'], Unirep_smote['Score'])






