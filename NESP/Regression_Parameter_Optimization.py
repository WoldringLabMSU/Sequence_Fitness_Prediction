
import optuna
import pandas as pd
import numpy as np
from Seq import Sequence
from scipy import stats

df_final=pd.read_csv('kaggle_reg_df.csv')

m=[]
for i in df_final['Sequence']:
    m.append(len(i))
df_final['length']=m
df_final=df_final[df_final['length']<=500]
df_final= df_final[(np.abs(stats.zscore(df_final['tm'])) < 3)]
df_final= df_final.reset_index()

protein_length=500
rf_n_estimators_list = [10, 1000]
rf_max_depth_list = [2, 32]


dff=Sequence()
dff.df=df_final
dff.encode_one_hot_padded(max_length=500)

embed_list= ['onehot', 'UniRep_Embedding', 'ESM_Embed']
params=[]
fold_n=5
pred=pd.DataFrame()
rands= np.arange(1)
for i in rands:
    for item in embed_list:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())  # Minimizing MSE
        study.optimize(lambda trial:dff.score_model_RF_Reg_kaggle(trial, item, protein_length, fold_n,rf_max_depth_list, rf_n_estimators_list, i),  n_trials=5)
        params.append([item,  study.best_params])
params= pd.DataFrame(params)
params.to_csv('Optimized_Params.csv')
