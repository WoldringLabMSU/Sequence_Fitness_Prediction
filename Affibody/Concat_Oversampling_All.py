

import pandas as pd
from Seq import Sequence 
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler 
import optuna
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
#%%
naive = Sequence()
enriched = Sequence()
pop_all = Sequence()
path= r"/mnt/home/Affibody/LR_sklearn"  

filename1 = 'Naive.csv'
filename2 = "Enriched.csv"
naive.read_df(filename1, path) 
enriched.read_df(filename2, path)
#%%
naive.df['label']=0
enriched.df['label']=1
df_final= pd.concat([naive.df.iloc[:, 1:], enriched.df.iloc[:, 1:]])
pop_all.df= df_final.reset_index()
#%% OneHot representations
enriched.encode_one_hot()
naive.encode_one_hot()
pop_all.encode_one_hot()

test_split_ratio=0.3

embedding_concat_list= ['UniRep_Embedding', 'ESM_Embed2', 'onehot',  ['UniRep_Embedding', 'ESM_Embed2', 'onehot']]
params=[]  
sampling_list=[RandomOverSampler(), SMOTE()]
rand_list=np.arange(0,20)
info_df=[]
preds=pd.DataFrame()
#%%
for m in rand_list:
  
    for s in sampling_list:
       
        preds[str(m)+ str(h)]=pop_all.LR_concat(test_split_ratio, s , embedding_concat_list, fold_n, m)

preds.to_csv('LR_concat_all.csv')
