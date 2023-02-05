import pandas as pd
from Seq import Sequence 
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler 

#%%
naive = Sequence()
enriched = Sequence()
pop_all = Sequence()

path= r"/mnt/home/Affibody/LR_sklearn"  
filename1= 'Naive.csv'
filename2= 'Enriched.csv'

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

#%%Finding the best params, random_oversampling

# Store the best variables in params for each representation
embed_list= ['onehot', 'UniRep_Embedding', 'ESM_Embed2']
sampling_list=[RandomOverSampler(), SMOTE()]

rand_list=np.arange(0,20)

preds=pd.DataFrame()
for i in rand_list:

   
    for j in embed_list:
    
        for s in sampling_list:
            preds[str(k) +j + str(i)]=pop_all.LR_oversampling(0.3,s,j,i) 
            
preds.to_csv('LR_oversampling-20rands.csv')   

