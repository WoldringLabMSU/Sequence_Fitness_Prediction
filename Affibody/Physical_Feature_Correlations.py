
import pandas as pd
import numpy as np
from Seq import Sequence
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



df =pd.read_csv('Aff_physical_rep.csv')

N=200

df1= df[df['label']==0].sample(n=N)
df2= df[df['label']==1].sample(n=N)


df_all= pd.concat([df1,df2])
seq=Sequence()
seq.df= df_all
seq.encode_one_hot()

#Cleaning the obtained data
seq.get_local_features_from_modlamp('Eisenberg', 'enriched')
seq.get_global_features_from_modlamp('enriched')
df_glob_enriched=pd.read_csv('enrichedgoldesc.csv')
df_loc_enriched=pd.read_csv('enrichedpepdesc.csv')
df_glob_enriched = df_glob_enriched.replace("b", '', regex=True)
df_loc_enriched = df_loc_enriched.replace("b", '', regex=True)
df_glob_enriched=df_glob_enriched.iloc[:,1:]
df_loc_enriched=df_loc_enriched.iloc[:,1:]
df_glob_enriched = df_glob_enriched.replace("'", "", regex=True)
df_loc_enriched = df_loc_enriched.replace("'", "", regex=True)

enriched_features= pd.concat([df_glob_enriched, df_loc_enriched], axis=1)
                             
scaler = StandardScaler()
data_features=enriched_features
feature_scaled= scaler.fit_transform(data_features)

feature_scaled= pd.DataFrame(feature_scaled)


m=list(df_glob_enriched.columns) +list( df_loc_enriched.columns)
                             
#Finding correlations between features 
corr = pd.DataFrame(feature_scaled).corr()


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, 20, step=0.01))  # Set label locations.
plt.xticks(np.arange(20), m)  # Set text labels.
sns.heatmap(corr,cmap="Blues")


plt.savefig('Corr for Physical_Features.png', dpi=600)
