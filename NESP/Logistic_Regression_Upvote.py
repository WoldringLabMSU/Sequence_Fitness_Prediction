

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
import statistics
#%%
df= pd.read_csv('LR-500.csv').iloc[:,1:]
df1= pd.read_csv('Concat_ESM_OH.csv').iloc[:,1:]
df2= pd.read_csv('Concat_UniRep_OH.csv').iloc[:,1:]
df3= pd.read_csv('Concat_All.csv').iloc[:,1:]
df4= pd.read_csv('ESM_UniRep.csv').iloc[:,1:]

#%%
#size 20

df_20=pd.DataFrame()
df_20['label']= list(df.iloc[4,16::24])
df_20['oh']=list(df.iloc[2,0::24])
df_20['oh_soft']=list(df.iloc[3,0::24])

df_20['unirep']=list(df.iloc[2,8::24])
df_20['unirep_soft']=list(df.iloc[3,8::24])

df_20['esm']=list(df.iloc[2,16::24])
df_20['esm_soft']=list(df.iloc[3,16::24])

df_20['esm_oh']= list(df1.iloc[2,0::7])
df_20['unirep-oh']=  list(df2.iloc[2,0::7])
df_20['esm_unirep'] = list(df4.iloc[2,0::7])
df_20['All']=list(df3.iloc[2,0::7])

#Size40

df_40=pd.DataFrame()
df_40['label']= list(df.iloc[4,1::24])
df_40['oh']=list(df.iloc[2,1::24])
df_40['oh_soft']=list(df.iloc[3,1::24])

df_40['unirep']=list(df.iloc[2,9::24])
df_40['unirep_soft']=list(df.iloc[3,9::24])

df_40['esm']=list(df.iloc[2,17::24])
df_40['esm_soft']=list(df.iloc[3,17::24])

df_40['esm_oh']= list(df1.iloc[2,1::7])
df_40['unirep-oh']=  list(df2.iloc[2,1::7])
df_40['esm_unirep'] = list(df4.iloc[2,1::7])
df_40['All']=list(df3.iloc[2,1::7])


# size80
df_80=pd.DataFrame()
df_80['label']= list(df.iloc[4,2::24])
df_80['oh']=list(df.iloc[2,2::24]) 
df_80['oh_soft']=list(df.iloc[3,2::24]) 

df_80['unirep']=list(df.iloc[2,10::24])
df_80['unirep_soft']=list(df.iloc[3,10::24])

df_80['esm']=list(df.iloc[2,18::24])
df_80['esm_soft']=list(df.iloc[3,18::24])

df_80['esm_oh']= list(df1.iloc[2,2::7])
df_80['unirep-oh']=  list(df2.iloc[2,2::7])
df_80['esm_unirep'] = list(df4.iloc[2,2::7])
df_80['All']=list(df3.iloc[2,2::7])


# size 200

df_200=pd.DataFrame()
df_200['label']= list(df.iloc[4,4::24])
df_200['oh']=list(df.iloc[2,4::24])
df_200['oh_soft']=list(df.iloc[3,4::24])
 
df_200['unirep']=list(df.iloc[2,12::24])
df_200['unirep_soft']=list(df.iloc[3,12::24])
df_200['esm']=list(df.iloc[2,20::24])
df_200['esm_soft']=list(df.iloc[3,20::24])

df_200['esm_oh']= list(df1.iloc[2,4::7])
df_200['unirep-oh']=  list(df2.iloc[2,4::7])
df_200['esm_unirep'] = list(df4.iloc[2,4::7])
df_200['All']=list(df3.iloc[2,4::7])


    #size 400
df_400=pd.DataFrame()
df_400['label']= list(df.iloc[4,5::24])
df_400['oh']=list(df.iloc[2,5::24]) 
df_400['oh_soft']=list(df.iloc[3,5::24]) 

df_400['unirep']=list(df.iloc[2,13::24])
df_400['unirep_soft']=list(df.iloc[3,13::24])

df_400['esm']=list(df.iloc[2,21::24])
df_400['esm_soft']=list(df.iloc[3,21::24])

df_400['esm_oh']= list(df1.iloc[2,5::7])
df_400['unirep-oh']=  list(df2.iloc[2,5::7])
df_400['esm_unirep'] = list(df4.iloc[2,5::7])
df_400['All']=list(df3.iloc[2,5::7])

#size 1000

df_1000=pd.DataFrame()
df_1000['label']= list(df.iloc[4,6::24])
df_1000['oh']=list(df.iloc[2,6::24])
df_1000['oh_soft']=list(df.iloc[3,6::24])
 
df_1000['unirep']=list(df.iloc[2,14::24])
df_1000['unirep_soft']=list(df.iloc[3,14::24])

df_1000['esm']=list(df.iloc[2,22::24])
df_1000['esm_soft']=list(df.iloc[3,22::24])

df_1000['esm_oh']= list(df1.iloc[2,6::7])
df_1000['unirep-oh']=  list(df2.iloc[2,6::7])
df_1000['esm_unirep'] = list(df4.iloc[2,6::7])
df_1000['All']=list(df3.iloc[2,6::7])

#%%

def cal_median(list_):
    a=[]
    for i in list_:
        a.append(statistics.median(i))
    return a

def cal_mean(list_):
    b=[]
    for i in list_:
        b.append(statistics.mean(i))
    return b   
        
def calculate_f1(df, embed, exclude=None):
    score=[] 
    for i,j in zip( df[embed], df['label']):
        score.append(f1_score(i,j)*100)
    return score



def get_violin(rep_list, df):
    df_list=[]
    for i,j in zip(rep_list, np.arange(0, len(rep_list))):
    
        var= 'dff' +str(j)
        var= pd.DataFrame()
        var['Score']= calculate_f1(df,rep_list[j])
        var['Embedding']= rep_list[j]
            
        df_list.append(var)
    f = plt.figure()
    f.set_figwidth(8)
    f.set_figheight(6)
    df_final= pd.concat(df_list)
    sns.violinplot(x='Embedding', y='Score', data=df_final,palette='Set2', scale_hue=True, linewidth=1,width=0.5,orient='v', legend_out=False,cut=0)
    plt.ylim([-3,103])
    return df_final

def calculate_statistics(df_list, columns):
    statistics = {}
    for column in columns:
        data = [df[column] for df in df_list]
        median = cal_median(data)
        mean = cal_mean(data)
        statistics[f'median_{column}'] = median
        statistics[f'mean_{column}'] = mean
    return statistics

#%%


def calc_votes(df):
    mostCommonVote = []
    softVote = []

    for row in df[['oh', 'unirep', 'esm']].values:
        common_votes = []
        soft_votes = []

        for i,j,k in zip(row[0], row[1], row[2]):
            if i + j + k >= 2:
                common_votes.append(1)
            else:
                common_votes.append(0)
        
        mean = np.mean([i, j, k])
        if mean >= 0.5:
            soft_votes.append(1)
        else:
            soft_votes.append(0)
        
        mostCommonVote.append(common_votes)
        softVote.append(soft_votes)
    
    df['Upvoted'] = mostCommonVote
    df['Soft_Voted'] = softVote
    
    return df

df_list = [df_20, df_40, df_80, df_200 ,df_400, df_1000]
for df in df_list:
    df = calc_votes(df) 
    

#%%
a=get_violin(['oh', 'unirep', 'esm', 'unirep-oh','esm_oh', 'esm_unirep','All',  'Upvoted', 'Soft_Voted'], df_20)  
b=get_violin(['oh', 'unirep', 'esm','unirep-oh','esm_oh', 'esm_unirep', 'All' ,'Upvoted','Soft_Voted'], df_40) 
c= get_violin(['oh', 'unirep', 'esm','unirep-oh','esm_oh', 'esm_unirep', 'All' ,'Upvoted','Soft_Voted'], df_80) 
d=get_violin(['oh', 'unirep', 'esm', 'unirep-oh','esm_oh', 'esm_unirep','All' ,'Upvoted','Soft_Voted'], df_200) 
e= get_violin(['oh', 'unirep', 'esm','unirep-oh','esm_oh', 'esm_unirep', 'All' ,'Upvoted','Soft_Voted'], df_400) 
f= get_violin(['oh', 'unirep', 'esm','unirep-oh','esm_oh', 'esm_unirep', 'All' ,'Upvoted','Soft_Voted'], df_1000) 
#%%
df_20['oh']= a[a['Embedding']=='oh']
df_20['unirep']= a[a['Embedding']=='unirep']
df_20['esm']= a[a['Embedding']=='esm']
df_20['unirep-oh']= a[a['Embedding']=='unirep-oh']
df_20['esm_oh']= a[a['Embedding']=='esm_oh']
df_20['esm_unirep']= a[a['Embedding']=='esm_unirep']
df_20['All']= a[a['Embedding']=='All']
df_20['Upvoted1']= a[a['Embedding']=='Upvoted']['Score']
df_20['Upvoted2']= a[a['Embedding']=='Soft_Voted']['Score']

df_40['oh']= b[b['Embedding']=='oh']
df_40['unirep']= b[b['Embedding']=='unirep']
df_40['esm']= b[b['Embedding']=='esm']
df_40['unirep-oh']= b[b['Embedding']=='unirep-oh']
df_40['esm_oh']= b[b['Embedding']=='esm_oh']
df_40['esm_unirep']= b[b['Embedding']=='esm_unirep']
df_40['All']= b[b['Embedding']=='All']
df_40['Upvoted1']= b[b['Embedding']=='Upvoted']['Score']
df_40['Upvoted2']= b[b['Embedding']=='Soft_Voted']['Score']


df_80['oh']= c[c['Embedding']=='oh']
df_80['unirep']= c[c['Embedding']=='unirep']
df_80['esm']= c[c['Embedding']=='esm']
df_80['unirep-oh']= c[c['Embedding']=='unirep-oh']
df_80['esm_oh']= c[c['Embedding']=='esm_oh']
df_80['esm_unirep']= c[c['Embedding']=='esm_unirep']
df_80['All']= c[c['Embedding']=='All']
df_80['Upvoted1']= c[c['Embedding']=='Upvoted']['Score']
df_80['Upvoted2']= c[c['Embedding']=='Soft_Voted']['Score']

df_200['oh']= d[d['Embedding']=='oh']
df_200['unirep']= d[d['Embedding']=='unirep']
df_200['esm']= d[d['Embedding']=='esm']
df_200['unirep-oh']= d[d['Embedding']=='unirep-oh']
df_200['esm_oh']= d[d['Embedding']=='esm_oh']
df_200['esm_unirep']= d[d['Embedding']=='esm_unirep']
df_200['All']= d[d['Embedding']=='All']
df_200['Upvoted1']= d[d['Embedding']=='Upvoted']['Score']
df_200['Upvoted2']= d[d['Embedding']=='Soft_Voted']['Score']

df_400['oh']= e[e['Embedding']=='oh']
df_400['unirep']= e[e['Embedding']=='unirep']
df_400['esm']= e[e['Embedding']=='esm']
df_400['unirep-oh']= e[e['Embedding']=='unirep-oh']
df_400['esm_oh']= e[e['Embedding']=='esm_oh']
df_400['esm_unirep']= e[e['Embedding']=='esm_unirep']
df_400['All']= e[e['Embedding']=='All']
df_400['Upvoted1']= e[e['Embedding']=='Upvoted']['Score']
df_400['Upvoted2']= e[e['Embedding']=='Soft_Voted']['Score']

   
df_1000['oh']= f[f['Embedding']=='oh']
df_1000['unirep']= f[f['Embedding']=='unirep']
df_1000['esm']= f[f['Embedding']=='esm']
df_1000['unirep-oh']= f[f['Embedding']=='unirep-oh']
df_1000['esm_oh']= f[f['Embedding']=='esm_oh']
df_1000['esm_unirep']= f[f['Embedding']=='esm_unirep']
df_1000['All']= f[f['Embedding']=='All']
df_1000['Upvoted1']= f[f['Embedding']=='Upvoted' ]['Score']
df_1000['Upvoted2']= f[f['Embedding']=='Soft_Voted' ]['Score']

#%%


columns = ['oh', 'unirep', 'esm', 'All', 'Upvoted1', 'Upvoted2', 'esm_unirep', 'unirep-oh', 'esm_oh']

stats = calculate_statistics(df_list, columns)
median_oh = stats['median_oh']
mean_oh = stats['mean_oh']
median_unirep = stats['median_unirep']
mean_unirep = stats['mean_unirep']
median_esm = stats['median_esm']
mean_esm = stats['mean_esm']
median_all = stats['median_All']
mean_all = stats['mean_All']
median_upvote1 = stats['median_Upvoted1']
mean_upvote1 = stats['mean_Upvoted1']
median_upvote2 = stats['median_Upvoted2']
mean_upvote2 = stats['mean_Upvoted2']
median_esm_unirep = stats['median_esm_unirep']
mean_esm_unirep = stats['mean_esm_unirep']
median_unirep_oh = stats['median_unirep-oh']
mean_unirep_oh = stats['mean_unirep-oh']
median_esm_oh = stats['median_esm_oh']
mean_esm_oh = stats['mean_esm_oh']
#%%


plt.plot([range(1,7)], np.array(mean_oh).reshape(1,6), marker="d", color='olive', label='OH')
plt.plot([range(1,7)], np.array(mean_unirep).reshape(1,6), marker="v",color='green', label='unirep')

plt.plot([range(1,7)], np.array(mean_esm).reshape(1,6), marker="^", color='magenta', label='esm')

plt.plot([range(1,7)], np.array(mean_unirep_oh).reshape(1,6), marker="P", color='purple',label= 'unirep-oh')

plt.plot([range(1,7)], np.array(mean_esm_oh).reshape(1,6), marker="*", color='blue', label= 'esm-oh')

plt.plot([range(1,7)], np.array(mean_esm_unirep).reshape(1,6), marker='o', color='salmon', label='esm-unirep')
plt.plot([range(1,7)], np.array(mean_all).reshape(1,6), marker="<", color='orange', label='all')

plt.plot([range(1,7)], np.array(mean_upvote1).reshape(1,6), marker="x", color='gray', label='Upvote1')

plt.plot([range(1,7)], np.array(mean_upvote2).reshape(1,6), marker=">", color='darkcyan', label='Upvote2')
 


def legend_without_duplicate_labels(figure):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 0.80), loc='upper left', borderaxespad=0)
legend_without_duplicate_labels(plt)

plt.title('Mean F1-Score for Different Protein Representations')
