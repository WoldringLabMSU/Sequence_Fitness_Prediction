#Importing modules
import pandas as pd
import numpy as np
import os
import textdistance
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import tensorflow
from tensorflow.keras.optimizers import Adam
import sklearn
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import modlamp
import modlamp.descriptors
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor 
import optuna
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from sklearn import metrics
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV


class Sequence:
    
    def __init__(self):
        self.alph= ["A","C", "D", "E","F", "G", "H", "I", "K", "L", "M", "N", "P","Q","R", "S", "T", "V","W", "Y"]
        self.df=None
        self.aff_wt= 'VDNKFNKELGWATWEIFNLPNLNGVQVKAFIDSLRDDPSQSANLLAEAKKLNDAQAPK'
    def read_df(self,file_name, file_path):
        
        self.name = file_name
        self.path = file_path
        self.df = pd.read_csv(os.path.join(self.path, self.name),index_col=False)
        
    def is_valid(self): # This aims to recheck of the sequence just contains 20 natural AAs
        
       
        invalids=[]
        for item, ind in self.df['Sequence'].iterrows():

            if not any(AA in item for AA in self.alph):
                invalids.append(ind)
                self.invalids=invalids
        if not self.invalids: # If we have no invalids all are acceptable sequences
            print(' All dataframe sequences are valid')
        else:
            self.df = self.df.drop(self.invalids, axis=0)
            print(len(self.invalids) ,'sequences were removed from the dataset')
            
    def find_unique_seqs(self):  #Finding unique sequences & counts
        self.df['count'] = 1
        self.df_unique = self.df.groupby("Sequence").agg({'count':'sum'}).reset_index()
        if self.df.equals(self.df_unique):
            print('No repetitions')
      
    def remove_freq_below_limit(self, freq_limit): # Removing low-count frequencies especially for bead sort data
        self.limit = freq_limit
        above_limit = self.df_unique['count'] >self.limit
        self.df_filtered = self.df_unique[above_limit]
        
        
    def find_single_mutants(self):
        single_list=[]
        for index, row in self.df.iloc[:].iterrows():
            if textdistance.hamming(row['Sequence'], self.aff_wt)==1:
                single_list.append(row['Sequence'], row['label'])
        self.single_mutant_list= single_list
        
    def find_double_mutants(self):
        single_list=[]
        for index, row in self.df.iloc[:].iterrows():
            if textdistance.hamming(row['Sequence'], self.aff_wt)==2:
                single_list.append(row['Sequence'], row['label'])
        self.single_mutant_list= single_list    
        
        
        
    # Defining 3 rep functions that I am using to import their data to my class  
        
    def encode_one_hot(self): #Onehot encoding the sequences, spaerse representations
         
        one_hot_length = len(self.alph)
        one_hots=[]

        for i in range(one_hot_length):
            a = np.zeros(one_hot_length)
            a[i] = 1
            one_hots.append(a)
   
        one_hot_dict = dict(zip(self.alph,one_hots))


        seq_one_hot_list =[]
       
        for seq in list(self.df['Sequence'].str.replace("\'", "")):
            seq_one_hot =[]
            for item in seq:
                seq_one_hot.append(one_hot_dict[item])
            seq_one_hot = np.concatenate(seq_one_hot).ravel()
            seq_one_hot_list.append(seq_one_hot)
            
        
        self.df['onehot'] = seq_one_hot_list
        
    def read_esm_1b(self, embed_df): # Embedding from ESM paper
            
        self.df['esm'] =self.esm
            
    def read_unirep(self, unirep_df): # Embedding from UniRep paper
        self.df['unirep']= unirep_df
        
        
    
         
    def get_local_features_from_modlamp(self, feature_type,name): #Adapted from modlamp python package
        pepdesc = PeptideDescriptor(list(self.df.Sequence), feature_type) 
    
        pepdesc.calculate_global()  # calculate global  hydrophobicity
        pepdesc.calculate_moment(append=True)  # calculate Eisenberg hydrophobic moment

        # load other AA scales
        pepdesc.load_scale('gravy')  # load GRAVY scale
        pepdesc.calculate_global(append=True)  # calculate global GRAVY hydrophobicity
        pepdesc.calculate_moment(append=True)  # calculate GRAVY hydrophobic moment
        pepdesc.load_scale('z3')  # load old Z scale
        pepdesc.calculate_autocorr(1, append=True)  # calculate global Z scale (=window1 autocorrelation)
        pepdesc.load_scale('levitt_alpha')
        pepdesc.calculate_global(append=True)
        pepdesc.load_scale('MSS') 
        pepdesc.calculate_global(append=True)
        pepdesc.load_scale('MSW')
        pepdesc.calculate_global(append=True)
        pepdesc.load_scale('refractivity')
        pepdesc.calculate_global(append=True)
        pepdesc.load_scale('flexibility')  
        pepdesc.calculate_global(append=True)
        pepdesc.load_scale('bulkiness')
        pepdesc.calculate_global(append=True)
        
        
        col_names1 = 'Sequence,H_Eisenberg,uH_Eisenberg,H_GRAVY,uH_GRAVY,Z3_1,Z3_2,Z3_3,levitt_alpha,MSS, MSW,refractivity, flexibility, bulkiness'   
        pepdesc.save_descriptor(name+'pepdesc.csv',header=col_names1)
           
        
    
    def get_global_features_from_modlamp(self,name): 
        globdesc = GlobalDescriptor(list(self.df.Sequence))
        globdesc.length()  # sequence length
        globdesc.boman_index(append=True)  # Boman index
        globdesc.aromaticity(append=True)  # global aromaticity
        globdesc.aliphatic_index(append=True)  # aliphatic index
        globdesc.instability_index(append=True)  # instability index
        globdesc.calculate_charge(ph=7.4, amide=False, append=True)  # net charge
        globdesc.calculate_MW(amide=False, append=True)  # molecular weight

        # save descriptor data to .csv file
        col_names2 = 'Sequence,Length,BomanIndex,Aromaticity,AliphaticIndex,InstabilityIndex,Charge,MW'
        globdesc.save_descriptor(name+'goldesc.csv', header=col_names2)
        
        
            
    def LR_oversampling(self, test_split_ratio,sampling_method, embedding_type, random_state):
    # Logistic regression (LR) 
        df_naive=self.df[self.df['label']==0]
        df_enriched=self.df[self.df['label']==1]
        split_dist = len(df_naive)/len(self.df)
        df_test_len= int(test_split_ratio*(len(self.df)))  #determine the number of test set instances
        df_naive_test= df_naive.sample(round(split_dist *df_test_len),random_state=random_state)
        df_enriched_test=df_enriched.sample(round((1-split_dist) *df_test_len), random_state=random_state)
        df_naive_train= df_naive.drop(df_naive_test.index, axis=0)
        df_enriched_train=df_enriched.drop(df_enriched_test.index, axis=0)
   
        test_set= pd.concat([df_naive_test, df_enriched_test]) 
        test_set=test_set.sample(frac=1, random_state=random_state)
        df_aside = pd.concat([df_naive_train, df_enriched_train])
        df_aside= df_aside.sample(frac=1, random_state=random_state)
        df_naive_aside=df_aside[df_aside['label']==0]
        df_enriched_aside=df_aside[df_aside['label']==1]
        df_naive_aside = df_naive_aside.sample(frac=1, random_state=random_state) 
        df_enriched_aside=df_enriched_aside.sample(frac=1,random_state=random_state) 
        model_scores=[]
        score_f1_mean_over_folds_=[]
        Prediction_on_test_set_= []     
        results = [] 
        results_ = []                             
        list_rep_train=[] 
        list_rep_test=[]
         # The embeddings were saved as strings and I am changing them back to numeric here
        for i in df_aside[embedding_type]:
        
            if embedding_type=='onehot' or embedding_type=='physical' :
                list_rep_train.append(i)
            else:
                list_rep_train.append(np.fromstring(i[1:-1], dtype=float, sep=','))
        X_train= list_rep_train
        X_train= pd.DataFrame(X_train)
        Y_train= np.array(df_aside['label'])

        for i in test_set[embedding_type]:
            if embedding_type=='onehot' or embedding_type=='physical' :
                list_rep_test.append(i)
            else:
                list_rep_test.append(np.fromstring(i[1:-1], dtype=float, sep=','))
        X_test= list_rep_test
        X_test= pd.DataFrame(X_test)
        Y_test= np.array(test_set['label'])
        
        ros = sampling_method 
        X_train_res, Y_train_res= ros.fit_resample(X_train, Y_train)
   
        clf = LogisticRegression(penalty='l2', max_iter=10000).fit(X_train_res, Y_train_res)
        prediction= clf.predict(X_test)
        prediction_prob= clf.predict_proba(X_test)
        Prediction_df_test = pd.DataFrame(prediction)
        Prediction_df_test = [0 if x<0.5  else 1 for x in Prediction_df_test.iloc[:,-1]]
        tn, fp, fn, tp= confusion_matrix(test_set['label'], Prediction_df_test).ravel()
        results.append([tn,fp,fn,tp])
        f1_score_= f1_score(Y_test, Prediction_df_test)
   
        return  pd.DataFrame(prediction_prob), list(Y_test), results, Prediction_df_test, f1_score_
            
    def LR_concat(self,test_split_ratio,  sampling_method , embedding_concat_list, random_state):
     
        df_naive=self.df[self.df['label']==0]
        df_enriched=self.df[self.df['label']==1]
        split_dist = len(df_naive)/len(self.df)
        df_test_len= int(test_split_ratio*(len(self.df)))  #determine the number of test set instances
        df_naive_test= df_naive.sample(round(split_dist *df_test_len),random_state=random_state)
        df_enriched_test=df_enriched.sample(round((1-split_dist) *df_test_len), random_state=random_state)
        df_naive_train= df_naive.drop(df_naive_test.index, axis=0)
        df_enriched_train=df_enriched.drop(df_enriched_test.index, axis=0)
        df_enriched_aside= df_enriched.drop(df_enriched_test.index)
        df_naive_aside= df_naive.drop(df_naive_test.index).sample(len(df_enriched_aside))
       

        test_set= pd.concat([df_naive_test,df_enriched_test]).sample(frac=1, random_state=random_state)
        train_set= pd.concat([df_naive_aside,df_enriched_aside]).sample(frac=1, random_state=random_state)
        print(len(df_naive_test))
        print(len(df_enriched_test))
        model_scores=[]
        results=[]               
        list_rep_train=[]
        list_rep_test=[]
        
        
        if len(embedding_concat_list)<3:
         # The embeddings were saved as strings and I am changing them back to numeric here
            for (i, j) in zip(train_set[embedding_concat_list[0]],train_set[embedding_concat_list[1]]):
                if embedding_concat_list[-1] !='onehot':
                    ll=np.concatenate([np.fromstring(i[1:-1], dtype=float, sep=','),np.fromstring(j[1:-1], dtype=float, sep=',')],axis=0).reshape(1,len(np.fromstring(i[1:-1], dtype=float, sep=','))+len(np.fromstring(j[1:-1], dtype=float, sep=',')))             
                    list_rep_train.append(ll)
                else:
                    list_rep_train.append(np.concatenate([np.fromstring(i[1:-1], dtype=float, sep=','),j], axis=0).reshape(1,len(np.fromstring(i[1:-1], dtype=float, sep=','))+len(j)))
            X_train= np.array(list_rep_train) 
            X_train= X_train.reshape(X_train.shape[0], X_train.shape[-1])
            Y_train= np.array(train_set['label'])
                  
              #This is necessary when the data is stored as string
            for (i,j) in zip(test_set[embedding_concat_list[0]],test_set[embedding_concat_list[1]]):
                if embedding_concat_list[-1] !='onehot':
         
                    list_rep_test.append(np.concatenate([np.fromstring(i[1:-1], dtype=float, sep=','),np.fromstring(j[1:-1], dtype=float, sep=',')],axis=0).reshape(1,len(np.fromstring(i[1:-1], dtype=float, sep=','))+len(np.fromstring(j[1:-1], dtype=float, sep=','))))
                else:
                    list_rep_test.append(np.concatenate([np.fromstring(i[1:-1], dtype=float, sep=','),j], axis=0).reshape(1,len(np.fromstring(i[1:-1], dtype=float, sep=','))+len(j)))
            X_test= np.array(list_rep_test)
            X_test= X_test.reshape(X_test.shape[0], X_test.shape[-1])
            Y_test= np.array(test_set['label'])
        else: 
            for (i, j, k) in zip(train_set[embedding_concat_list[0]],train_set[embedding_concat_list[1]], train_set[embedding_concat_list[2]]):
          
                list_rep_train.append(np.concatenate([np.fromstring(i[1:-1], dtype=float, sep=','),np.fromstring(j[1:-1], dtype=float, sep=','),k], axis=0).reshape(1,len(np.fromstring(i[1:-1], dtype=float, sep=','))+len(np.fromstring(j[1:-1], dtype=float, sep=','))+len(k)))
            X_train= np.array(list_rep_train)
            X_train= X_train.reshape(X_train.shape[0], X_train.shape[-1])
    
            Y_train= np.array(train_set['label'])
            
            
            for (i, j, k) in zip(test_set[embedding_concat_list[0]],test_set[embedding_concat_list[1]], test_set[embedding_concat_list[2]]):
          
                list_rep_test.append(np.concatenate([np.fromstring(i[1:-1], dtype=float, sep=','),np.fromstring(j[1:-1], dtype=float, sep=','),k], axis=0).reshape(1,len(np.fromstring(i[1:-1], dtype=float, sep=','))+len(np.fromstring(j[1:-1], dtype=float, sep=','))+len(k)))
    
                
            X_test= np.array(list_rep_test)
            X_test= X_test.reshape(X_test.shape[0], X_test.shape[-1])
            Y_test= np.array(test_set['label'])
     
        ros = sampling_method 
        X_train_res, Y_train_res= ros.fit_resample(X_train, Y_train)
        logistic_model = LogisticRegression(penalty='l2',max_iter=10000)
        logistic_model.fit(X_train_res, Y_train_res)
        Y_pred=logistic_model.predict(X_test)
        Y_pred_prob=logistic_model.predict(X_test)
        f1_score_= f1_score(Y_test,Y_pred)*100
        confusion_mat = confusion_matrix(Y_test,Y_pred)
        Y_pred = logistic_model.predict(X_test)
        Y_pred_prob=logistic_model.predict_proba(X_test)
        return f1_score_, confusion_mat, list(Y_pred), list(Y_pred_prob), list(Y_test)  
        
    #Different Splittings
    def controlled_train_test_split_oversampling_optuna(self,test_split_ratio, random_state): #Will use this for oversampling
        #split ratio is between train and test
        #Separating the test set first
        # distribution of data holds true for prediction set for more realstic model performance measurements
        df_naive=self.df[self.df['label']==0]
        df_enriched=self.df[self.df['label']==1]
        split_dist = len(df_naive)/len(self.df)
        df_test_len= int(test_split_ratio*(len(self.df)))  #determine the number of test set instances
        df_naive_test= df_naive.sample(round(split_dist *df_test_len),random_state=random_state)
        df_enriched_test=df_enriched.sample(round((1-split_dist) *df_test_len), random_state=random_state)
        df_naive_train= df_naive.drop(df_naive_test.index, axis=0)
        df_enriched_train=df_enriched.drop(df_enriched_test.index, axis=0)
        self.df_naive_test= df_naive_test
        self.df_naive_train =df_naive_train
        self.df_enriched_test =df_enriched_test
        self.df_enriched_train= df_enriched_train
   
  
   
    def LR_undersampling(self, embedding_type, random_state, N):
        
        self.df= self.df.sample(frac=1,random_state=random_state)
        df_naive=self.df[self.df['label']==0]
        df_enriched=self.df[self.df['label']==1]
        df_enriched_test= df_enriched.sample(n=N)  
        df_naive_test= df_naive.sample(n= round(len(df_naive)/len(df_enriched)*len(df_enriched_test)))
        df_enriched_aside= df_enriched.drop(df_enriched_test.index)
        df_naive_aside= df_naive.drop(df_naive_test.index).sample(len(df_enriched_aside))
      

        test_set= pd.concat([df_naive_test,df_enriched_test]).sample(frac=1, random_state=random_state)
        train_set= pd.concat([df_naive_aside,df_enriched_aside]).sample(frac=1, random_state=random_state)

        print(len(df_naive_test))
        print(len(df_enriched_test))
        model_scores=[]
        results=[]       
        list_rep_train=[] 
        list_rep_test=[]
        for i in test_set[embedding_type]:
                
             if embedding_type=='onehot' or embedding_type=='physical':
                     list_rep_test.append(i)
             else:
                     list_rep_test.append(np.fromstring(i[1:-1], dtype=float, sep=',')) #for formatting the obtained representations 
        X_test= list_rep_test
        X_test= pd.DataFrame(X_test)
        Y_test= np.array(test_set['label'])
   
                  
        for i in train_set[embedding_type]:
            if embedding_type=='onehot' or embedding_type=='physical':
                list_rep_train.append(i)
            else:
                list_rep_train.append(np.fromstring(i[1:-1], dtype=float, sep=','))
        X_train= list_rep_train
        X_train= pd.DataFrame(X_train)
        Y_train= np.array(train_set['label'])

        logistic_model = LogisticRegression(penalty='l2',max_iter=10000)
        logistic_model.fit(X_train, Y_train)
        Y_pred=logistic_model.predict(X_test)
        Y_pred_prob=logistic_model.predict(X_test)
        f1_score_= f1_score(Y_test,Y_pred)*100
        confusion_mat = confusion_matrix(Y_test,Y_pred)
        Y_pred = logistic_model.predict(X_test)
        Y_pred_prob=logistic_model.predict_proba(X_test)
        return f1_score_, confusion_mat, list(Y_pred), list(Y_pred_prob), list(Y_test)            
            
    def  score_model_for_kfold_keras_undersampling_RF(self, embedding_type,fold_n,n_estimate,N,  random_state):
        cv= StratifiedKFold(n_splits=fold_n)
        df_naive=self.df[self.df['label']==0]
        df_enriched=self.df[self.df['label']==1]
        df_enriched_test= df_enriched.sample(n=N, random_state= random_state)  
        df_naive_test= df_naive.sample(n= round(len(df_naive)/len(df_enriched)*len(df_enriched_test)), random_state= random_state)
        df_enriched_aside= df_enriched.drop(df_enriched_test.index)
        df_naive_aside= df_naive.drop(df_naive_test.index).sample(len(df_enriched_aside), random_state=random_state)
      

        test_set= pd.concat([df_naive_test,df_enriched_test]).sample(frac=1, random_state= random_state)
        model_scores=[]
        results=[] 
        for  (naive_train_fold_id, naive_val_fold_id),(enriched_train_fold_id, enriched_val_fold_id) in zip(cv.split(df_naive_aside,df_naive_aside['label']), cv.split(df_enriched_aside, df_enriched_aside['label'])):
              
             naive_train_fold,naive_val_fold = df_naive_aside.iloc[naive_train_fold_id,:], df_naive_aside.iloc[naive_val_fold_id,:]
             enriched_train_fold, enriched_val_fold = df_enriched_aside.iloc[enriched_train_fold_id,:], df_enriched_aside.iloc[enriched_val_fold_id,:]
             
             df_train_fold= pd.concat([naive_train_fold,enriched_train_fold]).sample(frac=1,random_state=random_state)
             df_val_fold= pd.concat([naive_val_fold,enriched_val_fold]).sample(frac=1,random_state=random_state)
             list_rep_train=[] 
             list_rep_val=[]
             list_rep_test=[]
             for i in df_train_fold[embedding_type]:
                
                 if embedding_type=='onehot' or embedding_type=='physical':
                     list_rep_train.append(i)
                 else:
                     list_rep_train.append(np.fromstring(i[1:-1], dtype=float, sep=','))
             X_train= list_rep_train
             X_train= pd.DataFrame(X_train)
             Y_train= np.array(df_train_fold['label'])
             for i in df_val_fold[embedding_type]:
                 if embedding_type=='onehot' or embedding_type=='physical':
                     list_rep_val.append(i)
                 else:
                     list_rep_val.append(np.fromstring(i[1:-1], dtype=float, sep=','))
                  #This is necessary when the data is stored as string
             X_val= list_rep_val
             X_val= pd.DataFrame(X_val)
             Y_val= np.array(df_val_fold['label'] ) 
                  
             for i in test_set[embedding_type]:
                 if embedding_type=='onehot' or embedding_type=='physical':
                     list_rep_test.append(i)
                 else:
                     list_rep_test.append(np.fromstring(i[1:-1], dtype=float, sep=','))
             X_test= list_rep_test
             X_test= pd.DataFrame(X_test)
             Y_test= np.array(test_set['label'])
      
        
           
                
             model=RandomForestClassifier(bootstrap=True, n_estimators = n_estimate, random_state = 0)  
             model=model.fit(X_train.values, y_train) 
             Prediction = model.predict(X_val.values, verbose = 2)
             Prediction_df = pd.DataFrame(Prediction)
             Prediction_df = [0 if x<0.5  else 1 for x in Prediction_df.iloc[:,0]]
             score_r = sklearn.metrics.recall_score(Y_val, Prediction_df)
             score_p=sklearn.metrics.precision_score(Y_val, Prediction_df)
             score_f1= f1_score(Y_val, Prediction_df)
             model_scores.append([score_r, score_p, score_f1])          
        Prediction_test = model.predict(X_test.values, verbose = 2)
        Prediction_df_test = pd.DataFrame(Prediction_test)
        Prediction_df_test = [0 if x<0.5  else 1 for x in Prediction_df_test.iloc[:,-1]]
        tn, fp, fn, tp= confusion_matrix(Y_test, Prediction_df_test).ravel()
        results.append([tn,fp,fn,tp])
        model_scores=pd.DataFrame(model_scores) 
        score_f1_mean_over_folds= np.mean(model_scores.iloc[:,-1])
        score_f1_test= f1_score(Y_test, Prediction_df_test)
        Prediction_on_test_set_ =list( Prediction_df_test)
        
        return Y_test,  results, Prediction_on_test_set_, score_f1_mean_over_folds, score_f1_test 
       


    def score_concat_reps_over_sampling_optuna(self,trial,  metrics, lr_list, decay_list, batch_size_list, epochs_list,fold_n,random_state,embedding_concat_list,sampling_method):  #if include onehot, bring at the end of the concat list
        test_set= pd.concat([self.df_naive_test, self.df_enriched_test]) 
        test_set=test_set.sample(frac=1, random_state=random_state)
        df_aside = pd.concat([self.df_naive_train, self.df_enriched_train])
        df_aside= df_aside.sample(frac=1, random_state=random_state)
        df_naive_aside=df_aside[df_aside['label']==0]
        df_enriched_aside=df_aside[df_aside['label']==1]
        df_naive_aside = df_naive_aside.sample(frac=1, random_state=random_state) 
        df_enriched_aside=df_enriched_aside.sample(frac=1,random_state=random_state) 
        model_scores=[]
        score_f1_mean_over_folds_=[]
        Prediction_on_test_set_= []     
        cv= KFold(n_splits=fold_n)
        results = [] 
        results_ = []
         
        for  (naive_train_fold_id, naive_val_fold_id),(enriched_train_fold_id, enriched_val_fold_id) in zip(cv.split(df_naive_aside), cv.split(df_enriched_aside)):
              
            naive_train_fold,naive_val_fold = df_naive_aside.iloc[naive_train_fold_id,:], df_naive_aside.iloc[naive_val_fold_id,:]
            enriched_train_fold, enriched_val_fold = df_enriched_aside.iloc[enriched_train_fold_id,:], df_enriched_aside.iloc[enriched_val_fold_id,:]
             
            df_train_fold= pd.concat([naive_train_fold,enriched_train_fold]).sample(frac=1,random_state=random_state)
            df_val_fold= pd.concat([naive_val_fold,enriched_val_fold]).sample(frac=1,random_state=random_state)
                                                                                                      
            list_rep_train=[] 
            list_rep_val=[]
            list_rep_test=[]
            if len(embedding_concat_list)<3:
             # The embeddings were saved as strings and I am changing them back to numeric here
                for (i, j) in zip(df_train_fold[embedding_concat_list[0]],df_train_fold[embedding_concat_list[1]]):
                    if embedding_concat_list[-1] !='onehot':
                        ll=np.concatenate([np.fromstring(i[1:-1], dtype=float, sep=','),np.fromstring(j[1:-1], dtype=float, sep=',')],axis=0).reshape(1,len(np.fromstring(i[1:-1], dtype=float, sep=','))+len(np.fromstring(j[1:-1], dtype=float, sep=',')))             
                        list_rep_train.append(ll)
                    else:
                        list_rep_train.append(np.concatenate([np.fromstring(i[1:-1], dtype=float, sep=','),j], axis=0).reshape(1,len(np.fromstring(i[1:-1], dtype=float, sep=','))+len(j)))
                X_train= np.array(list_rep_train) 
                X_train= X_train.reshape(X_train.shape[0], X_train.shape[-1])
                Y_train= np.array(df_train_fold['label'])
                      
                for (i,j) in zip(df_val_fold[embedding_concat_list[0]],df_val_fold[embedding_concat_list[1]]):
                    if embedding_concat_list[-1] !='onehot':
           
                        list_rep_val.append(np.concatenate([np.fromstring(i[1:-1], dtype=float, sep=','),np.fromstring(j[1:-1], dtype=float, sep=',')],axis=0).reshape(1,len(np.fromstring(i[1:-1], dtype=float, sep=','))+len(np.fromstring(j[1:-1], dtype=float, sep=','))))
                    else:
                        list_rep_val.append(np.concatenate([np.fromstring(i[1:-1], dtype=float, sep=','),j], axis=0).reshape(1,len(np.fromstring(i[1:-1], dtype=float, sep=','))+len(j)))
                X_val_fold= np.array(list_rep_val)
                X_val_fold= X_val_fold.reshape(X_val_fold.shape[0], X_val_fold.shape[-1])

                Y_val_fold= np.array(df_val_fold['label'])
                  #This is necessary when the data is stored as string
                for (i,j) in zip(test_set[embedding_concat_list[0]],test_set[embedding_concat_list[1]]):
                    if embedding_concat_list[-1] !='onehot':
             
                        list_rep_test.append(np.concatenate([np.fromstring(i[1:-1], dtype=float, sep=','),np.fromstring(j[1:-1], dtype=float, sep=',')],axis=0).reshape(1,len(np.fromstring(i[1:-1], dtype=float, sep=','))+len(np.fromstring(j[1:-1], dtype=float, sep=','))))
                    else:
                        list_rep_test.append(np.concatenate([np.fromstring(i[1:-1], dtype=float, sep=','),j], axis=0).reshape(1,len(np.fromstring(i[1:-1], dtype=float, sep=','))+len(j)))
                X_test= np.array(list_rep_test)
                X_test= X_test.reshape(X_test.shape[0], X_test.shape[-1])
                Y_test= np.array(test_set['label'])
            else: 
                for (i, j, k) in zip(df_train_fold[embedding_concat_list[0]],df_train_fold[embedding_concat_list[1]], df_train_fold[embedding_concat_list[2]]):
              
                    list_rep_train.append(np.concatenate([np.fromstring(i[1:-1], dtype=float, sep=','),np.fromstring(j[1:-1], dtype=float, sep=','),k], axis=0).reshape(1,len(np.fromstring(i[1:-1], dtype=float, sep=','))+len(np.fromstring(j[1:-1], dtype=float, sep=','))+len(k)))
                X_train= np.array(list_rep_train)
                X_train= X_train.reshape(X_train.shape[0], X_train.shape[-1])

                Y_train= np.array(df_train_fold['label'])
                      
                for (i, j, k) in zip(df_val_fold[embedding_concat_list[0]],df_val_fold[embedding_concat_list[1]], df_val_fold[embedding_concat_list[2]]):
              
                    list_rep_val.append(np.concatenate([np.fromstring(i[1:-1], dtype=float, sep=','),np.fromstring(j[1:-1], dtype=float, sep=','),k], axis=0).reshape(1,len(np.fromstring(i[1:-1], dtype=float, sep=','))+len(np.fromstring(j[1:-1], dtype=float, sep=','))+len(k)))

                    
                X_val_fold= np.array(list_rep_val)
                X_val_fold= X_val_fold.reshape(X_val_fold.shape[0], X_val_fold.shape[-1])
                Y_val_fold= np.array(df_val_fold['label'])
                
                for (i, j, k) in zip(test_set[embedding_concat_list[0]],test_set[embedding_concat_list[1]], test_set[embedding_concat_list[2]]):
              
                    list_rep_test.append(np.concatenate([np.fromstring(i[1:-1], dtype=float, sep=','),np.fromstring(j[1:-1], dtype=float, sep=','),k], axis=0).reshape(1,len(np.fromstring(i[1:-1], dtype=float, sep=','))+len(np.fromstring(j[1:-1], dtype=float, sep=','))+len(k)))

                    
                X_test= np.array(list_rep_test)
                X_test= X_test.reshape(X_test.shape[0], X_test.shape[-1])
                Y_test= np.array(test_set['label'])
         
            ros = sampling_method 
            X_train_res, Y_train_res= ros.fit_resample(X_train, Y_train)
                 
  
            if trial==None:
                model=Sequential()
                model.add(Dense(1, activation='sigmoid'))
                model.compile(Adam(lr=lr_list, decay= decay_list), loss='binary_crossentropy',metrics=metrics)
         
                model.fit(X_train_res, Y_train_res,batch_size =batch_size_list, epochs=epochs_list,verbose = 2)
         
            else:   
   
             # Defining the parameters
                
                lr_ = trial.suggest_categorical("lr", lr_list)
                decay_ = trial.suggest_categorical("decay", decay_list)
                batch_size_ = trial.suggest_categorical("batch_size", batch_size_list)
                epochs_ = trial.suggest_categorical("epochs", epochs_list)
             
                model=Sequential()
                model.add(Dense(1, activation='sigmoid'))
                model.compile(Adam(lr=lr_ , decay= decay_), loss='binary_crossentropy',metrics=metrics)
          
                model.fit(X_train_res, Y_train_res,batch_size =batch_size_, epochs=epochs_,verbose = 2)
           

            Prediction = model.predict(X_val_fold, verbose = 2)
            Prediction_df = pd.DataFrame(Prediction)
            Prediction_df = [0 if x<0.5  else 1 for x in Prediction_df.iloc[:,0]]
            score_r = sklearn.metrics.recall_score(Y_val_fold, Prediction_df)
            score_p=sklearn.metrics.precision_score(Y_val_fold, Prediction_df)
            score_f1= f1_score(Y_val_fold, Prediction_df)
            model_scores.append([score_r, score_p, score_f1])         
        Prediction_test = model.predict(X_test, verbose = 2)
        Prediction_df_test = pd.DataFrame(Prediction_test)
        Prediction_df_test = [0 if x<0.5  else 1 for x in Prediction_df_test.iloc[:,-1]]
        tn, fp, fn, tp= confusion_matrix(test_set['label'], Prediction_df_test).ravel()
        results.append([tn,fp,fn,tp])
        model_scores=pd.DataFrame(model_scores) 
        score_f1_mean_over_folds= np.mean(model_scores.iloc[:,-1])
     
        Prediction_on_test_set_ =list( Prediction_df_test)
        score_f1_test= f1_score(Y_test, Prediction_df_test)
  
            
        if trial==None:
            return  Y_test, results, Prediction_on_test_set_, score_f1_mean_over_folds, score_f1_test 
        
        else: 
            self.Prediction_on_test_set_ = Prediction_on_test_set_
            self.mean_kfold_recall = model_scores.iloc[:,0].mean()
            self.mean_kfold_precision = model_scores.iloc[:,1].mean()
            self.score_f1_mean_over_folds_ = score_f1_mean_over_folds_
            self.results_= results_
            self.test_set=test_set
            return  score_f1_mean_over_folds   #return this for optuna decision
    
       
        
            

    
        

