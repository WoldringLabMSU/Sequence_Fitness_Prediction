
import pandas as pd
import numpy as np
import jax_unirep
from jax_unirep import get_reps
from jax.random import PRNGKey
from jax_unirep.evotuning_models import mlstm64
from jax_unirep.utils import load_params

Data= pd.read_csv('data.csv')
 
Data=Data.drop_duplicates(subset='Sequence', keep='first')   #Removing duplicated sequences


Aff_dict = {}
seq = pd.DataFrame(Data['Sequence'])
seq = pd.DataFrame(seq)
seq = seq.applymap(str)
seq = seq.values.tolist() # the only important thing here was that the sequences should be feeded to model as a list
seq = [i[0] for i in seq]
seq = list (seq)
Embeds = get_reps(seq)[0] #mean rep
Embeds = pd.DataFrame(Embeds)
E=[]
for row in Embeds.iterrows():
    E.append(list(row[1]))
Data['Embedding']= E
Data.to_csv('data_UniRep_Embeded.csv')
