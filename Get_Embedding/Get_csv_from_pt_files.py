
# This file is changing the .pt files geerated per sequence to a single csv file


import os
import numpy as np
import torch
import pandas as pd
import zipfile
import io
from collections import defaultdict

directory = './ESM_rep'
zip_file= './ESM_rep/ESM_reps.zip'   # all generated .pt files per sequence in fasta are zipped

files =[]
prefixes = []

my_dict = defaultdict(list)



#Read in zip file
with  zipfile.ZipFile(zip_file,'r') as my_file:
	list_files = my_file.namelist()

#Iterate through files in zip file

       
	for zipfilename in list_files:

    #Read contents of the file
        	with my_file.open(zipfilename) as h:



                	p_file = torch.load(h)
                	dict1 = p_file['mean_representations']
                	dict2 = dict1[33]
                	arr= dict2.numpy()
                	my_dict[zipfilename].append(list(arr))

	os.chdir(directory)
	dataframe = pd.DataFrame(my_dict)
	dataframe.to_csv('Embed_ESM.csv')

