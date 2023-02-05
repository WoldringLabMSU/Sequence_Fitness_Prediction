This directory goes throug more guidance to obain sequence representations via ESM and UniRep. 

Note that one-hot and physiochemical representations can be directly obtained from seq.py

Both UniRep and ESM representations are obtained first via cloning their GitHub ( UniRep: https://github.com/ElArkk/jax-unirep), 
(ESM: https://github.com/facebookresearch/esm)


For ESM, there are various options to obtain the representations based on the project need. Tables and keywords for obtaining the reps are in ESM GitHub link. Here, 2 files (one .sb and one .py all available as a template). The sequences are changes to a fasta format and submitted as a .sb job. Then ESM created a .pt file per sequence that can be transformed to a final csv file with the Get_csv_from_pt_esm.py file. 

For UniRep, we installed the jax_unirep package. Note that jax required a Linux system. Afterwards, we ran the Get_Unirep_Seq.py to obtain all the sequence representations in a csv file. More details about representation choices, fine-tuning, etc. are in (UniRep: https://github.com/ElArkk/jax-unirep).


