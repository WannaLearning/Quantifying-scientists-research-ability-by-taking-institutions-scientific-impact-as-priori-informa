# Quantifying-scientists-research-ability-by-taking-institutions-scientific-impact-as-priori-informa
We supply the code for ***"Quantifying scientists’ research ability by taking institutions’ scientific impact as priori information"***

# Code
***bbvi_em.py***:     the simplest model, which assumes all scientists from the same institution.  
***bbvi_em_mp.py***:  the multiprocessing is employed to quicken learning speed.  
***bbvi_em_org.py***: The instituion Q-model adpoted in our paper, in which scientists from the same institution share the same prior distribution. The BBVI-EM algorithm is employed to estimate the variational parameters and model parameters iteratively.    
***results_compared_with_ml.py***: our model is compared with svr and lstm model  
***results_evaluate.py***: our model is compared with the Q-model  
***extract_mag_affiliation.py***: data preprocessing  
***extract_mag_aid.py***: data preprocessing
