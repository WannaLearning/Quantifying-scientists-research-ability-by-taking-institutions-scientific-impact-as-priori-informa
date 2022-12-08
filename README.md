# Quantifying-scientists-research-ability-by-taking-institutions-scientific-impact-as-priori-informa
We supply the code for ***"Quantifying scientists’ research ability by taking institutions’ scientific impact as priori information"***.  
  
In this paper, we propose the institution Q-model, which integrates scientists’ affiliated institutions as valuable prior information, and jointly evaluate all scientists from different institutions. The core idea of our model is to assume that the research ability of scientists from the same institution shares the same distribution, by which our model can cope with data sparse in author-level evaluation, and also explain the research ability of institutions. This simple but intuitive  idea not surprisingly improves the performance of our model and broadens the thought for incorporating academic environment into scientific evaluation. 


# Code
***bbvi_em.py***:     the simplest model, which assumes all scientists share the same distribution.  
***bbvi_em_mp.py***:  the multiprocessing is employed to quicken learning speed.  
***bbvi_em_org.py***: The instituion Q-model presented in our paper, in which scientists from the same institution share a prior distribution. Unlike the Q-model, the productivity of a scientist also plays a non-ignorable role in the institution Q-model; that is, more high-quality articles authored by the scientist will help him/her score higher research ability in our model by twisting known prior information. Different from the original Q-model, the inferential problem of our probabilistic graphical model is to compute the posterior distribution of the hidden variable given the observed data. The BBVI-EM algorithm is employed to estimate the variational parameters and model parameters iteratively.
  
***results_compared_with_ml.py***: our model is compared with support vector regression(svr) and long-short term memory(lstm) model.  
***results_evaluate.py***: our model is compared with the Q-model. 
  
***extract_mag_affiliation.py***: data preprocessing for the MAG dataset.  
***extract_mag_aid.py***: data preprocessing for the MAG dataset.
