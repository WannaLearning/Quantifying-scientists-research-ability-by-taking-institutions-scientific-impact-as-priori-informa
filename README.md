# Quantifying scientists research ability by taking institutions scientific impact as priori information
We share the code for our paper, ***"Quantifying scientists’ research ability by taking institutions’ scientific impact as priori information"***.  
  
In this paper, we propose the institution Q-model (IQ) and its two simple variants (IQ-2 and IQ-3), which integrates institutions, countries, and collaboration as valuable prior information, and jointly evaluate all scientists from different institutions. The core idea of our model is to assume that the research ability of scientists from the same institution shares the same distribution, by which our model can cope with data sparse in author-level evaluation, and also explain the research ability of institutions. This simple but intuitive idea not surprisingly improves the performance of our model and broadens the thought for incorporating academic environment into scientific evaluation. The IQ-2 model get the optimal results.


# Code
## Model
***BBVI-EM_example.py***:  A simplest model for testing the BBVI-EM algorithm. The BBVI-EM algorithm update iteratively the variational parameters and model parameters.

***BBVI-EM_IQModel.py***: The instituion Q model (IQ model) presented in our paper, in which scientists from the same institution share a prior distribution. Unlike the Q-model, the productivity of a scientist also plays a non-ignorable role in the institution Q-model; that is, more high-quality articles authored by the scientist will help him/her score higher research ability in our model by twisting known prior information. Different from the original Q-model, the inferential problem of our probabilistic graphical model is to compute the posterior distribution of the hidden variable given the observed data.  

***BBVI-EM_IQ2Model.py***: The IQ-2 model in our paper. The IQ-2 model assume that the research ability of institutions from the same country shares the same distribution. 

***BBVI-EM_IQ3Model.py***: The IQ-3 model in our paper. The IQ-3 model use a linear combination of Q of authors to determine the quality of a paper.  

***BBVI-EM_IQ3Model_MP.py***: call BBVI-EM_IQ3Model.py

 ## Experiments
***Results_of_IQModel.py***: The results of the IQ model.

***Results_of_IQ-2Model.py***: The results of the IQ-2 model.

***Results_of_IQ-3Model.py***: The results of the IQ-3 model.

***Results_of_comparing_with_MLs.py***: our model is compared with common machine learning models.

***Results_of_Plotting_Figs.py***: Plot figure in our paper.

 ## DataProcess
***extract_mag_affiliation.py***: data preprocessing for the MAG dataset.  extract the location of institutions.

***extract_mag_aid.py***: data preprocessing for the MAG dataset. extract scientists from the CS/Phyics field.
