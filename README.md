<!-- ABOUT THE PROJECT -->
## About The Contents

Here, we aimed to construct the machine leaning models, which predict whether Transcription factor (TF) binding motifs in each gene promoter is in a DEG or non-DEG promoter region. We used Lasso regrresion to constrct the models, and evaluate the models by using The area under the precision-recall curve (AUPRC) and The are under the curve (AUC). This github site include all script and data we used. Our script include the constrcing models by using Lasso, and evaluation the result by AUPRC and AUC, futhermore drawing the curve with 5 times Cross-validation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTENT -->

## resources

All table include DNA shape parameters, row of class labels, and level of expression differences.
'Gene_type' is the row of class labels. 'T' is 'True', which means these genes are detected both gSELEX-Seq and DNA microarray. 'FT' is 'False Positive', which means these genes are detected only gSELEX-Seq. 'ED' is 'Expression differences', which means levels of expression differences obtained in DNA microarray.

sense_all.csv - csv file for all genes of sense strand

antisense_all.csv - csv file for all genes of antisense strand

sense_single.csv - csv file for sense strand genes possessing one motif in their promoters

sense_multi.csv - csv file for sense strand genes possessing multiple motifs in their promoters

antisense_single.csv - csv file for antisense strand genes possessing one motif in their promoters

antisense_multi.csv - csv file for antisense strand genes possessing multiple motifs in their promoters


## script

50bp_extraction.py
  - script for serching binding motifs and extracting 22 b of upstream and downstream sequences from the motifs. This function output fasta file.

box_plot_AUPRC_AUC.py
  - script for drawing boxplot from the result of 10 times construction of Lasso regression. This script use the result of 'lasso_balance.py' or 'lasso_balance_random.py'.

 box_plot_feature_importance.py
   - script for drawing boxplot of feature importance. This results were used for Figure 4 in the paper. This script use the result of 'lasso_balance.py'.

lasso_balance.py
  - script for constructing lasso regression. This script construct lasso models 10 times with 5 times cross validation, and output the feataure importance, AUPRC, AUC, PR curve, and ROC curve.

lasso_balance_random.py
  - script for constructing lasso regression with random stratification. This script construct lasso models 10 times with 5 times cross validation, and output the feataure importance, AUPRC, AUC, PR curve, and ROC curve.

lasso_shuffle_test.py
  - script for validate the importance of specific feature in any models. This script replace one specific parameter into randomly selected parameters, and evaluate the AUPRC value with that parameters.


<!-- CONTACT -->

## Contact

Hiroya Oka - okahiro994@gmail.com

