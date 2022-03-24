# GRAIMatter WP 1+2 meeting agenda and notes
## March 24, 2022

1. Apologies 
    + Simon, Alba
2. WP1
	- Hyperparameter/model loop experiments
    + Ongoing work to include boosted trees, SVMs, and SDGs
    + DP SVC working within the loop (some initial results here: https://github.com/jim-smith/GRAIMatter/pull/116)
    + Adding additional attack scenario comprising SDG-based shadow data as discussed last week
    + Some results: [can be found in Teams - GRAIMatter - General - files - ExperimentalResults]
      - See AUC by classifier at 95% CI and advantage by classifier at 95% CI. Note advantage is absolute difference between TPR and FPR
      - Also see [pointplots](https://dmail.sharepoint.com/:b:/r/sites/GRAIMatter/Shared%20Documents/General/ExperimentalResults/pointplots_RF_DT_SVC_ADA.pdf?csf=1&web=1&e=qOVFr9) and [density plots](https://dmail.sharepoint.com/:b:/r/sites/GRAIMatter/Shared%20Documents/General/ExperimentalResults/densityplots_RF_DT_SVC_ADA.pdf?csf=1&web=1&e=GkJ8w9)
    + Discussion points on metrics
      - Current metrics are largely properties of the whole dataset; we should also look at 'best' and 'worst'
      - JS: maybe most confident you can be when you're saying someone is in the training set, minus most confident you can be when you're saying they're not? JL: potentially by calibration: frequency of training set inclusion amongst 10% highest-probability samples minus training set inclusion amongst lowest 10%
      - FR: Important to differentiate separate concerns over accuracy of membership inference against confidence in membership inference; that is, whether someone _thinks_ they can successfully perform membership inference on a given sample, as opposed to whether they actually can. CC: again it is worth considering confidence on individual samples, rather than on average. JL: one area this might arise is in unsupervised learning of results from released models; if given some shadow data and predictions from various models, if the predictions clearly cluster with no other obvious reason an attacker may suspect clustering is due to training set membership. 
      - FR/CC: could we introduce a game in which we give various datasets to a test population of putative hackers (perhaps from the other SPRINT projects) and assess how well they do, ideally before the general SPRINT meeting?
    + Discussion points on differential privacy
      - JS: What should our recommendations be on epsilon and delta values for differentially private classifiers?
3. WP2 Update
    + Data sanity
       - JS: in an example, blood calcium measurements are given to six significant figures, which is unrealistic for real measurements. Data curation needs to be considered
    + Attribute Inference experiments
       - Extended to work with different versions of things which reduce MIA risk 
       - Strangely they increase confidence that you can predict an attribute, but decrease the accuracy. 
       - Relates to FR point about confidence versus accuracy
    + Look for missing value which gives the highest confidence in attack
    + Safe model work ready to be plugged in to hyperparameter loop
4. Action points
    + Arrange meeting to discuss use of WP2 safe models in WP1 hyperparameter loop
    + Begin assembling list of evidence-based recommendations for April meeting (JL to facilitate)
5. AOB
    + FR: PPI went well
    + CC/FR/AS to meet tomorrow regarding next PPIE session.
7. Next meeting
    + EMB to chair; if she is unable to make the meeting, she will find a substitute chair.
	
