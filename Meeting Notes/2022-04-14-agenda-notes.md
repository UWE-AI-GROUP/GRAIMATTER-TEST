## WP1 update
1. New metrics added:  
   - start running now - 
   - keras stuff not ready yet
   - can always append results when James has scenario 3 ready
   - we have budget for more hardware on AWS if needed
   -  
3. Refactored and added more unit tests 

. Looked at over-fitting metrics,  
  done some plotting of over-fitting vs min_AUC
  - fascinating results becuase no evidence of correlation
  - 
Simon suggested starting analysis by doing univariate tests for sensitivity- see what that gives us.  
- we can use different datasets for validation in a CV-sort of way
- might be able to rule out some hyperparam values then repeat
- use average of the 5 repeats rather than worst case to start with


## WP2 update
- Safe_SVC now implemented 
  xgboost hasa scikit-learn wrapper already - details in config file
- lots of refactoring of safe_model and in a structure to think about future maintenance
 - **action** liase with Albeto about keras/tensorflow 
## Ongoing / Issues

- We should collate a single *how-to...* guide for the various tools we have created
- Defining a format for a description of the input data format to be exported alongside a trained model
  (this could let us automate some attacks for the TRE output checkers). 
  HDR UK and Research Data Scotland and deciding on some standards for meta data capture
  
  We need something simple and machine readable 
  **Raise an an issue**
  
- ongoing support / maintenance: need to start planning for a new repository via conda?  
  **action all**

## AOB
Sprint meeting is on wednesday

## Next week's chair
Chris the week after.  
Chris and Jim not here next week.  
Simon to chair.  

