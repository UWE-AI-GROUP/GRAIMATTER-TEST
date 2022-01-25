# Must
- Read "risky" pararameter combinations for supported algorithms from read-only file
- Provide user with option to select safe version of class implementing constructor(), fit(), predict() and saveModel() methods
- Provide user with option to call requestRelease() method that takes as input names of saved mode files and generates report for TRE output checkers
- Integrate checks into saveModel() s  to check whether user has changed model parameters from default
- Provide TRE with useful report detailing what algorithms the user ran to generste model on which they are calling saveModel()
- Behave gracefully when users calls saveModel() (possibly more than once) but forgets to call requestRelease()
- Distinguish between "support" classes/methods (e.g. prteprocessing, choosing metrics) and classes that repressent model types

# Should
- Integrate post-hoc static analysis to check whether user has set model parameters for safe class outside the constructor
- Integrate post-hoc static analysis to check whether user has  made direct calls to the underlying models fit() 
method
- Integrate post-hoc analysis to report on any direct calls to sklearn model classes or keras/pytorch model classes
- Support different filetypes for model save - pkl, hd5 etc 



# Could
- Give user option option to select between different DP variants of fit() method where available
- Make saved model files and TRE report read-only so malicious users can't edit after creation
- Integrate knowledge from "risky parameters combinations" into static analysis e.g.  
   -  "they've called X directly but in a safe way"
   -  "they've called Y but it is difficult to tell if it is safe because they are using a variable value for parameter Z"



# Won't
