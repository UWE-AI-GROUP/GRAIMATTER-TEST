class SafeModel():
    import pickle, getpass
    def __init__(self):
        '''super class  constructor, gets researcher name'''
        self.modelType="None"
        self.model=None
        try:
            self.researcher = self.getpass.getuser()
        except: #shouldn't ever happen
            self.researcher="unknown"
            
    def saveModel(self, name="undefined"):
        '''writes model to file in apropriate format'''
        #write model to pickle file
        self.modelSaveFile= name
        while(self.modelSaveFile=="undefined"):
            self.modelSaveFile= input("Please input a name with extension for the model to be saved.")
        #TODO implement more file types
        if (self.modelSaveFile[-4:]==".pkl"):
            self.pickle.dump(self.model, open(self.modelSaveFile, 'wb'))
        else:
            print("only .pkl file saves currently  implemented")
        

    def getConstraints(self):
        '''gets constraints  relevant to the model type from the master read-only file'''
        paramsdict = {}
        #TODO change to json format from text?
        with open("params.txt",'r') as file:
            for line in file:
                contents= line.split()
                if contents[0] == self.modelType:
                    key = contents[1]
                    value = [contents[2], contents[3]]
                    paramsdict[key] = value
        return paramsdict
    
    
    
    def applyConstraints(self,**kwargs):
        '''sets model attributes according to constraints'''
        paramsDict = self.getConstraints()
        for key in kwargs:
            setattr(self.model, key, kwargs[key])
        for key in paramsDict:
            #TODO distinguish between ints and floats as some models take both and behave differently
            if(paramsDict[key][0]=="min" or paramsDict[key][0]=="max"):
                setattr(self.model, key,int(paramsDict[key][1]))
            else:
                setattr(self.model, key,paramsDict[key][1])
    
    def checkModelParams(self):
        '''Checks whether current model parameters have been changed from constrained settings'''
             # check through to see if sensitive params have been altered
        possiblyDisclosive = False
        paramsCheckTxt = ""
        paramsDict=self.getConstraints()

        for key in paramsDict:
            operator = paramsDict[key][0]
            recommendedVal= paramsDict[key][1]
            currentVal= str(eval(f"self.model.{key}"))
            print(f"checking key {key}: currentval {currentVal}, recomended {recommendedVal}")

            if (currentVal==recommendedVal):
                thisSubString= "\tparameter " + key + " unchanged at recommended value " + recommendedVal
            elif paramsDict[key][0]=="min":
                if float(currentVal) > float(recommendedVal):
                    thisSubString= "\tparameter " +key +" increased from recommended min value of " +recommendedVal
                    thisSubString= thisSubString + " to " + currentVal +", this is not problematic.\n"
                else:
                    thisSubString= "\tparameter " +key+ " decreased from recommended min value of " +recommendedVal
                    thisSubString= thisSubString+ " to " + currentVal +", THIS IS POTENTIALLY PROBLEMATIC.\n"
                    possiblyDisclosive = True
            elif paramsDict[key][0]=="max":
                if float(currentVal) < float(recommendedVal):
                    thisSubString= "\tparameter " +key +" decreased from recommended max value of " +recommendedVal
                    thisSubString= thisSubString+ " to " + currentVal +", this is not problematic.\n"
                else:
                    thisSubString= "\tparameter " +key +" increased from recommended max value of " +recommendedVal
                    thisSubString= thisSubString+ " to " + currentVal +", THIS IS POTENTIALLY PROBLEMATIC.\n"
                    possiblyDisclosive = True
            elif paramsDict[key][0]=="equals":
                thisSubString= "\tparameter " + key +" changed from recommended fixed value of " +recommendedVal
                thisSubString= thisSubString+ " to " + currentVal + ", THIS IS POTENTIALLY PROBLEMATIC.\n"
                possiblyDisclosive = True
            else:
                thisSubString("\tunknown operator in parameter specification " + paramsDict[key][0])
        
            paramsCheckTxt = paramsCheckTxt + thisSubString + "\n"
        return paramsCheckTxt, possiblyDisclosive
            
    def requestRelease(self,filename="undefined"):
        '''Saves model to filename specified and creates a report for the TRE output checkers'''
        if(filename=="undefined"):
            print("You must provide the name of the file you want to save your model into")
            print("For security reasons, this will overwritw previous versions")
        else:
            #resave thew model
            # ideally we would then prevent over-writing
            self.filename=filename
            self.saveModel(filename)
            
            # creates a report for TRE output checkers
            paramsCheckTxt, possiblyDisclosive = self.checkModelParams() 
        
            outputfilename= self.researcher+"_checkFile.txt"
            with open(outputfilename,'a') as file:
                file.write (f" {self.researcher} created model of type {self.modelType} saved as {self.modelSaveFile} \n")
                if(possiblyDisclosive==False):
                    file.write(f"Model has not been changed to increase risk of disclosure, these are the params:\n   {paramsCheckTxt}")
                else:
                    file.write(f"WARNING: model has been changed in way that increases disclosure risk:\n  {paramsCheckTxt}\n")
    
            
    def preliminaryCheck(self):
        '''Allows user to  test whether model parameters break safety constraints prior to requesting release'''
        #report to user before they request release
        paramsCheckTxt, possiblyDisclosive = self.checkModelParams()
        if(possiblyDisclosive==False):
            print(f"Model has not been changed to increase risk of disclosure, these are the params:\n")
        else:
            (f"WARNING: model has been changed in way that increases disclosure risk:\n")                              
        print(paramsCheckTxt + "\n")        
            
            
    def __str__(self):
        '''returns string with model description'''
        output = self.modelType + ' with parameters: ' +str(self.model.__dict__)
        return output
            
            
            
class SafeDecisionTree(SafeModel):
    from sklearn.tree import DecisionTreeClassifier as DT
    
        
    def __init__(self,**kwargs):
        ''' Creates model and applies constraints to params'''
        #TODO allow users to specify other parameters at invocation time
        #TODO consider moving specification of the researcher name into a separate "safe_init" function
        super().__init__()
        self.modelType="DecisionTreeClassifier"
        self.model = self.DT()
        super().applyConstraints(**kwargs)
   
    
    def apply(self,X, check_input=True):
        '''Return the index of the leaf that each sample is predicted as.'''
        return self.model.apply(X, check_input=check_input)
    
    def cost_complexity_pruning_path(self,X, y, sample_weight=None):
        '''Compute the pruning path during Minimal Cost-Complexity Pruning.'''
        ccp = self.model.cost_complexity_pruning_path(X, y, sample_weight=sample_weight)
        return ccp
        
    def decision_path(self,X, check_input=True):
        '''Return the decision path in the tree.'''
        return self.model.decision_path(X, check_input=check_input)
    
    def fit(self,X, y, sample_weight=None, check_input=True, X_idx_sorted='deprecated'):
        '''Build a decision tree classifier from the training set (X, y).'''
        self.model.fit(X, y, sample_weight=sample_weight, check_input=check_input, X_idx_sorted=X_idx_sorted)
        return self.model     
        
    def get_depth(self):
        '''Return the depth of the decision tree.'''
        return self.model.get_depth()
    
    def get_n_leaves(self):
        '''Return the number of leaves of the decision tree.'''
        return self.model.get_n_leaves()
    
    def get_params(self,deep=True):
        '''Get parameters for this estimator.- AN EXAMPLE OF A METHOD BEING BLOCKED'''
        return "This function is deprecated in the SafeMode lclass, please use the method getParams()"

    def predict(self,X, check_input=True):
        '''Predict class or regression value for X.'''
        return self.model.predict(X, check_input=check_input)
    
    def predict_log_proba(self,X):
        '''Predict class log-probabilities of the input samples X.'''
        return self.model.predict_log_proba(X)
    
    def predict_proba(self,X, check_input=True):
        '''Predict class probabilities of the input samples X.'''
        return self.model.predict_proba(X, check_input=check_input)
    
    def score(self,X, y, sample_weight=None):
        '''Return the mean accuracy on the given test data and labels.'''
        return self.model.score(X,y,sample_weight=sample_weight)
    
    def set_params(self,**params):
        '''Set the parameters of this estimator.'''
        #TODO  check against recommendations and flag warnings here
        self.model.set_params(**params)
    
   
