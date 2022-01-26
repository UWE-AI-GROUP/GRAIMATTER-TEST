class SafeModel():
    import pickle, getpass
    def __init__(self):
        self.modelType="None"
        self.model=None
        try:
            self.researcher = self.getpass.getuser()
        except: #shouldn't ever happen
            self.researcher="unknown"
            
    def saveModel(self, name="undefined"):
        #write model to pickle file
        self.modelSaveFile= name
        while(self.modelSaveFile=="undefined"):
            self.modelSaveFile= input("Please input a name with extension for the model to be saved.")
        if (self.modelSaveFile[-4:]==".pkl"):
            self.pickle.dump(self.model, open(self.modelSaveFile, 'wb'))
        else:
            print("only .pkl file saves currently  implemented")
        

    def getParams(self):
        paramsdict = {}
        with open("params.txt",'r') as file:
            for line in file:
                contents= line.split()
                if contents[0] == self.modelType:
                    key = contents[1]
                    value = [contents[2], contents[3]]
                    paramsdict[key] = value
        return paramsdict
    
    
    
    def applyParams(self,**kwargs):
        paramsDict = self.getParams()
        for key in kwargs:
            setattr(self.model, key, kwargs[key])
        for key in paramsDict:
            if(paramsDict[key][0]=="min" or paramsDict[key][0]=="max"):
                setattr(self.model, key,int(paramsDict[key][1]))
            else:
                setattr(self.model, key,paramsDict[key][1])
    
    def checkModelParams(self):
             # check through to see if sensitive params have been altered
        possiblyDisclosive = False
        paramsCheckTxt = ""
        paramsDict=self.getParams()

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
        #report to user before they request release
        paramsCheckTxt, possiblyDisclosive = self.checkModelParams()
        if(possiblyDisclosive==False):
            print(f"Model has not been changed to increase risk of disclosure, these are the params:\n")
        else:
            (f"WARNING: model has been changed in way that increases disclosure risk:\n")                              
        print(paramsCheckTxt + "\n")        
            
            
    def __str__(self):
        output = self.modelType + ' with parameters: ' +str(self.model.__dict__)
        return output
            
            
            
class SafeDecisionTree(SafeModel):
    from sklearn.tree import DecisionTreeClassifier as DT
    
        
    def __init__(self,**kwargs):
        #TODO allow users to specify other parameters at invocation time
        #TODO consider moving specification of the researcher name into a separate "safe_init" function
        super().__init__()
        self.modelType="DecisionTreeClassifier"
        self.model = self.DT()
        super().applyParams(**kwargs)
   
        
        
    
    def fit (self,X, y, sample_weight=None, check_input=True):
        self.model.fit(X,y,sample_weight,check_input)
        return self.model
    
    def score(self,X, y, sample_weight=None):
        return self.model.score(X, y, sample_weight)
    
 
