### Must
1. Keep as much reporting functionality as possible in super class
2. Keep as much  parameter checking as possible in super class
3. Support restrictions composed of combinations of parameters (AND, OR, multiplied)

### Should
1. Infer user's name automatically/programmatically rather than in constructor to keep constructor invocation as close as possible to sklearn
2. Create a separate preliminaryReport() method the user could run before they sent their model for release
3. Rename MakeReport() to RequestRelease("modelname") to make it more transparent what the user is asking to happen

### Could
1. Have a single read-only parameters.txt file containing restrictions for all models
2. Have a SafeModel.describeRisks("algorithmType") method the researcher can call **before** they run a model.
   This would explain the risks form certain types of algorithms/models and what parameter or optimiser settings should be used.
3. Have a "requestException"() method the user could use to input text justifying why the automatic rules should be over-ridden in their case

### Won't
1. Run the post-hoc tests Richard is proposing in V1 - although these could be linked later.
