# GRAIMatter WP1+2 Kick-off meeting 13/01/2021

**Action for WP leads**: formulate plan for these  assumptions  to be checked with representative researchers, TRE output checkers, PPIE groups

## Assumption 1

Given the groundswell of research,   increasing numbers of researchers will be under the impression that using DP-variants of their preferred methods will solve all their problems, 

- and we may face will kick back/ reduced uptake if we suggest something else  
  **this seems obvious to me but I may be wrong about the visibility of DP**

**Action 1**: tabulate, for common methods, whether DP versions exist in _trustworthy_ repositories. i.e. pypi / CRAN.   
This means agreeing:
1. list of trustworthy repositories (pypi, CRAN, ...)
2. criteria for inclusion of algorithms in repositories (tensorflow, pytorch, IBMDiffpriv vs. a.n.student)


## Assumption 2
The vast majority of TRE users will be using pre-implemented methods – i.e. we’re not worried about people coming in to develop their new ML from scratch.

## Assumption 3
TRE output checkers will not understand ML and just want a simple report saying  
*researcher R used method X this is the risk ...*

## Assumption 4
PPIE partners will be more comfortable with controls we can explain easily.
 - Especially they will prefer a single/few consistent answers

## Assumption 5
For PPIE partners, trustworthiness accrues from the people who select and implement an algorithm, not from the algorithm alone 
- i.e., they understand the difference between a method that is good in theory and a ‘good’ implementation of that method.  
My guess is that they get this intuitively since so many apps seem great but are flaky.  
An example we might suggest  could be a combination lock with lots of dials – people will intuitively understand this will keep their possession safe,  unless it has been made out of cheap materials

- **important because this heavily influence the recommendations we make.**
 - **I’d like to assume the latter so we can rule lots of papers out of scope immediately**

## Assumption 6
Any toolsets we recommend should be simple for TRE IT staff to deploy e.g. based on standard mechanisms like anaconda or  “pip install”.   
- If a tool/algoroithm need building that is enough to rule it out of scope.
- **Action**: any tools we create should be put in pypi before end of project  
  liase with Chris and other TRE providers to see if pip/conda/another is their preference

## Question 0: What do we report to TRE output checkers?
and do researchers see the same report?

Can we provide the TRE output checkers with a report just based on the algorithm + hyper-parameter choices?  (WP1)

- YES: This could be something that is done by the wrapper class (WP2)
-  NO, which tools should they also run? With what settings,  
   can we automate that process for them? (WP2)

**Action** start planning network activities with TRE output checkers to answer the central question here

### Scenario 1: Researcher wants to use supervised ML algorithm A
If there is NO DP-version of A: 
-  how do we quantify risk of A?  (WP1)  
   Is it inherent (k-NN/SVM)  
   or does it relate to hyper-parameter choices, 
- if possible, how do we remediate that risk (WP2)  
   e.g. enforce good parameter choices,  
   ‘hide’ SVM support vectors inside wrapper class so they can’t be accessed. 

 
If there are 1+ DP-versions of A:    
- what is the most appropriate DP-variant  (dataset noise, SGD noise, gaussian vs Renyi 
- for a given variant, what is the most stable toolchain to recommend?
  e.g. based on pytorch or tensorflow or another
- **for a given variant, what is the relationship between privacy budget and the risk of membership or attribute inference?** (WP1)
- how do we propose / report on the choice of privacy budget? (WP2)
- **is DP enough, or do we need to enforce additional constraints?** (WP2)
 
 

### Scenario 2: Researcher wants to use unsupervised algorithm B
Similar questions to scenario 1.

DP versions of k-means exist.  (but I haven't read the papers)
-    Is that because they perturb update gradients for EM in the same way that DP for MLP perturbs updates for SGD?
- What about algorithms that don’t use iterative (and therefore perturbable) updates?  
  E.g. DBSCAN, agglomerative clustering. 
  How would you consider releasing a model like that?   
  As the convex hull of the clusters?   
  Q: If so can noise be added to the convex hull to give privacy guarantees?     
- Or in that case do you use a variant of DP based on input perturbation?

 

### Action from scenarios 1 and 2
Collectively collate a list of algorithms we are supporting prioritisied for
-  testing their privacy risk (WP1)
- testing automated tools for assessing individual models learned by those algoritsm (WP2)


### Scenario 3: Researcher wants do some exploratory work prior to supervised learning
e.g. clustering, visualisation prior to creating supervised ML model but is agnostic about what algorithms to use
 - how do we tackle the desire to permit outputs from exploratory work e.g. visualisations using PCA/TSNE/UMAP reductions, what are the risks involved there?

 - how do we make them aware of what algorithms we are allowing
- how do we make them  and the TRE output checker aware of the relationships between methods for protecting unsupervised algorithms, and those for supervised algorithms


## Other points 1: Creating Synthetic dataasets
- How do we explain the difference to ‘traditional DP’?
- How do we quantify the risk that as the synthetic dataset becomes more realistic, it effectively contains members of the original training set?
- What's the workplan here?


## Other points 2: Federated ML across TREs
Not out of scope because several other exemplar project are proposing this

I suggest we do a piece of work focussed on answering
*What are the relevant forms of attack to consider for (DP) Federated learning across TREs?*
- e.g. is poisoning irrelevant?

