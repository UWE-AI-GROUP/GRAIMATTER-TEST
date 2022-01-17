# Draft Workplan for WP2/ UWE
Subject to changes as we progress

## Milestone 1 (End March): Preliminary Findings and Initial Tools
End of March we have a meeting of all the different `sprint exemplar' projects,  
where we will present drafts of what we have created so far for feedback.  
Currently I think this should include:

**Coupled with WP1 and possibly WP3**
1. Prioritised list of algorithms to be considered (**G1**)
2. Criteria for assessing reliability/stability of repository: CRAN vs pypi vs ... (**G2**)
3. Criteria for assessing software quality of specific tools: size of team, frequency of updates , ... (**G3**)
4. List of datasets to be considered. (**G4**)

**WP2 Specific**
1. Report on what types of attack should be considered if ML is applied within a setting of federated TREs. (Andy)
  - Ideally to include a section on what the implications might be for using any privacy tools identified.
2. List of open source pricacy assessment tools (Richard) broken down into:
 - Assessment vs criteria G3 and G3
 - Experimental protocol for assessing computational burden of tools   
   - series of tests such as *Algorithm A1 trained on Dataset D1*, *Algorithm A2 trained on Dataset D1*, etc.
   - details of h/w platform used for assessment (e.g. HICs AWS)
 - initial experimental results.
3. Initial prototype of wrapper class for a subset of algorithms identified in G1.  
   To include:
   - brief presentation of concept and workflow to show to TREs, PPIE people (?), and representative group of researchers
   - **AIM: establish a group of TREs and researchers to do co-creation of research wrt wrappers for phase 2** 
4. For each of the algorithms in G1, details of what (if any) DP-variants are available. 
   NB. selecting only those variants that meet the criteria G2 and G3 to make the process achievable 
   - Summary of claimed results in original papers
   - Experimental protocol for testing them, especially wrt empirical trade off between budget, accuracy and disclosure risk
   - Initial results
5. Synthetic data / DP-variants that work by disturbing the input data rather than  the training gradients.

## Milestone 2 (end June): Detailed work
Decision about  what is going to go into the final report and tools, even if they are not completed
1. Is DP (when available) enough?
2. What is the best trade-off between (A) "designing for privacy" vs. (B) "design for flexibility then apply model-checking tools" ?
3.  (A) Privacy by design:  
    Wrapper classes authored and tested for selected subset of methods. 
    - needs testing protocol etc. but informed by work for first milestone
4.  (B) Privacy by Testing  
   4.1 Need some post-hoc code analysis tool?  
   4.2 What model-checking tools are available, robust and easy to use:
     - which algorithms do they cover (or not)
     - how can we provide (automated?) help to users to (i) configure them, (ii) run them , and (iii) intepret their results
