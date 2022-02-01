# Public data sets

The following is a description of publicly available datasets that may be useful within the project.

## Imaging

### Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning

[Paper](https://www.sciencedirect.com/science/article/pii/S0092867418301545?via%3Dihub)

Uses transfer learning (original network trained on imagenet) to perform multiclass (4) classification on OCT images. Includes approx 108,000 from about 4,000 patients. Data available [here](https://data.mendeley.com/datasets/rscbjbr9sj/3). Code included with data.

### IXI

[Brain MRI data](https://brain-development.org/ixi-dataset/)

### Kaggle Medical MNIST

[64 x 64 images](https://www.kaggle.com/andrewmvd/medical-mnist)

## Medical Records

### MIMIC

[MIMIC](https://mimic.mit.edu/) is a very large resource of medical record type data. Looks like it would certainly be adequate for our needs. Perhaps best strategy is to find papers that use this and piggyback on their pipelines to turn the data into standard ML tasks.

For example, [Prediction model of in-hospital mortality in intensive care unit patients with heart failure: machine learning-based, retrospective analysis of the MIMIC-III database](https://bmjopen.bmj.com/content/11/7/e044779) to build a model to predict in-hospital mortality. They processed MIMIC data to create a suitable dataset, and it is [freely available](https://datadryad.org/stash/dataset/doi:10.5061%2Fdryad.0p2ngf1zd)

Dataset based on [MIMIC-II database](https://physionet.org/content/mimic2-iaccd/1.0/) is freely available.

[Machine Learning Approach to Predict Positive Screening of Methicillin-Resistant Staphylococcus aureus During Mechanical Ventilation Using Synthetic Dataset From MIMIC-IV Database](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8635043/). It claims data will be made available.

[Central venous pressure measurement is associated with improved outcomes in septic patients: an analysis of the MIMIC-III database](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7358999/) [datasets available](https://physionet.org/works/MIMICIIIClinicalDatabase/files/)

[Deep-Learning-Based Survival Prediction of Patients in Coronary Care Units.](https://www.hindawi.com/journals/cmmm/2021/5745304/) [data available](https://mimic.mit.edu/)

[Convolutional Neural Network Model for Intensive Care Unit Acute Kidney Injury Prediction](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8116756/)

[A Racially Unbiased, Machine Learning Approach to Prediction of Mortality: Algorithm Development Study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7644374/)

### Paediatric pneumonia liver segmentation daatasets
Referenced in Zillet at al (2021) *Medical imaging deep learning with differential privacy* [paper](https://www.nature.com/articles/s41598-021-93030-0.pdf)

[Paediatric pneumonia](https://data.mendeley.com/datasets/rscbjbr9sj/3)

[liver segmentation dataset](http://medicaldecathlon.com)



## Genetic data

European Genome-Phenome Archive [EGA](https://ega-archive.org/). Access needs to be requested for each data set.

The database of Genotypes and Phenotypes [dbGaP](https://www.ncbi.nlm.nih.gov/gap/). This database might be crosslinked witht EGA. Also access need to be requested to access some of the data.

For example the *Lothian Birth cohort* contains several types of genomics data, some of which access can be requested. An example is this dataset [methilation age and mortality](https://ega-archive.org/studies/phs000821)
