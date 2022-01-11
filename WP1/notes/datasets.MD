# Public data sets

The following is a description of publically available datasets that may be useful within the project.

## Imaging

### Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning

[Paper](https://www.sciencedirect.com/science/article/pii/S0092867418301545?via%3Dihub)

Uses transfer learning (original network trained on imagenet) to perform multiclass (4) classification on OCT images. Includes approx 108,000 from about 4,000 patients. Data available [here](https://data.mendeley.com/datasets/rscbjbr9sj/3). Cannot find a link to code in the paper.

### IXI

[Brain MRI data](https://brain-development.org/ixi-dataset/)

### Kaggle Medical MNIST

[64 x 64 images](https://www.kaggle.com/andrewmvd/medical-mnist)

## Medical Records

### MIMIC

[MIMIC](https://mimic.mit.edu/) is a very large resource of medical record type data. Looks like it would certainly be adequate for our needs. Perhaps best strategy is to find papers that use this and piggyback on their pipelines to turn the data into standard ML tasks.

For example, [Prediction model of in-hospital mortality in intensive care unit patients with heart failure: machine learning-based, retrospective analysis of the MIMIC-III database](https://bmjopen.bmj.com/content/11/7/e044779) to build a model to predict in-hospital mortality. They processed MIMIC data to create a suitable dataset, and it is [freely available](https://datadryad.org/stash/dataset/doi:10.5061%2Fdryad.0p2ngf1zd)

### Paediatric pneumonia liver segmentation daatasets
Referenced in Zillet at al (2021) *Medical imaging deep learning with differential privacy* [paper](https://www.nature.com/articles/s41598-021-93030-0.pdf)

[Paediatric pneumonia] (https://data.mendeley.com/datasets/rscbjbr9sj/3)

[liver segmentation dataset](http://medicaldecathlon.com)
