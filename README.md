# Match prediction enhancement for dating applications
Match prediction for dating app using a [Speed Dating](https://www.kaggle.com/datasets/whenamancodes/speed-dating) survey dataset from Kaggle. Final project for the course "Machine Learning and Intelligent Systems" @Eurecom
___

## Project structure

- [data](./data) Raw and processed data
    - [raw_data.csv](./data/raw_data.csv) original dataset.
    - [data.pkl](./data/data.pkl) cleaned and reshaped dataset produced by the [Dataset analysis](Dataset_analysis.ipynb) notebook.
    - [SpeedDatingSurveyAndDataKey.doc](./data/SpeedDatingSurveyAndDataKey.doc) Original survey given to participants annotated with correspondant dataset fields names.
- [images](./images/) Images used in reports
- [reports](./reports/) Reports produced at the beginning and during the development
- [Dataset_analysis](./Dataset_analysis.ipynb) Dataset analysis, cleaning, reshape and features selection.
- [Isolation_Forest](./Isolation_Forest.ipynb) Isolation forest model experiments.
- [Models_Pipeline](./Models_Pipeline.ipynb) ML models trainings and testing.
- [Principal_Component_Analysis](./Principal_Component_Analysis) PCA
- [gmm.py](./gmm.py) Python code defining a GMM class used in [models.py](./models.py)
- [models.py](./models.py) Python code implementing models and pipelines used in the notebook [Models_Pipeline](./Models_Pipeline.ipynb)
- [preprocessing.py](./preprocessing.py) Python code implementing preprocessing functions (interactions computations, PCA, training/test split) used in [Models_Pipeline](./Models_Pipeline.ipynb)
- [requirements](./requirements.txt) Project's dependencies
- [utilities.py](./utilities.py) Nice printing functions.
