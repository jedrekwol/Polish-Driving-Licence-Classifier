# PoliDriving License Questions Classifier

## Project Overview

The Polish Driving License Questions Classifier is a versatile tool for performing natural language processing (NLP) and machine learning (ML) tasks related to Polish driving license questions. It includes modules for data transformation, model optimization, text processing, and Jupyter notebooks for generating predictive models.

## Features

- **Data Transformation**: Load and preprocess data from Excel files.
- **Model Optimization**: Optimize machine learning model hyperparameters using Optuna.
- **NLP Tasks**: Perform common NLP tasks, including lemmatization, stopword removal, and punctuation removal.
- **Utility Functions**: A collection of utility functions for various tasks.
- **Jupyter Notebooks**: Explore Jupyter notebooks for training and evaluating ML models on driving license questions.

## Repo structure

```
├── data
│   └── questions.xlsx
├── main.py
├── notebooks
│   └── Driving License analysis pl.ipynb
├── readme.md
├── requirements.txt
├── src
│   ├── data_transformer.py
│   ├── modeler.py
│   ├── nlp_modeler.py
│   └── optimizer.py
└── utils
    └── functions.py
```


## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the project, subject to the terms and conditions specified in the license.



* model and analysis are stored in jupyter file
* backend is in other files
* files and folder stucture is as follows

## Dependencies

Install requirements

`pip install -r requirements.txt`

install language model

`python -m spacy download pl_core_news_lg`