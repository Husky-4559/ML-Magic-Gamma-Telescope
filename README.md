# MAGIC Gamma Telescope Classification


This project aims to classify gamma-ray events in telescope data using machine learning techniques. The dataset is the MAGIC Gamma Telescope Dataset, which contains real-world data from the MAGIC telescopes used for identifying high-energy gamma particles. The goal of the project is to apply various machine learning models to classify events as gamma (signal) or hadron (background) events, contributing to the advancement of gamma-ray astronomy.

## Table Of Contents
	•	Project Overview
	•	Installation
	•	Dataset
	•	Modeling Approach
	•	Evaluation
	•	How to Use
	•	Technologies Used
	•	Future Work

### Project Overview

In this project, I leverage Python and machine learning libraries to classify gamma-ray events based on a dataset from the UCI Machine Learning Repository. The project applies data preprocessing techniques, machine learning algorithms, and model evaluation metrics to develop a robust classification model.

As an entry-level software engineer, my objective is to demonstrate my problem-solving skills, understanding of machine learning, and ability to build scalable models in real-world use cases via:

	•	Solving complex problems using data-driven approaches
	•	Working with large datasets
	•	Implementing and evaluate various machine learning models

### Installation

To run this project locally, follow these steps:

1. Clone The Repository:
  git clone https://github.com/Husky-4559/ML-Magic-Gamma-Telescope.git

2. Navigate To The Project Directory:
  cd ML-Magic-Gamma-Telescope

3. Run The Jupyter Notebook:
   jupyter notebook MAGIC_Gamma_Telescope.ipynb

### Dataset

The dataset used for this project is the MAGIC Gamma Telescope Dataset. It consists of 10,000+ events recorded by the MAGIC telescope, labeled as either gamma (signal) or hadron (background). The dataset contains the following features:

	•	fLength: Major axis of the ellipse (continuous)
	•	fWidth: Minor axis of the ellipse (continuous)
	•	fSize: Total event size (continuous)
	•	fAlpha: Angle of the event (continuous)
	•	And several other relevant features that characterize the event shape.

The goal is to build a machine learning model that predicts whether an event is a gamma or hadron event based on these features.


### Modeling Approach

The project follows these key steps:

	1.	Data Preprocessing:
	•	Handling missing values
	•	Feature scaling using StandardScaler
	•	Train-validation-test split to evaluate model performance
	2.	Model Training:
	•	Experimentation with multiple machine learning algorithms:
	•	Logistic Regression
	•	Random Forest
	•	Gradient Boosting Classifier
	•	Neural Networks (for advanced trials)
	•	Hyperparameter tuning for optimal model performance.
	3.	Model Evaluation:
	•	Models are evaluated using various metrics such as accuracy, precision, recall, and F1-score.
	•	Cross-validation to ensure model generalization.
	•	Overfitting prevention using dropout and regularization techniques.

### Evaluation

The models are evaluated using the following metrics:

	•	Accuracy: Overall correctness of the model.
	•	Precision: Ability to correctly classify positive samples.
	•	Recall: Ability to capture all relevant positive samples.
	•	F1-score: A balance between precision and recall.

For the best performing model, Random Forest and Gradient Boosting yielded high precision and recall on classifying gamma events. Further optimization can be done to improve model generalization.

### How to Use

	1.	Run the notebook and load the dataset.
	2.	Train the models by executing the code cells.
	3.	Use the pre-trained models to classify new gamma-ray events.

Example

The following Python function can be used to predict a single event’s classification:
  def predict_event(model, event):
    return model.predict(event)

### Technologies Used

	•	Python: Core programming language for the project.
	•	Jupyter Notebook: Used for the interactive coding environment.
	•	Google Colab: Used for writing and executing code in a cloud-based environment.
	•	scikit-learn: For model building and evaluation.
	•	NumPy & Pandas: For data manipulation.
	•	Matplotlib & Seaborn: For data visualization.
	•	TensorFlow/Keras: For deep learning model trials.

### Future Work

This project provides a baseline for classifying gamma events using classical machine learning models. The next steps include:

	•	Exploring Neural Networks for more complex feature interactions.
	•	Incorporating additional features to improve classification.
	•	Hyperparameter optimization using more advanced techniques such as grid search or random search.

With this project I hope to showcase my ability to build, tune, and evaluate machine learning models on real-world data. I aspire to utilize data and become better at problem-solving, data-driven decision making, and the development of AI models that contribute to meaningful outcomes.

### Acknowledgments

Data retrieved from the UCI Machine Learning Repository.
 
