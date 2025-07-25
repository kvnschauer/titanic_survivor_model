## Description
Project to predict titanic survivors based off Kaggle data.

### TODO
* Visualize distributions of each feature and find appropriate scaler
* Test out different models to see performance
  * Validate its not underfitting or overfitting

### Notes
* Program is run via command line from the main.py file
* Options are to analyze the data or train the model
  * After training performance is analyzed

### Training Notes
* Feature scaling and encodings
  * Sex and Pclass features to be hot encoded
  * Age to be min max scaled/normalized, bell curve distribution