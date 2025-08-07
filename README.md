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

### Training Notes~~~~
* Feature scaling and encodings
  * Sex, parch, sibs, embarked and Pclass features to be hot encoded
  * Age to be min max scaled/normalized, bell curve distribution
  * Fare to be transformed via log and min maxed
    * Left leaning heavy tail
* Feature notes
  * Fare cost of 0 is strongly correlated to survival only 1/15 survived with 0 cost Fare
  * 30% survival for null cabins vs 66.7% survival for ones with a cabin number
  * 