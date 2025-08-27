## Description
Project to predict titanic survivors based off Kaggle data.

### TODO
* Visualize distributions of each feature and find appropriate scaler
* Test out different models to see performance
  * Validate it's not underfitting or overfitting

### Notes
* Program is run via command line from the main.py file
* Options are to analyze the data or train the model
  * After training performance is analyzed

### Training Notes
* Feature scaling and encodings
  * Sex, parch, sibs, embarked and Pclass features to be hot encoded
  * Age to be min max scaled/normalized, bell curve distribution
    * Empty vals filled with average age for that sex
  * Fare to be transformed via log and min maxed
    * Left leaning heavy tail
  * Ticket feature to be split into Ticket Number and Ticket Text features
  * Name feature to be dropped as it is likely not relevant
* Feature notes
  * Fare cost of 0 is strongly correlated to survival only 1/15 survived with 0 cost Fare
  * 30% survival for null cabins vs 66.7% survival for ones with a cabin number
  * cabin data point to be left out for now as it is closely related to class and only 23% of instances have this data point
* Training notes
  * initial cross val score results with default params
    * SGDClassifier
      * [0.69273743 0.80337079 0.76966292 0.78089888 0.80898876]
      * average = 0.771131756
    * LinearRegression
      * [0.29813726 0.36337874 0.37446556 0.32983911 0.47047528]
      * average = 0.36725919
    * LinearSVC
      * [0.80446927 0.83707865 0.79213483 0.79213483 0.81460674]
      * average = 0.808084864
    * DecisionTreeClassifier
      * [0.70949721 0.74157303 0.76966292 0.79213483 0.79213483]
      * average = 0.761000564