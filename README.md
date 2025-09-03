## Description
Project to predict titanic survivors based off Kaggle data.
Current model predicts with 78% accuracy on test set. Current leaderboard position (as of 9/2/25) on Kaggle 3007/13165 top 23%.
https://www.kaggle.com/kevinschauer

### TODO
* Improve model further to reach better than 0.77990% on test set
* Experiment with other ensemble methods such as boosting or stacking or using combination of other models

### Notes
* Program is run via command line from the main.py file
* Options are to analyze the data or train the model or predict
  * After training performance is analyzed

### Training Notes
* Feature scaling and encodings
  * Sex, parch, sibs, embarked and Pclass features to be hot encoded
  * Age to be min max scaled/normalized, bell curve distribution
    * Empty vals filled with average age for that sex
  * Fare to be transformed via log and min maxed
    * Left leaning heavy tail
  * Ticket feature to be split into Ticket Number and Ticket Prefix features
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
    * SVC 10th degree c=1
      * degree = 10, c = 1 
      * [0.69832402 0.73595506 0.7752809  0.78089888 0.7752809 ]
      * average = 0.753147952
    * RandomForestClassifier
      * n_estimators = 500, max_leaf_nodes=16
      * [0.76536313 0.82022472 0.84269663 0.78089888 0.8258427 ]
      * average = 0.807005212
  * Ticket Prefix feature dropped as it degrades performance
    * Random Forest feature_importances_ shows this
  * Random forest GridSearchCV best accuracy and params
    * 0.83841566756638
    * {'classifier__max_depth': 15, 'classifier__max_leaf_nodes': 100, 'classifier__n_estimators': 500}
    * Overfits (94.6% training accuracy vs 83.6% OBB accuracy)
  * Best Random Forest params after correcting for overfitting
    * Training Score: 0.8451446864603603 
    * OBB score: 0.8338945005611672 
    * Best params: {'estimator__max_depth': 10, 'estimator__min_samples_split': 5, 'estimator__n_estimators': 50}
  * Noise reduction
    * SibSp and Parch features created noise where feature importance in many cases was < 0.001%
    * Reducing less relevant hot encoded values into a single one improved accuracy by 2% on test set