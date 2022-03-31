## Model Details
We used a random forest classifier to help predict if someone's income exceed or subceed 50k per year.
The training is done based on different features.
## Intended Use
Help predict the income of a person based on different features.
Might be used to figure out underpayed jobs, race and gender.
Might be used to understand the features that have direct impact on the salary.
## Training Data
The census Dataset was downloaded from this website: https://archive.ics.uci.edu/ml/datasets/census+income.
We used 80% of the data for training.
## Evaluation Data
Same as the training data the validation data presents the rest of the 20% of the data downloaded from the same website: https://archive.ics.uci.edu/ml/datasets/census+income.
## Metrics
The following are the global results:
To check for possible bias please check the results per category in the log folder.

precision = 0.7584846093133386, recall = 0.6236210253082414, fbeta = 0.6844729344729346 
## Ethical Considerations
This data is opensouced for educational purposes.
## Caveats and Recommendations
Model can be be improved by used a different machine learning technique.
Model can be improved by performing a K-fold cross validation.
Some features can be ommited since there is redundancy, such as education, which is in a direct relationship with education-num.
