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
Citation Policy:

If you publish material based on databases obtained from this repository, then, in your acknowledgements, please note the assistance you received by using this repository. This will help others to obtain the same data sets and replicate your experiments. We suggest the following pseudo-APA reference format for referring to this repository:

    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

Here is a BiBTeX citation as well:

    @misc{Dua:2019 ,
    author = "Dua, Dheeru and Graff, Casey",
    year = "2017",
    title = "{UCI} Machine Learning Repository",
    url = "http://archive.ics.uci.edu/ml",
    institution = "University of California, Irvine, School of Information and Computer Sciences" }

A few data sets have additional citation requests. These requests can be found on the bottom of each data set's web page. 
## Caveats and Recommendations
Model can be be improved by used a different machine learning technique.
Model can be improved by performing a K-fold cross validation.
Some features can be ommited since there is redundancy, such as education, which is in a direct relationship with education-num.
