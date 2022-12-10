# Match prediction enhancement for dating applications

## Project update

_Authors_
- Marco Cavenati
- Mayank Narang
- Ilaria Pilo

## Motivation
With the pandemic, the usage of dating applications has seen a rapid increase. However, users often struggle to find like-minded people with whom to build a durable and successful relationship.

To overcome these limitations, we aim to develop a machine learning model that allows the system to propose higher quality matches between users in less time.

Our model will be trained starting from data collected during a speed dating experiment. In such an experiment, participants were posed different questions about themselves and their ideal partner, with the goal of studying dating behaviour. We believe that the same questions could be easily replicable in any dating application during the user's registration process, and used as input to predict whether a given match would be successful or not.

In this way, the system can propose to the user people with a successfully predicted match.

## Dataset overview
Our task is a binary classification problem, with the goal of predicting the outcome of a match (successful or not successful).

We are using the [Speed Dating | Kaggle](https://www.kaggle.com/datasets/whenamancodes/speed-dating) dataset, which stores the results of a series of speed dating events. The dataset contains the information collected before, during and after the event, for a total of __195 columns__. The total number of recorded interactions is __8378__. The dataset is unbalanced — 7k "No match" records against 1.4k "match" records — and it contains mainly categorical attributes.

Alongside with the dataset, we have at our disposal the original survey given to the people who participated in the event, where we can retrieve the semantics of the columns present in the dataset.

### Cleaning the dataset
The two biggest and time consuming issues faced with the dataset were:
1. Analyse and understand the semantics of the couluns to understand if the same feature could be retreived in the dating app context in the registration phase.
2. Manage the inconsistency between the survey methodologies used in the different waves of the event.

After the selection process only 57 columns are left, the detailed reasons for the removal of each column are explained in the _Dataset_analysis_ notebook.

To tackle the second problem, we remove from the dataset the data collected with different criterias, since it hasn't been possible to map that data into a form consistent with the rest of the data. Also, the records containing `NA` values has been dropped.

After completing the cleaning procedure, we obtain 3982 "No match" samples and 736 "match" samples.

![Count of 'match' records](/images/count_of_match.png "Count of 'match' records")

### Reshaping the dataset
Then, we reshape the data into a suitable form for the ML algorithms. We also tried to minimize the memory requirements by choosing optimal datatypes.


...

### Applying one hot encoding
Since the dataset contains a vast majority of categorical features, we have to choose the proper encoding for each of them. In particular, we observe two different cases:
1. The categorical feature represents a scaling. For example, __date__ can take the following values:
    | Value | Meaning |
    | :-----: | ------- |
    | 1 | Several times a week |
	| 2 | Twice a week |
	| 3 | Once a week |
	| 4 | Twice a month |
	| 5 | Once a month |
	| 6 | Several times a year |
	| 7 | Almost never |
2. The categorical feature is simply a set of values. For example, __goal__ can take the following values:
    | Value | Meaning |
    | :-----: | ------- |
    | 1 | Seemed like a fun night out |
	| 2 | To meet new people |
	| 3 | To get a date |
	| 4 | Looking for a serious relationship |
	| 5 | To say I did it |
	| 6 | Other |
Clearly, the two cases need to be addressed separately. While a standard encoding is appropriate to represent case (1), we want to apply one hot encoding to case (2), so that our models do not assume any scale of importance among different values.

Overall, we apply one hot encoding to the features `field_cd`, `race`, `goal`, `career_c` for both `_x` and `_y`.

After this final step, we obtain a total of 187 features (excluding the label `match`).

## Methodology
- k-fold
- grid search
- interactions

TODO
### The imbalance problem
- using stratified split
- using oversampling
- using appropriate metrics 

TODO
### FIghting the curse of dimensionality
- using appropriate algorithms (and bad ones to see if they are acutally bad)
- using PCA

TODO

## Models / Methodology
FIXME e che ne so?

## Next steps
- cost-based training
- more severe feature selection
- bias-value decomposition to see why the models are bad
- try more interactions
- try more algorithms

## Contribution

## References
