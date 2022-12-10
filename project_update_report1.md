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

After the selection process only 57 columns are left, the detailed reasons for the removal of each column are explained in the _Dataset_analysis_ notebook. Here we briefly describe the features that we have selected:
- __iid__: id of the person participating in the event
- __gender__: gender of the person
- __pid__: partner id
- __match__: match (label of the model)
- __age__
- __field_cd__: Field of study (coded)
- __race__ (coded)
- __imprace__: How important is it for the person (on a scale of 1-10) that a person she dates is of the same racial/ethnic background.
- __goal__: primary goal in participating in the event (coded)
- __date__: How frequently the person goes on dates (coded)
- __go_out__: How often the person goes out (not necessarily on dates) (coded)
- __career_c__: Intended career (coded)
- __sports__: interest in Playing sports/ athletics on a scale of 1-10
- __tvsports__: interest in Watching sports
- __excersice__: interest in Body building/exercising
- __dining__: interest in Dining out
- __museums__: interest in Museums/galleries
- __art__: interest in Art
- __hiking__: interest in Hiking/camping
- __gaming__: interest in Gaming
- __clubbing__: interest in Dancing/clubbing
- __reading__: interest in Reading
- __tv__: interest in Watching TV
- __theater__: interest in Theater
- __movies__: interest in Movies
- __concerts__: interest in Going to concerts
- __music__: interest in Music
- __shopping__: interest in Shopping
- __yoga__: interest in Yoga/meditation

100 points to be distributed among the following attributes, more points to those attributes that are more important in a potential date and fewer points to those attributes that are less important in a potential date. Total points must equal 100.
- __attr1_1__: Attractive
- __sinc1_1__:  Sincere
- __intel1_1__: Intelligent
- __fun1_1__: Fun
- __amb1_1__: Ambitious
- __shar1_1__: Has shared interests/hobbies

What the person thinks MOST of men/women look for in the opposite sex.
100 points to be distributed among the following attributes, more points to those attributes that are more important in a potential date and fewer points to those attributes that are less important in a potential date. Total points must equal 100.
- __attr4_1__: Attractive
- __sinc4_1__: Sincere
- __intel4_1__: Intelligent
- __fun4_1__: Fun
- __amb4_1__: Ambitious
- __shar4_1__: Shared Interests/Hobbies

What the person thinks the opposite sex looks for in a date.
100 points distributed among the following attributes, more points more important. Total points must equal 100.
- __attr2_1__: Attractive
- __sinc2_1__: Sincere
- __int2_1,__: Intelligent
- __fun2_1__: Fun
- __amb2_1__: Ambitious
- __shar2_1__: Has shared interests/hobbies

How the person rates herself. Scale of 1-10
- __attr3_1__: Attractive
- __sinc3_1__: Sincere
- __int3_1__: Intelligent
- __fun3_1__: Fun
- __amb3_1__: Ambitious

Person's opinion on how she is perceived by others. Scale of 1-10
- __attr5_1__: Attractive
- __sinc5_1__: Sincere
- __int5_1__: Intelligent
- __fun5_1__: Fun
- __amb5_1__: Ambitious

To tackle the second problem, we remove from the dataset the data collected with different criterias, since it hasn't been possible to map that data into a form consistent with the rest of the data. Also, the records containing `NA` values has been dropped.

After completing the cleaning procedure, we obtain 3982 "No match" samples and 736 "match" samples.

![Count of 'match' records](/images/count_of_match.png "Count of 'match' records")

### Reshaping the dataset
Then, we reshape the data into a suitable form for the ML algorithms. We also tried to minimize the memory requirements by choosing optimal datatypes.
In particular, we use the `iid` and `pid` fields to join the dataframe with itself, in this way in each record there are all the answers of the person and all the answers of the partner.
Then we drop the `iid` and `pid` fields .

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
In order to face the classification task, our choices in terms of methodology have so far been the following: 
- The k-Nearest Neighbors algorithm  
- Logistic Regression  
- Support Vector Machines  

To avoid problems such as overfitting in the training of these algorithms, we decided to employ k-fold cross validation starting from an experimental amount of two folds. Moreover, we applied grid search for the optimization of our model's hyperparameters.  

Given the dataset's heavy unbalance, we opted against accuracy as a metric to evaluate the chosen models. Instead, it seemed preferable to pick alternatives such as:  
- Balanced Accuracy  
- F1-Score  
- Recall  

Furthermore, we attempted to manage such unbalance by making use of a Stratisfied Split for the generation of the training and testing sets.  
This problem, together with the very high number of features in our dataset, caused us to observe some bad performances in the testing of the three algorithms (e.g., in the convergence of the Logistic Regression function). We plan to face these issues by further investigation the relations among some features as well as in the ways indicated in the next section.

## Next steps
- cost-based training
- more severe feature selection
- bias-value decomposition to see why the models are bad
- try more interactions
- try more algorithms

## Contribution
- Marco Cavenati: dataset analysis and features reduction.
- Mayank Narang: implementation of the algorithms.
- Ilaria Pilo: preparation of the data functions (i.e., train/validation/test split), model evaluation.

## References
