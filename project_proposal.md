# Match prediction enhancement for dating applications

### Project proposal for the MALIS course @EURECOM

Marco Cavenati

Mayank Narang

Ilaria Pilo

## Motivation

With the pandemic the usage of dating applications has seen a rapid increase. However, users often struggle to find like-minded people with whom to build a durable and successful relationship.

To overcome these limitations, we aim to develop a machine learning model that allows the system to propose higher quality matches between users in less time.

Such model will be trained starting from data collected during a Speed Dating experiment where different questions were posed to the participants. Those questions are easily replicable in any dating application during the user registration process.

## Methodology and experiments

Our task is a binary classification problem, with the goal to predict the outcome of a match (successful/not successful).

Therefore, after a data preprocessing phase, we will split the data in three subset for training, validation and testing. We plan to use different machine learning techniques seen during the lectures, such as Logistic Regression, Random Forest and Support Vector Machine. 

To evaluate their performance we will evaluate them on the validation set comparing their accuracy, as well as other metrics that we will see in the future lectures.

Finally, after comparing their perfomance, we will select the most promising one, and we will apply the model on the test set.

The data will be taken from [Speed Dating | Kaggle](https://www.kaggle.com/datasets/whenamancodes/speed-dating)
