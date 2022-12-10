# Match prediction enhancement for dating applications

## Project update

_Authors_
- Marco Cavenati
- Mayank Narang
- Ilaria Pilo

## Motivation
copy-paste dalla proposal

## Dataset overview
The chosen dataset contains the results of a series of speed dating events. The dataset contains a lot of informations collected before, during and after the event, for a total of __195 columns__. The total number of recorded interactions is __8378__.

Alongside with the dataset we have at our disposal the original survey given to the people who participated in the event, where we can retrieve the semantics of the columns present in the dataset.

The two biggest and time consuming issues faced with the dataset were:
1. Analyse and understand the semantics of the couluns to understand if the same feature could be retreived in the dating app context in the registration phase.
2. Manage the inconsistency between the survey methodologies used in the different waves of the event.

After the selection process only 57 columns are left, the detailed reasons for the removal of each column are explained in the _Dataset_analysis_ notebook.

To tackle the second problem, we decided to remove from the dataset the data collected with different criterias. It hasn't been possible to map that data into a form consistent with the rest of the data. Also the records containing `NA` values has been dropped.

Then, we reshaped the data into a suitable form for the ML algorithms. We also tried to minimize the memory requirements by choosing optimal datatypes.

## Models / Methodology
FIXME e che ne so?

## Contribution

## References
