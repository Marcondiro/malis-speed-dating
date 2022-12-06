##########################################
#                                        #
#           DATA PREPROCESSING           #
#                                        #
##########################################

import pandas as pd

def preprocessing(dataset_file):
    """
    Prepares the dataset, applying different kinds of data preprocessing.

    
    Parameters
    ----------
    dataset_file: str
        The name of the file containing the dataset (.pkl)




    # FIXME - just some template to follow for later
    print_cols : bool, optional
        A flag used to print the columns to the console (default is False)

    Returns
    -------
    list
        a list of strings representing the header columns



    """
    # first, load the dataframe
    df = pd.read_pickle(dataset_file)

    # We apply OneHotEncoding
    # Notice that the only attributes that require OneHotEncoding are those which are:
    #   - categorical AND
    #   - their value does not match an actual scale of importance
    # field_cd, race, goal, career_c
    df = pd.get_dummies(data=df, columns=['field_cd_x', 'race_x', 'goal_x', 'career_c_x', 'field_cd_y', 'race_y', 'goal_y', 'career_c_y'])









preprocessing('data/data.pkl')

