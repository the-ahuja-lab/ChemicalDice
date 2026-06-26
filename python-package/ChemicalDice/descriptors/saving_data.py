import os
from sklearn.model_selection import train_test_split

def save_train_test_data_n_fold(dataframes, prediction_labels, train_index, test_index, output_dir ):
    """

    This function saves the training and testing data for n-fold cross validation.

    Parameters:
    dataframes (dict): A dictionary where the key is the name of the dataframe and the value is the dataframe itself.
    prediction_labels (Series): The labels for prediction.
    train_index (list): The indices for the training data.
    test_index (list): The indices for the testing data.
    output_dir (str): The directory where the training and testing data will be saved.

    Returns:
    train_dataframes (dict): A dictionary where the key is the name of the dataframe and the value is the training data.
    test_dataframes (dict): A dictionary where the key is the name of the dataframe and the value is the testing data.
    y_train (Series): The training labels for prediction.
    y_test (Series): The testing labels for prediction.
    
    """
    train_dataframes = {}
    test_dataframes = {}
    # output_dir_train = os.path.join(output_dir,"training_data/")
    # if not os.path.exists(output_dir_train):
    #     # Create the directory
    #     os.makedirs(output_dir_train)
    #     print(f"The directory {output_dir_train} was created.")
    # else:
    #     # Print a message
    #     print(f"The directory {output_dir_train} already exists.")
    
    # output_dir_test = os.path.join(output_dir,"testing_data/")
    # if not os.path.exists(output_dir_test):
    #     # Create the directory
    #     os.makedirs(output_dir_test)
    #     print(f"The directory {output_dir_test} was created.")
    # else:
    #     # Print a message
    #     print(f"The directory {output_dir_test} already exists.")
    for name, df in dataframes.items():
        X=df
        y=prediction_labels
        X_train = X.iloc[train_index,:]
        y_train = y[train_index]
        X_test = X.iloc[test_index,:]
        y_test = y[test_index]
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #print(X_train)
        train_dataframes.update({name:X_train})
        test_dataframes.update({name:X_test})
        #X_train2['prediction_label'] = y_train
        #X_test2['prediction_label'] = y_test
        #X_train.to_csv(os.path.join(output_dir,"training_data/", name+".csv"))
        #X_test.to_csv(os.path.join(output_dir,"testing_data/", name+".csv"))
        #print(X_train.shape)
        #print(X_test.shape)
    return train_dataframes, test_dataframes, y_train, y_test


def save_train_test_data_s_fold(dataframes, prediction_labels, train_index, val_index, test_index):
    """

    This function saves the training and testing data for n-fold cross validation.

    Parameters:
    dataframes (dict): A dictionary where the key is the name of the dataframe and the value is the dataframe itself.
    prediction_labels (Series): The labels for prediction.
    train_index (list): The indices for the training data.
    test_index (list): The indices for the testing data.
    output_dir (str): The directory where the training and testing data will be saved.

    Returns:
    train_dataframes (dict): A dictionary where the key is the name of the dataframe and the value is the training data.
    test_dataframes (dict): A dictionary where the key is the name of the dataframe and the value is the testing data.
    y_train (Series): The training labels for prediction.
    y_test (Series): The testing labels for prediction.
    
    """
    train_dataframes = {}
    test_dataframes = {}
    val_dataframes = {}
    # output_dir_train = os.path.join(output_dir,"training_data/")
    # if not os.path.exists(output_dir_train):
    #     # Create the directory
    #     os.makedirs(output_dir_train)
    #     print(f"The directory {output_dir_train} was created.")
    # else:
    #     # Print a message
    #     print(f"The directory {output_dir_train} already exists.")
    
    # output_dir_test = os.path.join(output_dir,"testing_data/")
    # if not os.path.exists(output_dir_test):
    #     # Create the directory
    #     os.makedirs(output_dir_test)
    #     print(f"The directory {output_dir_test} was created.")
    # else:
    #     # Print a message
    #     print(f"The directory {output_dir_test} already exists.")
    for name, df in dataframes.items():

        X=df
        y=prediction_labels
        X_train, X_val, X_test = X.loc[train_index], X.loc[val_index], X.loc[test_index]
        y_train, y_val, y_test = y[train_index], y[val_index], y[test_index]
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #print(X_train)
        train_dataframes.update({name:X_train})
        val_dataframes.update({name:X_val})
        test_dataframes.update({name:X_test})
        #X_train2['prediction_label'] = y_train
        #X_test2['prediction_label'] = y_test
        #X_train.to_csv(os.path.join(output_dir,"training_data/", name+".csv"))
        #X_test.to_csv(os.path.join(output_dir,"testing_data/", name+".csv"))
        #print(X_train.shape)
        #print(X_test.shape)
    return train_dataframes, val_dataframes, test_dataframes, y_train, y_val,y_test


def save_train_test_data(dataframes, prediction_labels, output_dir):
    """

    This function saves the training and testing data.

    Parameters:
    dataframes (dict): A dictionary where the key is the name of the dataframe and the value is the dataframe itself.
    prediction_labels (Series): The labels for prediction.
    output_dir (str): The directory where the training and testing data will be saved.

    Returns:
    train_dataframes (dict): A dictionary where the key is the name of the dataframe and the value is the training data.
    test_dataframes (dict): A dictionary where the key is the name of the dataframe and the value is the testing data.
    y_train (Series): The training labels for prediction.
    y_test (Series): The testing labels for prediction.

    """
    train_dataframes = {}
    test_dataframes = {}
    # output_dir_train = os.path.join(output_dir,"training_data/")
    # if not os.path.exists(output_dir_train):
    #     # Create the directory
    #     os.makedirs(output_dir_train)
    #     print(f"The directory {output_dir_train} was created.")
    # else:
    #     # Print a message
    #     print(f"The directory {output_dir_train} already exists.")
    
    # output_dir_test = os.path.join(output_dir,"testing_data/")
    # if not os.path.exists(output_dir_test):
    #     # Create the directory
    #     os.makedirs(output_dir_test)
    #     print(f"The directory {output_dir_test} was created.")
    # else:
    #     # Print a message
    #     print(f"The directory {output_dir_test} already exists.")
    for name, df in dataframes.items():
        X=df
        y=prediction_labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #print(X_train)
        train_dataframes.update({name:X_train})
        test_dataframes.update({name:X_test})
        #X_train2['prediction_label'] = y_train
        #X_test2['prediction_label'] = y_test
        #X_train.to_csv(os.path.join(output_dir,"training_data/", name+".csv"))
        #X_test.to_csv(os.path.join(output_dir,"testing_data/", name+".csv"))
    return train_dataframes, test_dataframes, y_train, y_test
