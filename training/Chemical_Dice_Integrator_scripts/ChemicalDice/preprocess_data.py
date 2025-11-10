import pandas as pd
from typing import Dict
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
from scipy import stats
import os

def clear_and_process_data(csv_files: Dict[str, str], id_column: str) -> Dict[str, pd.DataFrame]:
    """

    Read data from CSV files, set the specified ID column as the row index, and remove it.
    Also, change the column name of the prediction label.

    :param csv_files: Dictionary with dataset names as keys and CSV file paths as values.
    :type csv_files: dict
    :param id_column: The column name containing IDs to set as the row index.
    :type id_column: str
    :param prediction_label_column: The current column name containing prediction labels.
    :type prediction_label_column: str

    :return: Dictionary with dataset names as keys and processed DataFrames as values.
    :rtype: dict[str, pd.DataFrame]

    """
    dataset_dict = {}

    for dataset_name, file_path in csv_files.items():
        try:
            # Read CSV file into a Pandas DataFrame
            if dataset_name.lower() == "imagemol":
                df = pd.read_csv(file_path,sep="\t")
            else:
                df = pd.read_csv(file_path)

            # Set specified ID column as the index
            df.set_index(id_column, inplace=True)

            df.index.name = "id"

            if 'SMILES' in df.columns:
                df = df.drop('SMILES', axis=1)

            
            # Store the DataFrame in the dataset dictionary
            dataset_dict[dataset_name] = df
            print(f"Successfully loaded, processed for '{dataset_name}'.")
        except Exception as e:
            print(f"Error loading data for '{dataset_name}': {e}")
            raise
    return dataset_dict



#from tabulate import tabulate
def normalize_to_constant_sum(df, constant_sum=1, axis=1):
    """

    Normalizes the values in a dataframe to a constant sum along the specified axis.

    This function divides each value in the dataframe by the sum of all values in its row or column (depending on the specified axis), and then multiplies the result by a constant sum. The result is a dataframe where the sum of all values in each row or column equals the constant sum.

    :param df: The dataframe to normalize.
    :type df: pandas.DataFrame
    :param constant_sum: The constant sum to which the values should be normalized. Defaults to 1.
    :type constant_sum: float, optional
    :param axis: The axis along which to normalize the values. If 0, normalize along the columns. If 1, normalize along the rows. Defaults to 1.
    :type axis: int, optional
    :return: The normalized dataframe.
    :rtype: pandas.DataFrame
    :raises ValueError: If the specified axis is not 0 or 1.
    
    """
    if axis == 0:
        normalized_df = df.div(df.sum(axis=axis), axis=1) * constant_sum
    elif axis == 1:
        normalized_df = (df.T / df.sum(axis=axis)).T * constant_sum
    else:
        raise ValueError("Axis must be 0 or 1.")
    return normalized_df


def show_dataframe_info(dataframes: Dict[str, pd.DataFrame]):
    """

    Show the number of features and samples in each DataFrame and check for common rows based on row indices.

    :param dataframes: Dictionary with dataset names as keys and Pandas DataFrames as values.
    :type dataframes: dict

    :output: Prints a table summarizing the number of samples, features for each dataset and number of missing values.
            Prints the number of common rows across datasets based on row indices.


    """

    common_indices = set()

    headers = ["Dataset", "Number of Samples", "Number of Features", "Number of missing Values"]
    rows = []

    for dataset_name, df in dataframes.items():
        try:
            # Get the row indices for the current DataFrame
            current_indices = set(df.index)

            # Store the common row indices
            if not common_indices:
                common_indices = current_indices
            else:
                common_indices = common_indices.intersection(current_indices)

            # Get the number of features and samples
            num_samples, num_features = df.shape

            # Get the number of missing values in the Dataframe
            missing_values = df.isnull().sum().sum()

            
            # Append information for the current dataset to rows
            rows.append([dataset_name, num_samples, num_features, missing_values])
        except Exception as e:
            print(f"Error processing data for '{dataset_name}': {str(e)}")

    # Create a Pandas DataFrame from the rows
    result_df = pd.DataFrame(rows, columns=headers)

    # Print the Pandas DataFrame
    print(result_df)

    # Print the number of common rows across datasets
    num_common_rows = len(common_indices)
    print(f"\nNumber of common rows across datasets: {num_common_rows}\n")

    # Issue a warning if the number of common rows is not the same across datasets
    if len(set(df.shape[0] for df in dataframes.values())) != 1:
        warnings.warn("Number of samples is different across datasets. "
                      "Consider checking and handling common rows.")


import pandas as pd

def check_missing_values(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """

    Checks for missing values in each dataframe in a dictionary of dataframes and returns a dataframe with the results.

    This function iterates over each dataframe in the provided dictionary. For each dataframe, it calculates the number and percentage of missing values in each column. It then creates a Pandas Series with the dataset name, column name, number and percentage of missing values. The results are stored in a list which is then converted into a dataframe.

    :param dataframes: A dictionary where the key is the name of the dataframe and the value is the dataframe itself.
    :type dataframes: Dict[str, pd.DataFrame]
    :return: A dataframe where each row corresponds to a dataframe from the input dictionary and each column corresponds to the number and percentage of missing values in each column of the respective dataframe.
    :rtype: pd.DataFrame

    """
    # Create an empty list to store the results
    results = []

    # Loop through the dataframes dictionary
    for dataset_name, df in dataframes.items():
        # Get the number of missing values in each column
        num_missing = df.isnull().sum()

        # Get the percentage of missing values in each column
        percent_missing = num_missing * 100 / len(df)

        # Create a Pandas Series with the dataset name, column name, number and percentage of missing values
        series = pd.Series([dataset_name] + list(num_missing) + list(percent_missing), index=["Dataset", "id", "Prediction Label"] + [f"{col}_missing" for col in df.columns])

        # Append the series to the results list
        results.append(series)

    # Create a Pandas DataFrame from the results list
    result_df = pd.DataFrame(results)

    # Return the result DataFrame
    return result_df



def remove_columns_by_missing_threshold(dataframes, threshold):
    """

    Removes columns from each dataframe in a dictionary of dataframes based on a missing value threshold.

    This function iterates over each dataframe in the provided dictionary. For each dataframe, it calculates the minimum count of non-null values required based on the provided threshold. It then drops the columns which do not have the required number of non-null values.

    :param dataframes: A dictionary where the key is the name of the dataframe and the value is the dataframe itself.
    :type dataframes: dict of pandas.DataFrame
    :param threshold: The percentage of missing values allowed in the column. If a column has more missing values than this threshold, it will be dropped.
    :type threshold: float
    :return: The original dictionary of dataframes, but with columns dropped based on the missing value threshold.
    :rtype: dict of pandas.DataFrame
    
    """
    for name, df in dataframes.items():
        # Calculate the minimum count of non-null values required 
        min_count = int(((100 - threshold) / 100) * df.shape[0] + 1)
        # Drop columns with insufficient non-null values
        df_cleaned = df.dropna(axis=1, thresh=min_count)
        dataframes[name] = df_cleaned
    return dataframes



def plot_missing_values(df_list: List[pd.DataFrame], dataset_names: List[str], save_dir=None):
    """

    Plot stacked barcharts of missing vs non-missing values in the columns of each dataframe in a list.

    :param df_list: A list of Pandas DataFrames with missing values.
    :type df_list: List[pd.DataFrame]
    :param dataset_names: Names of the datasets corresponding to each dataframe in df_list.
    :type dataset_names: List[str]
    :param save_dir: Directory to save the plots. If None, the plots are shown. If provided, the plots are saved to this directory.
    :type save_dir: str, optional

    :output: Matplotlib figures with the barcharts.


    """
    for df, dataset_name in zip(df_list, dataset_names):
        # Get the number of missing and non-missing values in each column
        num_missing = df.isnull().sum()
        num_non_missing = df.notnull().sum()

        # Create a figure and an axis
        fig, ax = plt.subplots()

        # Plot the non-missing values as the bottom bars
        ax.bar(df.columns, num_non_missing, label="Non Missing", align='center', width=1)

        # Plot the missing values as the top bars
        ax.bar(df.columns, num_missing, bottom=num_non_missing, label="Missing")
        ax.set_xticklabels([])
        # Add some labels and a title
        ax.set_ylabel("Number of values")
        ax.set_title("Missing vs Non-Missing Values in " + dataset_name)
        ax.legend()

        # Check if running in Jupyter notebook
        if save_dir is not None:
            if not os.path.exists(save_dir): 
                # if the dir directory is not present then create it. 
                os.makedirs(save_dir) 
            plt.savefig(f'{save_dir}/{dataset_name}_missing_values_plot.png')
        else:
            plt.show()


def barplot_normality_check(df_list, dataset_names, alpha=0.05, save_dir=None):
    """

    Creates a barplot of p-values for normality test in Python for each dataframe in a list.

    :param df_list: The list of dataframes to be tested for normality.
    :type df_list: list of pandas.DataFrame
    :param dataset_names: The names of the datasets corresponding to each dataframe in df_list.
    :type dataset_names: list of str
    :param alpha: The significance level for the normality test. The default is 0.05.
    :type alpha: float, optional
    :param save_dir: The directory to save the plots. If None, the plots are shown. If provided, the plots are saved to this directory.
    :type save_dir: str, optional

    :output: Matplotlib figures with the barcharts.

    """
    # loop through each dataframe and dataset name in the list
    for df, dataset_name in zip(df_list, dataset_names):
        # initialize an empty list to store the p-values
        pvalues = []
        
        # loop through each column of the dataframe
        for col in df.columns:
            # perform the D'Agostino's K^2^ test on the column
            k2, p = stats.normaltest(df[col])
            # append the p-value to the list
            pvalues.append(p)
        
        # create a figure and an axis
        fig, ax = plt.subplots()
        
        # create a barplot of the p-values
        ax.bar(df.columns, pvalues, color='blue')
        # add a horizontal line at the significance level
        ax.axhline(y=alpha, color='red', linestyle='--')
        # add labels and a title
        ax.set_xlabel('Columns')
        ax.set_ylabel('P-values')
        ax.set_title('Barplot of P-values for Normality Test in ' + dataset_name)
        
        # check if a save directory is provided
        if save_dir is not None:
            # create the directory if it does not exist
            if not os.path.exists(save_dir): 
                os.makedirs(save_dir) 
            # save the plot to the directory
            plt.savefig(f'{save_dir}/{dataset_name}_normality_plot.png')
        else:
            # show the plot
            plt.show()
