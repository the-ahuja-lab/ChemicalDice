import os
from sklearn.impute import KNNImputer, SimpleImputer
# Import the necessary modules

from sklearn.preprocessing import normalize

from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.model_selection import KFold
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import PLSRegression, CCA, PLSCanonical
from sklearn.decomposition import IncrementalPCA

from itertools import combinations
from sklearn.cross_decomposition import CCA

from sklearn.decomposition import KernelPCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.manifold import Isomap, TSNE, SpectralEmbedding, LocallyLinearEmbedding

from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import itertools

from functools import reduce
import pandas as pd
from ChemicalDice.plot_data import *
from ChemicalDice.preprocess_data import *
from ChemicalDice.saving_data import *
from ChemicalDice.analyse_data import *

import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib.colors import to_rgba

# Import the necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
from sklearn.model_selection import train_test_split

#########################

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List
import os
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


class baseData:
    """
    Class for handling data and performing fusion tasks.

    Attributes:
        - data_paths (list): List of file paths containing the data.
        - id_column (str): Name of the column representing unique identifiers.
        - dataframes (dict): Dictionary containing processed dataframes.
        - prediction_label: Placeholder for prediction labels.
        - fusedData: Placeholder for fused data.
        - pls_model: Placeholder for the PLS model.
        - Accuracy_metrics: Placeholder for accuracy metrics.
        - train_dataframes: Placeholder for training dataframes.
        - test_dataframes: Placeholder for testing dataframes.
        - train_label: Placeholder for training labels.
        - test_label: Placeholder for testing labels.

    """
    def __init__(self, data_paths, id_column = "ID", prediction_label_column = "prediction_label"):

        """

        Initialize fusionData object with provided data paths, ID column name,
        and prediction label column name.

        Args:
            data_paths (list): List of file paths containing the data.
            id_column (str, optional): Name of the column representing unique identifiers.
                Defaults to "ID".
            prediction_label_column (str, optional): Name of the column containing prediction labels.
                Defaults to "prediction_label".

        """
        self.data_paths = data_paths
        loaded_data_dict = clear_and_process_data(data_paths, id_column, prediction_label_column)
        #check_missing_values(loaded_data_dict)
        self.prediction_label = None
        self.dataframes = loaded_data_dict


        self.dataframes_transformed = False
        self.olddataframes = None
        
        self.top_features = None
        self.fusedData = None
        self.scaling_done = False
        
        self.pls_model = None
        self.Accuracy_metrics=None
        self.mean_accuracy_metrics =None
        self.train_dataframes = None
        self.test_dataframes = None
        self.train_label = None
        self.test_label = None

    
    
    def ShowMissingValues(self):
        """
        Prints missing values inside dataframes in fusionData object.
        """
        dataframes = self.dataframes
        # Iterate through the dictionary and print missing DataFrame 
        for name, df in dataframes.items():
            missing_values = df.isnull().sum().sum()
            print(f"Dataframe name: {name}\nMissing values: {missing_values}\n")

    def show_info(self):
        """
        Summarize dataframes inside the fusionData object.
        """
        show_dataframe_info(self.dataframes)
    
    def plot_info(self, about, save_dir=None):
        """

        Generate plots for visualizing missing values and to check for normality of data.

        This method supports two types of plots:
           - 'missing_values': will generate a plot showing the missing values in the dataframes.
           - 'normality': will generate a bar plot to check the normality of the data in the dataframes.

        :param about: The topic for the plot. Must be either 'missing_values' or 'normality'.
        :type about: str

        """
        if about == "missing_values":
            plot_missing_values(list(self.dataframes.values()),
                                list(self.dataframes.keys()),save_dir)
        elif about == "normality":
            barplot_normality_check(list(self.dataframes.values()),
                                list(self.dataframes.keys()),save_dir)
        else:
            raise ValueError("Please select a valid plot type: 'missing_values' or 'normality'")

    def keep_common_samples(self):
        """
        Keep only the common samples or rows using the id in dataframes.

        """
        # Create an empty set to store the common indices
        common_indices = set()
        dataframes = self.dataframes
        # Loop through the df_dict
        for name, df in dataframes.items():
            # Get the index of the current dataframe and convert it to a set
            index = set(df.index)

            # Update the common indices set with the intersection of the current index
            if not common_indices:
                common_indices = index
            else:
                common_indices = common_indices.intersection(index)

        # Create an empty dictionary to store the filtered dataframes
        filtered_df_dict = {}

        # Loop through the df_dict again
        for name, df in dataframes.items():
            # Filter the dataframe by the common indices and store it in the filtered_df_dict
            filtered_df = df.loc[list(common_indices)]
            self.prediction_label =filtered_df['prediction_label']
            filtered_df.drop('prediction_label',axis=1, inplace=True)
            
            filtered_df_dict[name] = filtered_df
            
        self.dataframes = filtered_df_dict


        
    def remove_empty_features(self,threshold=100):
        
        """
        Remove columns with more than a certain percentage of missing values from dataframes.

        :param threshold: The percentage threshold of missing values to drop a column. It should be between 0 and 100.
        :type threshold: float

        """
        dataframes = self.dataframes
        for name, df in dataframes.items():
            # Calculate the minimum count of non-null values required 
            min_count = int(((100 - threshold) / 100) * df.shape[0] + 1)
            # Drop columns with insufficient non-null values
            df_cleaned = df.dropna(axis=1, thresh=min_count)
            dataframes[name] = df_cleaned
        self.dataframes = dataframes

    def ImputeData(self, method="knn"):
        """

        Impute missing data in the dataframes.

        This method supports five types of imputation methods: 
           - 'knn' will use the K-Nearest Neighbors approach to impute missing values.
           - 'mean' will use the mean of each column to fill missing values.
           - 'most_frequent' will use the mode of each column to fill missing values.
           - 'median' will use the median of each column to fill missing values.
           - 'interpolate' will use the Interpolation method to fill missing values.

        :param method: The imputation method to use. Options are "knn", "mean", "mode", "median", and "interpolate".
        :type method: str, optional

        """
       
        dataframes = self.dataframes
        for name, df in dataframes.items():

            missing_values = df.isnull().sum().sum()
            if missing_values > 0:
                if method == "knn":
                    imputer = KNNImputer(n_neighbors=2)
                    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
                elif method in ["mean", "most_frequent", "median"]:
                    imputer = SimpleImputer(strategy=method)
                    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
                elif method == "interpolate":
                    df.interpolate(method='linear', inplace=True)
                    df_imputed = df
                else:
                    raise ValueError("Please select a valid imputation method: 'knn', 'mean', 'most_frequent', 'median', or 'interpolate'")
                dataframes[name] = df_imputed
        self.dataframes = dataframes
        print("Imputation Done")


    def scale_data(self, scaling_type,  **kwargs):
        """

        Scale the dataFrame based on the specified scale type.

        The scaling methods are as follows:
            - 'standardize' Applies the standard scaler method to each column of the dataframe. This method transforms the data to have a mean of zero and a standard deviation of one. It is useful for reducing the effect of outliers and making the data more normally distributed.
            - 'minmax' Applies the min-max scaler method to each column of the dataframe. This method transforms the data to have a minimum value of zero and a maximum value of one. It is useful for making the data more comparable and preserving the original distribution.
            - 'robust' Applies the robust scaler method to each column of the dataframe. This method transforms the data using the median and the interquartile range. It is useful for reducing the effect of outliers and making the data more robust to noise.
            - 'pareto' Applies the pareto scaling method to each column of the dataframe. This method divides each element by the square root of the standard deviation of the column. It is useful for making the data more homogeneous and reducing the effect of skewness.
        
        :param scaling_type: The type of scaling to be applied. It can be one of these: 'standardize', 'minmax', 'robust', or 'pareto'.
        :type scaling_type: str
        :param kwargs: Additional parameters for specific scaling methods.
        :type kwargs: dict


        """

        dataframes = self.dataframes


        # Loop through the df_dict
        for dataset_name, df in dataframes.items():

            # Apply the scaling type to the dataframe
            if scaling_type == 'standardize':
                scaler = StandardScaler()
                scaled_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
            elif scaling_type == 'minmax':
                scaler = MinMaxScaler()
                scaled_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
            elif scaling_type == 'robust':
                scaler = RobustScaler()
                scaled_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
            elif scaling_type == 'pareto':
                scaled_df = df / np.sqrt(df.std())
            else:
                raise ValueError(f"""Unsupported scaling type: {scaling_type} 
                                    possible values are 'standardize', 'minmax', 'robust', or 'pareto' """)
            # Return the scaled dataframe
            dataframes[dataset_name] = scaled_df
        self.dataframes = dataframes

    def normalize_data(self,normalization_type, **kwargs):
        """

        This method supports four types of normalization methods:
            - 'constant sum' normalization. The default sum is 1, which can  specified using the 'sum' keyword argument in kwargs. It is a method used to normalize data such that the sum of values for each observation remains constant. It ensures that the relative contributions of individual features to the total sum are compared rather than the absolute values. This normalization technique is beneficial for comparing samples with different total magnitudes but similar distributions. Mathematically, each observation is normalized by dividing it by the sum of its values, and then multiplying by a constant factor to achieve the desired sum.
            - 'L1' normalization (Lasso Norm or Manhattan Norm): Also known as Lasso Norm or Manhattan Norm. Each observation vector is rescaled by dividing each element by the L1-norm of the vector.. L1-norm of a vector is the sum of the absolute values of its components. Mathematically, for a vector x, L1 normalization is given by: L1-Norm = ∑|x_i|. After L1 normalization, the sum of the absolute values of the elements in each vector becomes 1. Widely used in machine learning tasks such as Lasso regression to encourage sparsity in the solution.
            - 'L2' normalization (Ridge Norm or Euclidean Norm): Also known as Ridge Norm or Euclidean Norm. Each observation vector is rescaled by dividing each element by the L2-norm of the vector. L2-norm of a vector is the square root of the sum of the squares of its components. Mathematically, for a vector x, L2 normalization is given by: L2-Norm = √∑x_i^2. After L2 normalization, the Euclidean distance (or the magnitude) of each vector becomes 1. Widely used in various machine learning algorithms such as logistic regression, support vector machines, and neural networks.
            - 'max' normalization (Maximum Normalization): Scales each feature in the dataset by dividing it by the maximum absolute value of that feature across all observations. Ensures that each feature's values are within the range [-1, 1] or [0, 1] depending on whether negative values are present or not. Useful when the ranges of features in the dataset are significantly different, preventing certain features from dominating the learning process due to their larger scales. Commonly used in neural networks and deep learning models as part of the data preprocessing step.

        Normalize dataframes using different types of normalization.

        :param normalization_type: The type of normalization to apply. It can be one of these: 'constant_sum', 'L1', 'L2', or 'max'.        
        :type normalization_type: str
        :param kwargs: Additional arguments for some normalization types.
        :type kwargs: dict
        :raises ValueError: If the provided method is not 'constant_sum', 'L1' ,'L2' or 'max

        """


        # Create an empty dictionary to store the normalized dataframes
        normalized_df_dict = {}
        dataframes = self.dataframes

        # Loop through the df_dict
        for dataset_name, df in dataframes.items():

            # Apply the normalization type to the dataframe
            if normalization_type == 'constant_sum':
                constant_sum = kwargs.get('sum', 1)
                axis = kwargs.get('axis', 1)
                normalized_df = normalize_to_constant_sum(df, constant_sum=constant_sum, axis=axis)
            elif normalization_type == 'L1':
                normalized_df = df.apply(lambda x: x /np.linalg.norm(x,1))
            elif normalization_type == 'L2':
                normalized_df =  pd.DataFrame(normalize(df, norm='l1'), columns=df.columns,index = df.index)
            elif normalization_type == 'max':
                normalized_df = pd.DataFrame(normalize(df, norm='max'), index=df.index )
            else:
                raise ValueError(f"Unsupported normalization type: {normalization_type} \n possible values are 'constant_sum', 'L1' ,'L2' and 'max'   ")
            # Store the normalized dataframe in the normalized_df_dict
            dataframes[dataset_name] = normalized_df
        self.dataframes = dataframes

    def transform_data(self, transformation_type, **kwargs):
        """
        
        The transformation methods are as follows:
            - 'cubicroot': Applies the cube root function to each element of the dataframe.
            - 'log10': Applies the base 10 logarithm function to each element of the dataframe.
            - 'log': Applies the natural logarithm function to each element of the dataframe.
            - 'log2': Applies the base 2 logarithm function to each element of the dataframe.
            - 'sqrt': Applies the square root function to each element of the dataframe.
            - 'powertransformer': Applies the power transformer method to each column of the dataframe. This method transforms the data to make it more Gaussian-like. It supports two methods: 'yeo-johnson' and 'box-cox'. The default method is 'yeo-johnson', which can handle both positive and negative values. The 'box-cox' method can only handle positive values. The method can be specified using the 'method' keyword argument in kwargs.
            - 'quantiletransformer': Applies the quantile transformer method to each column of the dataframe. This method transforms the data to follow a uniform or a normal distribution. It supports two output distributions: 'uniform' and 'normal'. The default distribution is 'uniform'. The distribution can be specified using the 'output_distribution' keyword argument in kwargs.

        Transform dataframes using different types of mathematical transformations.

        :param transformation_type: The type of transformation to apply. It can be one of these: 'cubicroot', 'log10', 'log', 'log2', 'sqrt', 'powertransformer', or 'quantiletransformer'.
        :type transformation_type: str
        :param kwargs: Additional arguments for some transformation types.
        :type kwargs: dict


        :raises: ValueError if the transformation_type is not one of the valid options.

        """

        dataframes = self.dataframes

        # Loop through the df_dict
        for dataset_name, df in dataframes.items():

            # Apply the normalization type to the dataframe
            # Apply the transformation type to the dataframe
            if transformation_type == 'cubicroot':
                transformed_df = np.cbrt(df)
            elif transformation_type == 'log10':
                transformed_df = np.log10(df)
            elif transformation_type == 'log':
                transformed_df = np.log(df)
            elif transformation_type == 'log2':
                transformed_df = np.log2(df)
            elif transformation_type == 'sqrt':
                transformed_df = np.sqrt(df)
            elif transformation_type == 'powertransformer':
                # Create a scaler object with the specified method
                method = kwargs.get('method', 'yeo-johnson')
                scaler = PowerTransformer(method=method)

                # Transform the dataframe and convert the result to a dataframe
                transformed_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
            elif transformation_type == 'quantiletransformer':
                # Create a scaler object with the specified output distribution
                output_distribution = kwargs.get('output_distribution', 'uniform')
                scaler = QuantileTransformer(output_distribution=output_distribution)

                # Transform the dataframe and convert the result to a dataframe
                transformed_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
            else:
                raise ValueError(f"Unsupported transformation type: {transformation_type} possible values are \n 'cubicroot', 'log10', 'log', 'log2', 'sqrt', 'powertransformer', or 'quantiletransformer'")
            # Return the transformed dataframe
            
            dataframes[dataset_name] = transformed_df
        self.dataframes = dataframes




    def fuseFeatures(self, n_components, method="pca",**kwargs):
        """

        The fusion methods are as follows:
           - 'pca': Principal Component Analysis, a linear dimensionality reduction technique that projects the data onto a lower-dimensional subspace that maximizes the variance.
           - 'ica': Independent Component Analysis, a linear dimensionality reduction technique that separates the data into independent sources based on the assumption of statistical independence.
           - 'ipca': Incremental Principal Component Analysis, a variant of PCA that allows for online updates of the components without requiring access to the entire dataset.
           - 'cca': Canonical Correlation Analysis, a linear dimensionality reduction technique that finds the linear combinations of two sets of variables that are maximally correlated with each other.
           - 'tsne': t-distributed Stochastic Neighbor Embedding, a non-linear dimensionality reduction technique that preserves the local structure of the data by embedding it in a lower-dimensional space with a probability distribution that matches the pairwise distances in the original space.
           - 'kpca': Kernel Principal Component Analysis, a non-linear extension of PCA that uses a kernel function to map the data into a higher-dimensional feature space where it is linearly separable.
           - 'rks': Random Kitchen Sinks, a non-linear dimensionality reduction technique that uses random projections to approximate a kernel function and map the data into a lower-dimensional feature space.
           - 'SEM': Structural Equation Modeling, a statistical technique that tests the causal relationships between multiple variables using a combination of regression and factor analysis.
           - 'isomap': Isometric Mapping, a non-linear dimensionality reduction technique that preserves the global structure of the data by embedding it in a lower-dimensional space with a geodesic distance that approximates the shortest path between points in the original space.
           - 'lle': Locally Linear Embedding, a non-linear dimensionality reduction technique that preserves the local structure of the data by reconstructing each point as a linear combination of its neighbors and embedding it in a lower-dimensional space that minimizes the reconstruction error.
           - 'autoencoder': A type of neural network that learns to encode the input data into a lower-dimensional latent representation and decode it back to the original data, minimizing the reconstruction error.
           - 'plsda': Partial Least Squares Discriminant Analysis, a supervised dimensionality reduction technique that finds the linear combinations of the features that best explain both the variance and the correlation with the target variable.
           - 'tensordecompose': Tensor Decomposition, a technique that decomposes a multi-dimensional array (tensor) into a set of lower-dimensional arrays (factors) that capture the latent structure and interactions of the data.
        
        Fuse the features of multiple dataframes using different methods.

        :param n_components: The number of components use for the fusion.
        :type n_components: int
        :param method: The method to use for the fusion. It can be one of these: 'pca', 'ica', 'ipca', 'cca', 'tsne', 'kpca', 'rks', 'SEM', 'isomap', 'lle', 'autoencoder', 'plsda', or 'tensordecompose'.
        :type method: str
        :param kwargs: Additional arguments for specific fusion methods.
        :type kwargs: dict

        :raises: ValueError if the method is not one of the valid options.


        """
        # Iterate through the dictionary and fusion DataFrame
        # train data fusion
        dataframes1 = self.dataframes
        df_list = []
        for name, df in dataframes1.items():
            df_list.append(df)
        if method in ['pca', 'ica', 'ipca']:
            merged_df = pd.concat(df_list, axis=1)
            fused_df1 = apply_analysis_linear1(merged_df, analysis_type=method, n_components=n_components, **kwargs)
        elif method in ['cca']:
            all_combinations = []
            all_combinations.extend(combinations(df_list, 2))
            all_fused =[]
            n=0
            for df_listn in all_combinations:
                fused_df_t = ccafuse(df_listn[0], df_listn[1],n_components)
                all_fused.append(fused_df_t)
            fused_df1 = pd.concat(all_fused, axis=1, sort=False)
        elif method in ['tsne', 'kpca', 'rks', 'SEM']:
            merged_df = pd.concat(df_list, axis=1)
            fused_df1 = apply_analysis_nonlinear1(merged_df,
                                                analysis_type=method,
                                                n_components=n_components,
                                                **kwargs)
        elif method in ['isomap', 'lle']:
            merged_df = pd.concat(df_list, axis=1)
            fused_df1 = apply_analysis_nonlinear2(merged_df, analysis_type=method, n_neighbors=5, n_components=n_components, **kwargs)
            #print("done")
        elif method in ['autoencoder']:
            merged_df = pd.concat(df_list, axis=1)
            fused_df1 = apply_analysis_nonlinear3(merged_df,
                                                analysis_type=method,
                                                lr=0.001,
                                                num_epochs = 20, 
                                                hidden_sizes=[128, 64, 36, 18],
                                                **kwargs)


        prediction_label = self.prediction_label
        if method in ['plsda']:
            df_list = []
            for name, df in dataframes1.items():
                df_list.append(df)
            merged_df = pd.concat(df_list, axis=1)
            fused_df1, pls_canonical = apply_analysis_linear2(merged_df, prediction_label, analysis_type=method, n_components=n_components, **kwargs)
            self.pls_model = pls_canonical

        ######################
            
        if method in ['tensordecompose']:
            df_list = []
            for name, df in dataframes1.items():
                df_list.append(df)
            df_list_selected=[]
            top_features = []
            for df in df_list:
                num_features = 100
                fs = SelectKBest(score_func=f_regression, k=num_features)
                #print(df)
                X_selected = fs.fit_transform(df, prediction_label)
                # print(fs.get_feature_names_out())
                #print(df.columns)
                top_feature = list(fs.get_feature_names_out())
                top_features.append(top_feature)
                df_list_selected.append(df[top_feature])
            all_selected = np.array(df_list_selected)
            fused_df = apply_analysis_nonlinear4(all_selected,
                                                analysis_type=method,
                                                n_components=n_components,
                                                tol=10e-6,
                                                **kwargs)
            fused_df1 = pd.DataFrame(fused_df, index =df.index, columns = [f'TD{i+1}' for i in range(fused_df.shape[1])])
            self.top_features = top_features
        self.fusedData = fused_df1
        print("Data is fused and saved to fusedData attribute")


    def fuseFeaturesTrain(self, n_components, method="pca",**kwargs):
        # The rest of the code is unchanged

        # Iterate through the dictionary and fusion DataFrame
        # train data fusion
        dataframes1 = self.train_dataframes
        df_list = []
        for name, df in dataframes1.items():
            df_list.append(df)
        if method in ['pca', 'ica', 'ipca']:
            merged_df = pd.concat(df_list, axis=1)
            fused_df1 = apply_analysis_linear1(merged_df, analysis_type=method, n_components=n_components, **kwargs)
        elif method in ['cca']:
            all_combinations = []
            all_combinations.extend(combinations(df_list, 2))
            all_fused =[]
            n=0
            for df_listn in all_combinations:
                fused_df_t = ccafuse(df_listn[0], df_listn[1],n_components)
                all_fused.append(fused_df_t)
            fused_df1 = pd.concat(all_fused, axis=1, sort=False)
        elif method in ['tsne', 'kpca', 'rks', 'SEM']:
            merged_df = pd.concat(df_list, axis=1)
            fused_df1 = apply_analysis_nonlinear1(merged_df,
                                                analysis_type=method,
                                                n_components=n_components,
                                                **kwargs)
        elif method in ['isomap', 'lle']:
            merged_df = pd.concat(df_list, axis=1)
            fused_df1 = apply_analysis_nonlinear2(merged_df, analysis_type=method, n_neighbors=5, n_components=n_components, **kwargs)
            #print("done")
        elif method in ['autoencoder']:
            merged_df = pd.concat(df_list, axis=1)
            fused_df1 = apply_analysis_nonlinear3(merged_df,
                                                analysis_type=method,
                                                lr=0.001,
                                                num_epochs = 20, 
                                                hidden_sizes=[128, 64, 36, 18],
                                                **kwargs)


        prediction_label = self.train_label
        if method in ['plsda']:
            df_list = []
            for name, df in dataframes1.items():
                df_list.append(df)
            merged_df = pd.concat(df_list, axis=1)
            fused_df1, pls_canonical = apply_analysis_linear2(merged_df,
                                                            prediction_label,
                                                            analysis_type=method,
                                                            n_components=n_components,
                                                              **kwargs)
            self.training_pls_model = pls_canonical

        ######################
            
        if method in ['tensordecompose']:
            df_list = []
            for name, df in dataframes1.items():
                df_list.append(df)
            df_list_selected=[]
            top_features = []
            for df in df_list:
                num_features = 100
                fs = SelectKBest(score_func=f_regression, k=num_features)
                #print(df)
                X_selected = fs.fit_transform(df, prediction_label)
                # print(fs.get_feature_names_out())
                #print(df.columns)
                top_feature = list(fs.get_feature_names_out())
                top_features.append(top_feature)
                df_list_selected.append(df[top_feature])
            all_selected = np.array(df_list_selected)
            fused_df = apply_analysis_nonlinear4(all_selected,
                                                analysis_type=method,
                                                n_components=n_components,
                                                tol=10e-6,
                                                **kwargs)
            fused_df1 = pd.DataFrame(fused_df, index =df.index, columns = [f'TD{i+1}' for i in range(fused_df.shape[1])])
            self.top_features_train = top_features
        self.fusedData_train = fused_df1
        print("Training data is fused.")

    
    def fuseFeaturesTest(self, n_components, method="pca",**kwargs):
        # Iterate through the dictionary and fusion DataFrame
        # train data fusion
        dataframes1 = self.test_dataframes
        df_list = []
        for name, df in dataframes1.items():
            df_list.append(df)
        if method in ['pca', 'ica', 'ipca']:
            merged_df = pd.concat(df_list, axis=1)
            fused_df1 = apply_analysis_linear1(merged_df, analysis_type=method, n_components=n_components, **kwargs)
        elif method in ['cca']:
            all_combinations = []
            all_combinations.extend(combinations(df_list, 2))
            all_fused =[]
            n=0
            for df_listn in all_combinations:
                fused_df_t = ccafuse(df_listn[0], df_listn[1],n_components)
                all_fused.append(fused_df_t)
            fused_df1 = pd.concat(all_fused, axis=1, sort=False)
        elif method in ['tsne', 'kpca', 'rks', 'SEM']:
            merged_df = pd.concat(df_list, axis=1)
            fused_df1 = apply_analysis_nonlinear1(merged_df,
                                                analysis_type=method,
                                                n_components=n_components,
                                                **kwargs)
        elif method in ['isomap', 'lle']:
            merged_df = pd.concat(df_list, axis=1)
            fused_df1 = apply_analysis_nonlinear2(merged_df, analysis_type=method, n_neighbors=5, n_components=n_components, **kwargs)
            #print("done")
        elif method in ['autoencoder']:
            merged_df = pd.concat(df_list, axis=1)
            fused_df1 = apply_analysis_nonlinear3(merged_df,
                                                analysis_type=method,
                                                lr=0.001,
                                                num_epochs = 20, 
                                                hidden_sizes=[128, 64, 36, 18],
                                                **kwargs)
        
        if method in ['plsda']:
            df_list = []
            for name, df in dataframes1.items():
                df_list.append(df)
            merged_df = pd.concat(df_list, axis=1)
            if "prediction_label" in merged_df.columns:
                # Remove the column "prediction_label" from the DataFrame
                merged_df = merged_df.drop(columns=["prediction_label"])
            #print(merged_df)
            pls_canonical = self.training_pls_model
            fused_df1 = pd.DataFrame(pls_canonical.transform(merged_df),
                                    columns=[f'PLS{i+1}' for i in range(pls_canonical.n_components)],
                                    index = merged_df.index) 
            
        ######################
            
        if method in ['tensordecompose']:
            top_features = self.top_features_train
            top_features = list(itertools.chain(*top_features))
            df_list = []
            for name, df in dataframes1.items():
                df_list.append(df)
            df_list_selected=[]
            for df in df_list:
                top_feature = list(set(df.columns.to_list()).intersection(set(top_features)))
                df_list_selected.append(df[top_feature])
            all_selected = np.array(df_list_selected)
            fused_df = apply_analysis_nonlinear4(all_selected,
                                                analysis_type=method,
                                                n_components=n_components,
                                                tol=10e-6,
                                                **kwargs)
            fused_df1 = pd.DataFrame(fused_df, index =df.index, columns = [f'TD{i+1}' for i in range(fused_df.shape[1])])
        self.fusedData_test = fused_df1
        print("Testing data is fused. ")

    def evaluate_fusion_models(self, n_components=10, methods=['cca', 'pca'], regression=False):
        """
        Evaluate different fusion models on the dataframes and prediction labels. 

        The fusion methods are as follows: 

        - 'pca': Principal Component Analysis
        - 'ica': Independent Component Analysis
        - 'ipca': Incremental Principal Component Analysis
        - 'cca': Canonical Correlation Analysis
        - 'tsne': t-distributed Stochastic Neighbor Embedding
        - 'kpca': Kernel Principal Component Analysis
        - 'rks': Random Kitchen Sinks
        - 'SEM': Structural Equation Modeling
        - 'autoencoder': Autoencoder
        - 'tensordecompose': Tensor Decomposition
        - 'plsda': Partial Least Squares Discriminant Analysis 

        :param n_components: The number of components use for the fusion. The default is 10.
        :type n_components: int
        :param n_folds: The number of folds to use for cross-validation. The default is 10.
        :type n_folds: int
        :param methods: A list of fusion methods to apply. The default is ['cca', 'pca'].
        :type methods: list
        :param regression: If want to create a regression model. The default is False.
        :type regression: bool

        :raises: ValueError if any of the methods are invalid.
        """


        dataframes = self.dataframes
        prediction_labels = self.prediction_label
        self.train_dataframes, self.test_dataframes, self.train_label, self.test_label  = save_train_test_data(dataframes, prediction_labels, output_dir="comaprision_data")
        methods_chemdices = ['pca', 'ica', 'ipca', 'cca', 'tsne', 'kpca', 'rks', 'SEM', 'autoencoder', 'tensordecompose', 'plsda']
        valid_methods_chemdices = [method for method in methods if method in methods_chemdices]
        invalid_methods_chemdices = [method for method in methods if method not in methods_chemdices]



        if len(invalid_methods_chemdices):
            methods_chemdices_text = ",".join(methods_chemdices)
            invalid_methods_chemdices_text = ",".join(invalid_methods_chemdices)
            ValueError(f"These methods are invalid:{invalid_methods_chemdices_text}\nValid methods are : {methods_chemdices_text}")


        if regression == False:
            # Define a classification of models
            models = [
                ("Logistic Regression", LogisticRegression()),
                ("Decision Tree", DecisionTreeClassifier()),
                ("Random Forest", RandomForestClassifier()),
                ("Support Vector Machine", SVC(probability=True)),
                ("Naive Bayes", GaussianNB())
            ]
            # Initialize a dictionary to store the metrics
            metrics = {
                "Model type": [],
                "Model": [],
                "AUC": [],
                "Accuracy": [],
                "Precision": [],
                "Recall": [],
                "f1 score":[],
                "Balanced accuracy":[],
                "MCC":[],
                "Kappa":[]
            }



            for method_chemdice in valid_methods_chemdices:
                # Fuse all data in dataframes
                print("Method name",method_chemdice)
                self.fuseFeaturesTrain(n_components = n_components,  method=method_chemdice)
                X_train = self.fusedData_train
                y_train = self.train_label
                self.fuseFeaturesTest(n_components = n_components,  method=method_chemdice)
                X_test = self.fusedData_test
                y_test = self.test_label

                # Loop through the models and evaluate them
                for name, model in models:
                    if method_chemdice in ['pca', 'ica', 'ipca', 'cca', 'plsda']:
                        metrics["Model type"].append("linear")
                    else:
                        metrics["Model type"].append("Non-linear")
                    name = method_chemdice+" "+ name 
                    # Fit the model on the train set
                    model.fit(X_train, y_train)
                    # Predict the probabilities on the test set
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    # Compute the metrics
                    auc = roc_auc_score(y_test, y_pred_proba)
                    accuracy = accuracy_score(y_test, y_pred_proba > 0.5)
                    precision = precision_score(y_test, y_pred_proba > 0.5)
                    recall = recall_score(y_test, y_pred_proba > 0.5)
                    f1 = f1_score(y_test, y_pred_proba > 0.5)
                    baccuracy = balanced_accuracy_score(y_test, y_pred_proba > 0.5)
                    mcc = matthews_corrcoef(y_test, y_pred_proba > 0.5)
                    kappa = cohen_kappa_score(y_test, y_pred_proba > 0.5)
                    # Append the metrics to the dictionary
                    metrics["Model"].append(name)
                    metrics["AUC"].append(auc)
                    metrics["Accuracy"].append(accuracy)
                    metrics["Precision"].append(precision)
                    metrics["Recall"].append(recall)
                    metrics["f1 score"].append(f1)
                    metrics["Balanced accuracy"].append(baccuracy)
                    metrics["MCC"].append(mcc)
                    metrics["Kappa"].append(kappa)
            # Convert the dictionary to a dataframe
            metrics_df = pd.DataFrame(metrics)
            metrics_df = metrics_df.sort_values(by = "AUC", ascending=False)
            self.Accuracy_metrics = metrics_df
            
        else:
            # Define a list of regression models
            models = [
                ("Linear Regression", LinearRegression()),
                ("Ridge", Ridge()),
                ("Lasso", Lasso()),
                ("ElasticNet", ElasticNet()),
                ("Decision Tree", DecisionTreeRegressor()),
                ("Random Forest", RandomForestRegressor()),
                ("Gradient Boosting", GradientBoostingRegressor()),
                ("AdaBoost", AdaBoostRegressor()),
                ("Support Vector Machine", SVR()),
                ("K Neighbors", KNeighborsRegressor()),
                ("MLP", MLPRegressor()),
                ("Gaussian Process", GaussianProcessRegressor()),
                ("Kernel Ridge", KernelRidge())
            ]

            # Initialize a dictionary to store the metrics
            metrics = {
                "Model type": [],
                "Model": [],
                "R2 Score": [],
                "MSE": [],
                "RMSE":[],
                "MAE":[]
            }
            #methods_chemdices = ['pca', 'ica', 'ipca', 'cca', 'tsne', 'kpca', 'rks', 'SEM', 'autoencoder', 'tensordecompose', 'plsda']

            for method_chemdice in valid_methods_chemdices:
                # Fuse all data in dataframes
                print("Method name", method_chemdice)
                self.fuseFeaturesTrain(n_components=n_components, method=method_chemdice)
                X_train = self.fusedData_train
                y_train = self.train_label
                self.fuseFeaturesTest(n_components=n_components, method=method_chemdice)
                X_test = self.fusedData_test
                y_test = self.test_label

                # Loop through the models and evaluate them
                for name, model in models:
                    if method_chemdice in ['pca', 'ica', 'ipca', 'cca', 'plsda']:
                        metrics["Model type"].append("linear")
                    else:
                        metrics["Model type"].append("Non-linear")
                    name = method_chemdice + " " + name
                    # Fit the model on the train set
                    model.fit(X_train, y_train)
                    # Predict on the test set
                    y_pred = model.predict(X_test)
                    # Compute the metrics
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    # Append the metrics to the dictionary
                    metrics["Model"].append(name)
                    metrics["MSE"].append(mse)
                    metrics["RMSE"].append(rmse)
                    metrics["MAE"].append(mae)
                    metrics["R2 Score"].append(r2)

            # Convert the dictionary to a dataframe
            metrics_df = pd.DataFrame(metrics)
            metrics_df = metrics_df.sort_values(by="R2 Score", ascending=False)
            self.Accuracy_metrics = metrics_df


    def evaluate_fusion_models_nfold(self, n_components=10, n_folds=10, methods=['cca','pca'],regression = False):
        """
        Evaluate n-fold cross validations of different fusion models on the dataframes and prediction labels. 

        The fusion methods are as follows: 

        - 'pca': Principal Component Analysis
        - 'ica': Independent Component Analysis
        - 'ipca': Incremental Principal Component Analysis
        - 'cca': Canonical Correlation Analysis
        - 'tsne': t-distributed Stochastic Neighbor Embedding
        - 'kpca': Kernel Principal Component Analysis
        - 'rks': Random Kitchen Sinks
        - 'SEM': Structural Equation Modeling
        - 'autoencoder': Autoencoder
        - 'tensordecompose': Tensor Decomposition
        - 'plsda': Partial Least Squares Discriminant Analysis 

        :param n_components: The number of components use for the fusion. The default is 10.
        :type n_components: int
        :param n_folds: The number of folds to use for cross-validation. The default is 10.
        :type n_folds: int
        :param methods: A list of fusion methods to apply. The default is ['cca', 'pca'].
        :type methods: list
        :param regression: If want to create a regression model. The default is False.
        :type regression: bool

        :raises: ValueError if any of the methods are invalid.

        """


        methods_chemdices = ['pca', 'ica', 'ipca', 'cca', 'tsne', 'kpca', 'rks', 'SEM', 'autoencoder', 'tensordecompose', 'plsda']
            
            
        valid_methods_chemdices = [method for method in methods if method in methods_chemdices]
        

        invalid_methods_chemdices = [method for method in methods if method not in methods_chemdices]
        


        if len(invalid_methods_chemdices):
            methods_chemdices_text = ",".join(methods_chemdices)
            invalid_methods_chemdices_text = ",".join(invalid_methods_chemdices)
            ValueError(f"These methods are invalid:{invalid_methods_chemdices_text}\nValid methods are : {methods_chemdices_text}")

        if regression == False:
            # Define a list of models
            models = [
                ("Logistic Regression", LogisticRegression()),
                ("Decision Tree", DecisionTreeClassifier()),
                ("Random Forest", RandomForestClassifier()),
                ("Support Vector Machine", SVC(probability=True)),
                ("Naive Bayes", GaussianNB())
            ]
            # Initialize a dictionary to store the metrics
            metrics = {
                "Model type": [],
                "Fold": [],
                "Model": [],
                "AUC": [],
                "Accuracy": [],
                "Precision": [],
                "Recall": [],
                "f1 score":[],
                "Balanced accuracy":[],
                "MCC":[],
                "Kappa":[]
            }

            methods_chemdices = ['pca', 'ica', 'ipca', 'cca', 'tsne', 'kpca', 'rks', 'SEM', 'autoencoder', 'tensordecompose', 'plsda']
            
            
            valid_methods_chemdices = [method for method in methods if method in methods_chemdices]
            

            invalid_methods_chemdices = [method for method in methods if method not in methods_chemdices]
            
            methods_chemdices_text = ",".join(methods_chemdices)
            invalid_methods_chemdices_text = ",".join(invalid_methods_chemdices)

            if len(invalid_methods_chemdices):
                ValueError(f"These methods are invalid:{invalid_methods_chemdices_text}\n Valid methods are : {methods_chemdices_text}")
            dataframes = self.dataframes
            prediction_labels = self.prediction_label
            
            kf = KFold(n_splits=n_folds)
            for fold_number, (train_index, test_index) in enumerate(kf.split(list(dataframes.values())[0])): 

                self.train_dataframes, self.test_dataframes, self.train_label, self.test_label  = save_train_test_data_n_fold(dataframes, prediction_labels, train_index, test_index, output_dir="comaprision_data_fold_"+str(fold_number)+"_of_"+str(n_folds))


                for method_chemdice in valid_methods_chemdices:
                    # Fuse all data in dataframes
                    print("Method name",method_chemdice)
                    self.fuseFeaturesTrain(n_components = n_components,  method=method_chemdice)
                    X_train = self.fusedData_train
                    y_train = self.train_label
                    self.fuseFeaturesTest(n_components = n_components,  method=method_chemdice)
                    X_test = self.fusedData_test
                    y_test = self.test_label

                    # Loop through the models and evaluate them
                    for name, model in models:
                        if method_chemdice in ['pca', 'ica', 'ipca', 'cca', 'plsda']:
                            metrics["Model type"].append("linear")
                        else:
                            metrics["Model type"].append("Non-linear")
                        name = method_chemdice+" "+ name 
                        # Fit the model on the train set
                        model.fit(X_train, y_train)
                        # Predict the probabilities on the test set
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        # Compute the metrics
                        auc = roc_auc_score(y_test, y_pred_proba)
                        accuracy = accuracy_score(y_test, y_pred_proba > 0.5)
                        precision = precision_score(y_test, y_pred_proba > 0.5)
                        recall = recall_score(y_test, y_pred_proba > 0.5)
                        f1 = f1_score(y_test, y_pred_proba > 0.5)
                        baccuracy = balanced_accuracy_score(y_test, y_pred_proba > 0.5)
                        mcc = matthews_corrcoef(y_test, y_pred_proba > 0.5)
                        kappa = cohen_kappa_score(y_test, y_pred_proba > 0.5)
                        # Append the metrics to the dictionary
                        metrics["Model"].append(name)
                        metrics["Fold"].append(fold_number)
                        metrics["AUC"].append(auc)
                        metrics["Accuracy"].append(accuracy)
                        metrics["Precision"].append(precision)
                        metrics["Recall"].append(recall)
                        metrics["f1 score"].append(f1)
                        metrics["Balanced accuracy"].append(baccuracy)
                        metrics["MCC"].append(mcc)
                        metrics["Kappa"].append(kappa)
                        
            metrics_df = pd.DataFrame(metrics)

            # Group by 'Model' and calculate the mean of 'AUC'
            grouped_means = metrics_df.groupby('Model')["AUC"].mean()
            metrics_df = metrics_df.sort_values(by=['Model', 'Fold'], key=lambda x: x.map(grouped_means),ascending=[False, True])
            self.Accuracy_metrics = metrics_df
            
            grouped_means = metrics_df.groupby('Model').mean(numeric_only = True).reset_index()
            metrics_df = grouped_means.drop(columns=['Fold'])
            metrics_df = metrics_df.sort_values(by="AUC",ascending = False)
            self.mean_accuracy_metrics = metrics_df


        else:
            # Define a list of regression models
            models = [
                ("Linear Regression", LinearRegression()),
                ("Ridge", Ridge()),
                ("Lasso", Lasso()),
                ("ElasticNet", ElasticNet()),
                ("Decision Tree", DecisionTreeRegressor()),
                ("Random Forest", RandomForestRegressor()),
                ("Gradient Boosting", GradientBoostingRegressor()),
                ("AdaBoost", AdaBoostRegressor()),
                ("Support Vector Machine", SVR()),
                ("K Neighbors", KNeighborsRegressor()),
                ("MLP", MLPRegressor()),
                ("Gaussian Process", GaussianProcessRegressor()),
                ("Kernel Ridge", KernelRidge())
            ]

            # Initialize a dictionary to store the metrics
            metrics = {
                "Model type": [],
                "Model": [],
                'Fold': [],
                "R2 Score": [],
                "MSE": [],
                "RMSE":[],
                "MAE":[]
            }


            dataframes = self.dataframes
            prediction_labels = self.prediction_label
            
            kf = KFold(n_splits=n_folds)
            for fold_number, (train_index, test_index) in enumerate(kf.split(list(dataframes.values())[0])): 

                self.train_dataframes, self.test_dataframes, self.train_label, self.test_label  = save_train_test_data_n_fold(dataframes, prediction_labels, train_index, test_index, output_dir="comaprision_data_fold_"+str(fold_number)+"_of_"+str(n_folds))


                for method_chemdice in valid_methods_chemdices:
                    # Fuse all data in dataframes
                    print("Method name", method_chemdice)
                    self.fuseFeaturesTrain(n_components=n_components, method=method_chemdice)
                    X_train = self.fusedData_train
                    y_train = self.train_label
                    self.fuseFeaturesTest(n_components=n_components, method=method_chemdice)
                    X_test = self.fusedData_test
                    y_test = self.test_label

                    # Loop through the models and evaluate them
                    for name, model in models:
                        if method_chemdice in ['pca', 'ica', 'ipca', 'cca', 'plsda']:
                            metrics["Model type"].append("linear")
                        else:
                            metrics["Model type"].append("Non-linear")
                        name = method_chemdice + " " + name
                        # Fit the model on the train set
                        model.fit(X_train, y_train)
                        # Predict on the test set
                        y_pred = model.predict(X_test)
                        # Compute the metrics
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        # Append the metrics to the dictionary
                        metrics["Model"].append(name)
                        metrics["Fold"].append(fold_number)
                        metrics["MSE"].append(mse)
                        metrics["RMSE"].append(rmse)
                        metrics["MAE"].append(mae)
                        metrics["R2 Score"].append(r2)

            # Convert the dictionary to a dataframe
            metrics_df = pd.DataFrame(metrics)
            # Group by 'Model' and calculate the mean of 'R2 Score'
            grouped_means = metrics_df.groupby('Model')["R2 Score"].mean()
            metrics_df = metrics_df.sort_values(by=['Model', 'Fold'], key=lambda x: x.map(grouped_means),ascending=[False, True])
            self.Accuracy_metrics = metrics_df
            
            grouped_means = metrics_df.groupby('Model').mean(numeric_only = True).reset_index()
            metrics_df = grouped_means.drop(columns=['Fold'])
            metrics_df = metrics_df.sort_values(by="R2 Score",ascending = False)
            self.mean_accuracy_metrics = metrics_df



    def evaluate_base_models_nfold(self, n_components=10, n_folds=10,regression = False):

        if regression == False:
            # Define a list of models
            models = [
                ("Logistic Regression", LogisticRegression()),
                ("Decision Tree", DecisionTreeClassifier()),
                ("Random Forest", RandomForestClassifier()),
                ("Support Vector Machine", SVC(probability=True)),
                ("Naive Bayes", GaussianNB())
            ]
            # Initialize a dictionary to store the metrics
            metrics = {
                "Model type": [],
                "Fold": [],
                "Model": [],
                "AUC": [],
                "Accuracy": [],
                "Precision": [],
                "Recall": [],
                "f1 score":[],
                "Balanced accuracy":[],
                "MCC":[],
                "Kappa":[]
            }
            dataframes = self.dataframes
            prediction_labels = self.prediction_label
            
            kf = KFold(n_splits=n_folds)
            for fold_number, (train_index, test_index) in enumerate(kf.split(list(dataframes.values())[0])): 

                self.train_dataframes, self.test_dataframes, self.train_label, self.test_label  = save_train_test_data_n_fold(dataframes, prediction_labels, train_index, test_index, output_dir="comaprision_data_fold_"+str(fold_number)+"_of_"+str(n_folds))

                
                for (name_train,Data_train),(name_test,Data_test) in zip(self.train_dataframes.items(), self.test_dataframes.items()):
                    # Fuse all data in dataframes
                    print("Method name",name_train)
                    X_train = Data_train
                    y_train = self.train_label
                    X_test = Data_test
                    y_test = self.test_label

                    # Loop through the models and evaluate them
                    for name, model in models:
                        metrics["Model type"].append("No fusion")
                        name = name_train+" "+ name 
                        # Fit the model on the train set
                        model.fit(X_train, y_train)
                        # Predict the probabilities on the test set
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        # Compute the metrics
                        auc = roc_auc_score(y_test, y_pred_proba)
                        accuracy = accuracy_score(y_test, y_pred_proba > 0.5)
                        precision = precision_score(y_test, y_pred_proba > 0.5)
                        recall = recall_score(y_test, y_pred_proba > 0.5)
                        f1 = f1_score(y_test, y_pred_proba > 0.5)
                        baccuracy = balanced_accuracy_score(y_test, y_pred_proba > 0.5)
                        mcc = matthews_corrcoef(y_test, y_pred_proba > 0.5)
                        kappa = cohen_kappa_score(y_test, y_pred_proba > 0.5)
                        # Append the metrics to the dictionary
                        metrics["Model"].append(name)
                        metrics["Fold"].append(fold_number)
                        metrics["AUC"].append(auc)
                        metrics["Accuracy"].append(accuracy)
                        metrics["Precision"].append(precision)
                        metrics["Recall"].append(recall)
                        metrics["f1 score"].append(f1)
                        metrics["Balanced accuracy"].append(baccuracy)
                        metrics["MCC"].append(mcc)
                        metrics["Kappa"].append(kappa)
                        
            metrics_df = pd.DataFrame(metrics)

            # Group by 'Model' and calculate the mean of 'AUC'
            grouped_means = metrics_df.groupby('Model')["AUC"].mean()
            metrics_df = metrics_df.sort_values(by=['Model', 'Fold'], key=lambda x: x.map(grouped_means),ascending=[False, True])
            self.Accuracy_metrics = metrics_df
            
            grouped_means = metrics_df.groupby('Model').mean(numeric_only = True).reset_index()
            metrics_df = grouped_means.drop(columns=['Fold'])
            metrics_df = metrics_df.sort_values(by="AUC",ascending = False)
            self.mean_accuracy_metrics = metrics_df


        else:
            # Define a list of regression models
            models = [
                ("Linear Regression", LinearRegression()),
                ("Ridge", Ridge()),
                ("Lasso", Lasso()),
                ("ElasticNet", ElasticNet()),
                ("Decision Tree", DecisionTreeRegressor()),
                ("Random Forest", RandomForestRegressor()),
                ("Gradient Boosting", GradientBoostingRegressor()),
                ("AdaBoost", AdaBoostRegressor()),
                ("Support Vector Machine", SVR()),
                ("K Neighbors", KNeighborsRegressor()),
                ("MLP", MLPRegressor()),
                ("Gaussian Process", GaussianProcessRegressor()),
                ("Kernel Ridge", KernelRidge())
            ]

            # Initialize a dictionary to store the metrics
            metrics = {
                "Model type": [],
                "Model": [],
                'Fold': [],
                "R2 Score": [],
                "MSE": [],
                "RMSE":[],
                "MAE":[]
            }


            dataframes = self.dataframes
            prediction_labels = self.prediction_label
            
            kf = KFold(n_splits=n_folds)
            for fold_number, (train_index, test_index) in enumerate(kf.split(list(dataframes.values())[0])): 

                self.train_dataframes, self.test_dataframes, self.train_label, self.test_label  = save_train_test_data_n_fold(dataframes, prediction_labels, train_index, test_index, output_dir="comaprision_data_fold_"+str(fold_number)+"_of_"+str(n_folds))


                for (name_train,Data_train),(name_test,Data_test) in zip(self.train_dataframes.items(), self.test_dataframes.items()):
                    # Fuse all data in dataframes
                    print("Method name",name_train)
                    X_train = Data_train
                    y_train = self.train_label
                    X_test = Data_test
                    y_test = self.test_label

                    # Loop through the models and evaluate them
                    for name, model in models:
                        metrics["Model type"].append("No fusion")
                        name = name_train+" "+ name 
                        # Fit the model on the train set
                        model.fit(X_train, y_train)
                        # Predict on the test set
                        y_pred = model.predict(X_test)
                        # Compute the metrics
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        # Append the metrics to the dictionary
                        metrics["Model"].append(name)
                        metrics["Fold"].append(fold_number)
                        metrics["MSE"].append(mse)
                        metrics["RMSE"].append(rmse)
                        metrics["MAE"].append(mae)
                        metrics["R2 Score"].append(r2)

            # Convert the dictionary to a dataframe
            metrics_df = pd.DataFrame(metrics)
            # Group by 'Model' and calculate the mean of 'R2 Score'
            grouped_means = metrics_df.groupby('Model')["R2 Score"].mean()
            metrics_df = metrics_df.sort_values(by=['Model', 'Fold'], key=lambda x: x.map(grouped_means),ascending=[False, True])
            self.Accuracy_metrics = metrics_df
            
            grouped_means = metrics_df.groupby('Model').mean(numeric_only = True).reset_index()
            metrics_df = grouped_means.drop(columns=['Fold'])
            metrics_df = metrics_df.sort_values(by="R2 Score",ascending = False)
            self.mean_accuracy_metrics = metrics_df
