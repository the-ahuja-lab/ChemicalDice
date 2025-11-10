import os
import re
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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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
from ChemicalDice.getEmbeddings import AutoencoderReconstructor_training_other , AutoencoderReconstructor_training_8192, AutoencoderReconstructor_training_single
import time
from ChemicalDice.splitting import random_scaffold_split_train_val_test, scaffold_split_balanced_train_val_test, scaffold_split_train_val_test


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern
import pickle


class fusionData:
    """
    Class for handling data and performing fusion tasks.
    Initialize fusionData object with provided data paths, label file path, ID column name,
    and prediction label column name.

    Args:
        data_paths (dict): Dictionary of file paths containing the data.
        label_file_path (str): file paths containing the label data.
        id_column (str, optional): Name of the column representing unique identifiers.
            Defaults to "id".
        prediction_label_column (str, optional): Name of the column containing prediction labels.
            Defaults to "labels".

    Attributes:
        - data_paths (list): List of file paths containing the data.
        - id_column (str): Name of the column representing unique identifiers.
        - dataframes (dict): Dictionary containing processed dataframes.
        - prediction_label: Placeholder for labels.


    """
    def __init__(self, data_paths, label_file_path, id_column = "id", label_column = "labels"):

        self.data_paths = data_paths
        loaded_data_dict = clear_and_process_data(data_paths, id_column)



        label_df = pd.read_csv(label_file_path)
        label_df.set_index(id_column, inplace=True)
        self.prediction_label = label_df[label_column]
        self.smiles_col = label_df["SMILES"]

        #check_missing_values(loaded_data_dict)
        #self.prediction_label = None
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
        self.training_AER_model = None
        self.AER_model_embt_size = None
        self.AER_model_explainability = None

    
    
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
            filtered_df_dict[name] = filtered_df

        self.prediction_label = self.prediction_label.loc[list(common_indices)]
        self.smiles_col = self.smiles_col.loc[list(common_indices)]
        self.dataframes = filtered_df_dict


        
    def remove_empty_features(self,threshold=100):
        
        """
        Remove columns with more than a threshold percentage of missing values from dataframes.

        :param threshold: The percentage threshold of missing values to drop a column. It should be between 0 and 100.
        :type threshold: float

        """
        dataframes = self.dataframes
        for name, df in dataframes.items():
            if name == 'mordred':
                common_columns = ['SlogP_VSA9', 'ATS7are', 'n8HRing', 'n5Ring', 'FNSA4', 'ECIndex', 'NsssssP', 'ATSC1se', 'SMR', 'AATS3p', 'ATS5m', 'SssGeH2', 'ATSC5i', 'AMID', 'Sm', 'nG12AHRing', 'SssssN', 'n9aHRing', 'nRing', 'SsNH2', 'nF', 'MATS3v', 'AETA_beta_ns', 'SIC3', 'ATS5i', 'n9ARing', 'nBondsS', 'AATS6Z', 'GATS6s', 'ATSC2Z', 'nBondsT', 'AATS7Z', 'PEOE_VSA10', 'SddssSe', 'GGI5', 'Xc-5d', 'SsssSiH', 'n8aHRing', 'ATS4m', 'MATS5se', 'MPC10', 'GATS4m', 'nN', 'NsssP', 'NsssAs', 'n6AHRing', 'ATSC2s', 'Xpc-4dv', 'ATSC5dv', 'AATSC4pe', 'nFaHRing', 'ATSC7i', 'MWC08', 'ATSC0i', 'SpMax_Dzi', 'VSA_EState4', 'LogEE_Dzi', 'AATSC0c', 'ATSC4pe', 'AXp-4dv', 'GATS3are', 'StsC', 'SaaO', 'GATS1c', 'Xc-4dv', 'ATS8are', 'AMID_O', 'JGI6', 'SsssCH', 'SZ', 'MATS7c', 'SaaSe', 'SpDiam_Dzp', 'MATS5c', 'n6aHRing', 'SpMAD_Dzse', 'AATSC6se', 'MATS1i', 'n9FHRing', 'NssSnH2', 'MATS5v', 'GGI8', 'MATS2i', 'MOMI-Z', 'ATS4Z', 'NssssSi', 'AATS6are', 'n5FAHRing', 'NssSe', 'Sare', 'AETA_dBeta', 'n11ARing', 'SsPbH3', 'Xch-6d', 'ATSC0m', 'ETA_epsilon_5', 'Mse', 'SpAD_D', 'AATSC1i', 'n8AHRing', 'MID_X', 'SpMAD_Dzp', 'n9FARing', 'ATSC3s', 'AATSC0s', 'AATSC2m', 'ATSC0c', 'AATSC2c', 'VE3_Dzi', 'AATSC6c', 'GATS1m', 'MATS3i', 'SdSe', 'SssSe', 'AATSC2dv', 'n9FaRing', 'MPC4', 'SpAbs_A', 'C2SP2', 'AATSC5v', 'ETA_psi_1', 'NddssS', 'AATSC0d', 'GATS1pe', 'DPSA5', 'AATSC2se', 'MWC04', 'AATSC5p', 'SssSnH2', 'RNCS', 'SpMax_Dzse', 'ATS6are', 'PEOE_VSA2', 'NssO', 'AATSC7p', 'BIC0', 'AXp-1d', 'SpAbs_Dzi', 'ATSC1d', 'Xch-7d', 'nG12Ring', 'SdCH2', 'SMR_VSA7', 'ATSC0Z', 'n7FRing', 'MWC07', 'AATS0p', 'ATSC7pe', 'n5FHRing', 'ATS0pe', 'GATS4s', 'Xch-4dv', 'ATS2s', 'ATS6s', 'GATS3s', 'AETA_eta_R', 'MATS5d', 'NsPbH3', 'EState_VSA7', 'AATSC7pe', 'AATSC0pe', 'Xp-7d', 'n6aRing', 'SpMAD_Dzpe', 'SlogP_VSA2', 'n11FAHRing', 'ATS3v', 'GATS3Z', 'GATS6m', 'n9FAHRing', 'BCUTse-1h', 'TopoPSA(NO)', 'Zagreb2', 'ATSC0p', 'EState_VSA4', 'AATSC6v', 'AATS4m', 'nHeavyAtom', 'ATS8dv', 'VR1_Dzi', 'VSA_EState7', 'ATS3dv', 'MPC9', 'VE1_Dzpe', 'n12AHRing', 'NsNH2', 'ATSC0s', 'ATS8d', 'PNSA3', 'SpDiam_Dzi', 'SpDiam_Dzare', 'n11FaHRing', 'piPC1', 'piPC10', 'AATSC5i', 'MATS4c', 'AATSC3Z', 'NdNH', 'VR1_Dzse', 'MATS7pe', 'SssNH', 'ATSC1p', 'ATS3Z', 'PPSA4', 'AATSC3s', 'ATS1dv', 'PEOE_VSA6', 'AATS0m', 'AXp-5dv', 'n12HRing', 'VE1_Dzm', 'VR3_Dzare', 'SpMAD_Dzv', 'HybRatio', 'nBase', 'GATS1are', 'AATS2pe', 'SsCl', 'AATSC7m', 'mZagreb1', 'SsNH3', 'SM1_DzZ', 'VE3_A', 'NtsC', 'SssAsH', 'ETA_epsilon_2', 'TMPC10', 'BIC4', 'AATS3pe', 'nBondsKD', 'MOMI-Y', 'GATS1i', 'n8ARing', 'SlogP_VSA8', 'mZagreb2', 'AATS1m', 'GATS5pe', 'ATSC3pe', 'MATS4are', 'VMcGowan', 'MATS2Z', 'TIC3', 'PNSA5', 'BCUTv-1l', 'AATS3se', 'GATS4dv', 'piPC9', 'naHRing', 'ATSC8se', 'n11AHRing', 'GATS6se', 'GATS2m', 'WPSA2', 'AXp-7d', 'RNCG', 'VE1_Dzi', 'VE1_Dzv', 'AATS6pe', 'AATS5se', 'AETA_eta', 'SM1_Dzv', 'GATS7d', 'MATS6se', 'SRW02', 'JGI4', 'SpMAD_A', 'MPC5', 'SpDiam_D', 'GATS5are', 'AATSC4dv', 'VSA_EState9', 'n5FRing', 'MATS5m', 'n6FaHRing', 'GATS5dv', 'NaaSe', 'ATS1Z', 'MATS7are', 'MIC2', 'MATS1pe', 'StCH', 'ATSC2p', 'nG12aRing', 'MATS1d', 'AMW', 'SsCH3', 'n10FARing', 'n11FaRing', 'LogEE_Dzpe', 'Xpc-6dv', 'JGI8', 'VR3_Dzp', 'AATSC1dv', 'MATS6pe', 'ATS2dv', 'ATS4dv', 'SaaCH', 'BCUTse-1l', 'ATSC1dv', 'SpAbs_Dzare', 'AETA_eta_BR', 'MATS6i', 'Mm', 'AATS7i', 'SpMAD_Dzi', 'MATS6dv', 'PEOE_VSA13', 'n6Ring', 'NsssSiH', 'AATS7are', 'GATS3pe', 'n5HRing', 'SsSH', 'BCUTc-1l', 'n12ARing', 'PEOE_VSA7', 'GATS5c', 'nBondsO', 'ATSC5are', 'AATSC3dv', 'ZMIC4', 'GRAV', 'MATS6s', 'SM1_Dzare', 'JGI5', 'nAtom', 'AATS4i', 'FPSA1', 'n5FaRing', 'ATSC5Z', 'nS', 'GATS2v', 'ATSC4s', 'Xp-7dv', 'FNSA5', 'GRAVp', 'NsssSnH', 'VR2_D', 'AATS3m', 'LogEE_Dzm', 'NsSnH3', 'nFaRing', 'NssssN', 'DPSA3', 'n4FRing', 'StN', 'nG12FHRing', 'AATS7m', 'MATS2d', 'n12FaRing', 'ATSC3Z', 'NdssC', 'AATS6i', 'ATSC5p', 'FPSA3', 'SsSeH', 'C2SP3', 'AXp-6d', 'ATSC1Z', 'AATSC6s', 'SIC1', 'AATS4Z', 'SRW09', 'WPSA4', 'SIC2', 'ATSC8v', 'n7aRing', 'ATS1m', 'VR1_Dzv', 'nFAHRing', 'n11aHRing', 'DPSA2', 'AATS5dv', 'ATSC8c', 'VE3_Dzse', 'SpMax_Dzpe', 'NddC', 'Radius', 'Xp-5dv', 'n8FAHRing', 'BCUTd-1h', 'ATS6d', 'Xc-6d', 'TpiPC10', 'TMWC10', 'NssssPb', 'ATS8m', 'TIC5', 'TopoPSA', 'SssPbH2', 'AATSC2Z', 'n6FARing', 'ATS5p', 'AATS7se', 'NdSe', 'AATS6v', 'GATS7pe', 'VE3_Dzpe', 'Xc-3dv', 'SsAsH2', 'NssBH', 'GATS1dv', 'ATS8Z', 'NssPH', 'n10FaRing', 'ATSC1c', 'SpDiam_Dzse', 'GATS5se', 'EState_VSA5', 'AATSC5pe', 'AATS5Z', 'nBr', 'AATSC0Z', 'AATS4dv', 'naRing', 'SssssPb', 'n8Ring', 'ATSC1s', 'MATS1s', 'AATSC6pe', 'PNSA4', 'ETA_beta_ns_d', 'BCUTZ-1h', 'Xp-0d', 'ATSC8s', 'VE2_Dzp', 'CIC0', 'AATS3Z', 'SpAD_DzZ', 'NdsCH', 'nAromAtom', 'ATS6p', 'ETA_dAlpha_B', 'SpDiam_DzZ', 'AATS0i', 'GATS4pe', 'ATS4p', 'ATSC0dv', 'SpMax_A', 'NsAsH2', 'ETA_shape_x', 'ATS8i', 'ETA_dPsi_B', 'ATSC3se', 'SlogP_VSA6', 'VR2_A', 'NssssGe', 'AATS0pe', 'AATS7d', 'n7Ring', 'NdO', 'MATS4se', 'VE3_Dzm', 'GATS4v', 'MATS1dv', 'TIC4', 'MPC6', 'n9Ring', 'ATS5d', 'WPath', 'SRW04', 'ATS2Z', 'VR2_Dzpe', 'SlogP_VSA7', 'MATS1m', 'n7FARing', 'ATS3are', 'n11Ring', 'Xch-3dv', 'AATS2dv', 'SpMAD_Dzare', 'GATS7v', 'AATS0se', 'SpAD_Dzse', 'ATSC2dv', 'NaaNH', 'SpMax_Dzv', 'AATS0d', 'ATS2se', 'AATSC4m', 'Xch-6dv', 'AXp-5d', 'AXp-0dv', 'ATSC6dv', 'ATS2v', 'GeomPetitjeanIndex', 'GATS3v', 'MID_h', 'NssS', 'VE2_Dzpe', 'ETA_eta', 'MWC06', 'SdO', 'GeomDiameter', 'SMR_VSA2', 'SM1_Dzp', 'EState_VSA2', 'GGI1', 'n8aRing', 'SdsssP', 'FNSA3', 'ATSC1m', 'AATS1v', 'MIC4', 'ATSC6pe', 'AXp-2d', 'PPSA5', 'Xch-7dv', 'AETA_eta_RL', 'GATS2Z', 'AATS2s', 'BCUTi-1h', 'AATSC5are', 'MATS5s', 'ATS0m', 'ATSC3m', 'Xp-4d', 'MZ', 'ATS3p', 'VR3_Dzpe', 'SddC', 'MATS1se', 'n10FRing', 'SpMAD_D', 'MATS4s', 'n12FRing', 'VE2_A', 'n4HRing', 'ATSC6m', 'Xc-4d', 'GATS2are', 'WNSA5', 'AATSC4are', 'NsI', 'ATS6i', 'TIC2', 'n6FaRing', 'SpDiam_Dzpe', 'BertzCT', 'ATS2i', 'ATSC2m', 'PPSA3', 'NssPbH2', 'LogEE_A', 'AATS5v', 'nBondsD', 'DPSA4', 'SsGeH3', 'SMR_VSA6', 'NaaaC', 'GATS1p', 'ATS0dv', 'NaaCH', 'JGT10', 'SddssS', 'LogEE_Dzv', 'ATSC3dv', 'SssS', 'SpMax_Dzm', 'SpAbs_DzZ', 'ATSC1v', 'ATS0d', 'VR2_Dzi', 'MATS3m', 'ATSC0d', 'ETA_shape_y', 'NaasN', 'ATS0i', 'ATSC4se', 'AETA_eta_B', 'ATS1i', 'SdNH', 'AATSC1se', 'GATS7m', 'AATSC5se', 'n7FAHRing', 'AATS4are', 'n6HRing', 'VSA_EState8', 'GATS3d', 'AATSC6m', 'ATSC7v', 'SdsN', 'VR2_DzZ', 'WNSA3', 'AATSC2s', 'AATSC1d', 'ETA_eta_R', 'VR3_A', 'ETA_epsilon_1', 'NsOH', 'Xch-4d', 'VAdjMat', 'ATSC6are', 'nH', 'nX', 'SpDiam_Dzm', 'VE2_Dzare', 'VR1_D', 'n8FaHRing', 'VR1_DzZ', 'Xch-5dv', 'AATS3i', 'SdssSe', 'VR3_Dzv', 'SddsN', 'n4FAHRing', 'NsNH3', 'C1SP2', 'SMR_VSA3', 'Kier2', 'MATS7v', 'SsssGeH', 'n7aHRing', 'FNSA1', 'NaaN', 'ZMIC0', 'BCUTare-1h', 'piPC3', 'AATSC7d', 'VR1_A', 'SRW03', 'MATS1v', 'MID_N', 'SM1_Dzm', 'NssssSn', 'NssAsH', 'GATS3dv', 'LogEE_Dzp', 'MATS5Z', 'GATS6are', 'AATS3d', 'ATSC0se', 'SpAD_Dzpe', 'VR1_Dzare', 'n11FARing', 'Xp-1d', 'MATS6v', 'GGI2', 'ATS3se', 'VR3_D', 'ATS7pe', 'AATS2i', 'MWC01', 'n8FARing', 'FNSA2', 'ATSC5s', 'GATS2i', 'SpAD_Dzm', 'AXp-0d', 'RotRatio', 'MATS7s', 'SssCH2', 'TSRW10', 'Mi', 'WPSA3', 'MATS3dv', 'C1SP3', 'ATSC2se', 'VR1_Dzpe', 'BIC1', 'GeomRadius', 'AATS2se', 'Sp', 'ATS6Z', 'MATS2are', 'VR3_Dzm', 'MATS5i', 'MATS1Z', 'n4FaHRing', 'WNSA1', 'TIC0', 'MATS5are', 'SsssdAs', 'AATS0are', 'AATSC5c', 'MIC1', 'AATSC1are', 'ATSC4are', 'BCUTv-1h', 'ATSC4v', 'TIC1', 'nHRing', 'TASA', 'MATS3p', 'SM1_Dzse', 'MATS1are', 'ATS1v', 'MATS6d', 'ATS0se', 'Zagreb1', 'BIC2', 'ATSC7s', 'AATSC1v', 'ATS2m', 'AXp-3dv', 'NsssN', 'SLogP', 'VE1_D', 'n9AHRing', 'ATSC2d', 'nSpiro', 'BCUTc-1h', 'CIC1', 'EState_VSA10', 'AATS2v', 'GATS7i', 'ATS6m', 'MATS4i', 'ETA_epsilon_3', 'Xp-4dv', 'AATS2Z', 'AATSC4v', 'SssBH', 'SdssS', 'n7HRing', 'AATSC6are', 'SRW10', 'nB', 'n5aHRing', 'GGI10', 'n10FAHRing', 'ATS2p', 'SsF', 'AATS4v', 'ATSC4i', 'SpMax_DzZ', 'AATSC3v', 'ATS8se', 'SsBr', 'AATS0s', 'MATS6Z', 'NssGeH2', 'ATS7i', 'ETA_eta_L', 'LogEE_Dzare', 'SpMAD_Dzm', 'n12aRing', 'ATS7Z', 'nBondsKS', 'AATSC3c', 'EState_VSA3', 'VE1_A', 'n10ARing', 'BIC3', 'n4FHRing', 'ATS8s', 'ATS7dv', 'AATS6d', 'n7ARing', 'ATS2are', 'NtCH', 'AATSC4se', 'VR3_Dzi', 'MWC02', 'AETA_eta_FL', 'AATS3v', 'MATS2dv', 'AMID_N', 'ETA_dEpsilon_D', 'AATS2are', 'AATSC0v', 'n3aRing', 'SsSiH3', 'GATS1d', 'ATSC4p', 'GATS1v', 'SssssB', 'MATS7se', 'ATS0Z', 'ETA_dBeta', 'FPSA2', 'MATS2se', 'Kier1', 'Xpc-5dv', 'SpAbs_Dzm', 'MATS4pe', 'SsssNH', 'ATS0v', 'ATSC5d', 'ATS6pe', 'AATS6m', 'ETA_eta_FL', 'GATS2se', 'SpMax_Dzp', 'NsssdAs', 'GATS6Z', 'n4AHRing', 'Xpc-4d', 'AATS7p', 'MIC5', 'C2SP1', 'AATSC4Z', 'AATSC0dv', 'ZMIC3', 'Xc-6dv', 'nBonds', 'SIC4', 'nG12HRing', 'NdCH2', 'SsssB', 'AATSC3i', 'AMID_C', 'ATSC3p', 'n8FRing', 'GATS2c', 'AMID_h', 'AATS2p', 'VR3_Dzse', 'GGI7', 'SMR_VSA9', 'AATSC3se', 'nFARing', 'VR2_Dzse', 'PEOE_VSA4', 'NssSiH2', 'n3ARing', 'AATS2m', 'NssNH', 'GATS3c', 'nG12ARing', 'Mpe', 'ATSC1are', 'Mare', 'MATS6p', 'VE3_D', 'n10FHRing', 'ETA_eta_F', 'SsssssP', 'GATS6v', 'SssssSi', 'LabuteASA', 'BIC5', 'MATS2pe', 'AATSC5dv', 'MWC03', 'ZMIC2', 'PetitjeanIndex', 'GATS5v', 'NdssS', 'MATS7Z', 'nBridgehead', 'BCUTm-1l', 'ATS4s', 'AATS5are', 'nG12FaHRing', 'ATSC3are', 'n9FaHRing', 'BCUTi-1l', 'nAromBond', 'AATSC2i', 'AATSC5Z', 'PNSA2', 'NsSiH3', 'n11HRing', 'n11aRing', 'JGI2', 'DPSA1', 'SssNH2', 'GATS5Z', 'ETA_beta_ns', 'n11FHRing', 'GATS1se', 'NdsN', 'ATS7v', 'SssPH', 'ATSC6se', 'SsssSnH', 'n6FRing', 'NsBr', 'LogEE_DzZ', 'MATS3s', 'SRW06', 'MATS3d', 'MATS5dv', 'n12Ring', 'AATS1i', 'ATSC6Z', 'nCl', 'ATSC6p', 'PEOE_VSA1', 'WNSA4', 'ATSC7m', 'ATSC8m', 'GATS6i', 'SaaNH', 'AATS1p', 'RPSA', 'n3HRing', 'AATSC7c', 'VE1_Dzse', 'LogEE_Dzse', 'IC5', 'ATSC2c', 'ATSC6i', 'SsssAs', 'AATSC2v', 'AATS4d', 'VE2_Dzv', 'MATS3c', 'VSA_EState3', 'n6FAHRing', 'NsssNH', 'AATS6p', 'EState_VSA8', 'ETA_eta_BR', 'piPC2', 'ATSC3c', 'ATS3d', 'ATSC5c', 'ATSC8i', 'n11FRing', 'GATS6c', 'AATSC0m', 'GATS1Z', 'ATS5dv', 'n9FRing', 'ATSC7p', 'GATS5i', 'ATS2pe', 'Xc-5dv', 'SlogP_VSA3', 'AATSC5s', 'n10AHRing', 'BCUTp-1l', 'MATS7dv', 'AATSC3are', 'GATS4Z', 'MIC0', 'MIC3', 'ATSC4Z', 'n12aHRing', 'GATS2p', 'nBondsA', 'GATS7p', 'n7AHRing', 'SaasC', 'Xp-3dv', 'SlogP_VSA5', 'AXp-6dv', 'SRW07', 'n12FaHRing', 'JGI1', 'Xc-3d', 'ATSC6d', 'AATSC1p', 'nI', 'Xch-5d', 'SsOH', 'NsLi', 'ATS3pe', 'AATSC1c', 'SdsCH', 'GATS4are', 'AATS5p', 'Xp-5d', 'GATS4d', 'GATS7Z', 'nARing', 'MATS4p', 'ETA_shape_p', 'AATS5s', 'Si', 'GATS2d', 'AATSC4p', 'AATS6s', 'AATSC3p', 'n4FaRing', 'ATS7d', 'SRW05', 'GATS7are', 'NtN', 'BCUTpe-1l', 'VE3_Dzare', 'ATSC2are', 'MATS2c', 'Xpc-5d', 'Xp-1dv', 'AATS0dv', 'n12FARing', 'NdsssP', 'WPSA5', 'Mv', 'MPC2', 'MDEC-22', 'n4ARing', 'ATS5Z', 'VE2_DzZ', 'AATSC3pe', 'MWC09', 'FPSA5', 'GATS5p', 'n9aRing', 'NdS', 'AATS1pe', 'NssssBe', 'AATS5i', 'ATSC2i', 'VSA_EState5', 'n5aRing', 'AATS4pe', 'AATS1Z', 'piPC7', 'SpMax_D', 'PEOE_VSA12', 'GATS6d', 'MW', 'ATS4pe', 'ATS0are', 'PPSA1', 'AXp-2dv', 'Xp-6dv', 'ATSC4m', 'MATS7m', 'GGI3', 'AATS3dv', 'ATS4se', 'ATS5pe', 'WPSA1', 'AATS3are', 'ATSC1i', 'GGI6', 'SpAD_Dzp', 'MPC3', 'EState_VSA6', 'AATS6dv', 'n3AHRing', 'SpMAD_DzZ', 'ATS5se', 'AATS0Z', 'GATS4p', 'nHBAcc', 'ATS1s', 'AATS7v', 'AATSC7are', 'GATS2dv', 'AATSC7se', 'piPC8', 'SaasN', 'VE2_Dzi', 'MATS6are', 'AXp-4d', 'ATSC6v', 'SlogP_VSA11', 'MATS6m', 'VE3_Dzv', 'AATSC4i', 'ATS1are', 'BalabanJ', 'VE1_DzZ', 'AATS4s', 'n8FHRing', 'NssNH2', 'Mp', 'AATS7s', 'ETA_eta_RL', 'MOMI-X', 'VSA_EState6', 'MATS6c', 'Xp-6d', 'NaaO', 'RPCG', 'BCUTpe-1h', 'GATS2pe', 'IC0', 'ATSC2pe', 'AATSC7i', 'AXp-1dv', 'MPC7', 'n4FARing', 'PEOE_VSA11', 'SpAbs_Dzpe', 'VR2_Dzv', 'BCUTare-1l', 'ETA_beta_s', 'AXp-7dv', 'AATS3s', 'piPC6', 'CIC5', 'Xch-3d', 'BCUTd-1l', 'AATSC4d', 'ATSC7c', 'VE2_Dzse', 'ATSC7Z', 'AATS5pe', 'MATS7p', 'ATS6se', 'ATSC0pe', 'ATSC1pe', 'AATS5d', 'nBondsM', 'ATS4v', 'GATS6dv', 'VE2_Dzm', 'n12FHRing', 'NsCl', 'SssSiH2', 'MATS4d', 'AATSC6Z', 'NsssGeH', 'MATS2s', 'apol', 'NsssssAs', 'AATSC0se', 'n5ARing', 'GATS2s', 'ATS0s', 'SdS', 'NsSH', 'n12FAHRing', 'NssssB', 'SpAbs_Dzp', 'MWC10', 'n10FaHRing', 'SM1_Dzi', 'n7FaHRing', 'AETA_eta_F', 'GATS4i', 'AETA_alpha', 'SlogP_VSA4', 'nAHRing', 'PNSA1', 'ATSC8pe', 'SssssC', 'FPSA4', 'IC1', 'CIC3', 'NaasC', 'TPSA', 'SpMax_Dzare', 'nG12FAHRing', 'AATSC5d', 'WPol', 'AATSC0p', 'LogEE_D', 'MPC8', 'SpAD_Dzv', 'AETA_beta_s', 'n5FaHRing', 'ETA_beta', 'ATSC0are', 'MATS3Z', 'SaaN', 'C3SP3', 'AATS0v', 'n10aHRing', 'ATS3m', 'NsSeH', 'MID_O', 'JGI3', 'GATS5s', 'GATS3m', 'AATSC3m', 'CIC4', 'SdssC', 'MATS7i', 'ATS3i', 'TopoShapeIndex', 'GATS7dv', 'GGI4', 'ATSC8are', 'SpAD_Dzi', 'bpol', 'VE3_Dzp', 'MATS4m', 'MDEC-33', 'n3Ring', 'MATS3se', 'AATSC0i', 'VSA_EState1', 'ETA_dAlpha_A', 'SpDiam_A', 'MATS4Z', 'PEOE_VSA8', 'VR2_Dzare', 'NdssSe', 'AATS2d', 'SIC0', 'piPC5', 'Sse', 'AETA_eta_L', 'ATSC8p', 'ATS7p', 'ATS1pe', 'ATSC7dv', 'ATSC0v', 'SsssN', 'MATS4dv', 'GATS3se', 'SsI', 'n10Ring', 'MID', 'SpAD_A', 'n5AHRing', 'GATS3p', 'ATS5s', 'nG12aHRing', 'piPC4', 'SMR_VSA1', 'ATSC3i', 'n7FHRing', 'SpAbs_D', 'n4aRing', 'MWC05', 'ATS6dv', 'SpAbs_Dzv', 'BCUTdv-1l', 'nO', 'AATS4p', 'ATSC8Z', 'ATSC3d', 'AETA_beta_ns_d', 'MID_C', 'SM1_Dzpe', 'ATSC6c', 'GATS7s', 'AATS4se', 'GATS6p', 'ATSC8d', 'AATSC6p', 'MATS3are', 'BCUTdv-1h', 'AATSC7v', 'PEOE_VSA5', 'AATSC2pe', 'NsssCH', 'JGI10', 'NsssB', 'C4SP3', 'NsssPbH', 'NssBe', 'SsLi', 'ETA_dEpsilon_A', 'Xp-3d', 'VSA_EState2', 'NsCH3', 'AATSC1Z', 'SsPH2', 'GATS3i', 'ATS6v', 'VR1_Dzp', 'SRW08', 'PEOE_VSA3', 'ETA_dPsi_A', 'Xp-2dv', 'nG12FRing', 'AATSC5m', 'Xp-0dv', 'ATS8v', 'AATS1d', 'VR1_Dzm', 'SssssGe', 'BCUTm-1h', 'AATS7dv', 'SpAD_Dzare', 'ATS8pe', 'SssO', 'AATSC6i', 'SIC5', 'SlogP_VSA1', 'PEOE_VSA9', 'n4Ring', 'ATSC4d', 'AATS1dv', 'MATS5pe', 'ETA_dEpsilon_C', 'VR2_Dzm', 'nP', 'NddssSe', 'n10HRing', 'AATS1se', 'SssssBe', 'BCUTZ-1l', 'GATS5d', 'Spe', 'AATSC1pe', 'ATS4i', 'nHBDon', 'PPSA2', 'AXp-3d', 'ATSC6s', 'ATSC8dv', 'AATS6se', 'SpAbs_Dzse', 'nFRing', 'n5FARing', 'nG12FARing', 'n8FaRing', 'SMR_VSA5', 'VE1_Dzp', 'MATS1c', 'NsF', 'MATS7d', 'fMF', 'GATS1s', 'ATSC4dv', 'Kier3', 'MATS4v', 'ATS1d', 'AATSC3d', 'ATSC7se', 'ATS7se', 'GATS4se', 'SssBe', 'AATSC2d', 'AATS1are', 'MATS2m', 'BCUTp-1h', 'MATS3pe', 'AETA_beta', 'nC', 'MATS2v', 'AATSC7dv', 'IC3', 'IC4', 'GATS5m', 'WNSA2', 'GATS7se', 'MATS2p', 'ATSC5pe', 'fragCpx', 'AATSC7s', 'AATSC6d', 'EState_VSA9', 'VE2_D', 'ZMIC5', 'BCUTs-1h', 'ATSC5m', 'SsssssAs', 'C1SP1', 'GATS7c', 'AATSC4c', 'ATSC7are', 'n4aHRing', 'n6FHRing', 'AATSC1m', 'ATS4d', 'BCUTs-1l', 'IC2', 'ETA_dEpsilon_B', 'NsPH2', 'AATSC0are', 'ATS2d', 'C3SP2', 'GGI9', 'Xp-2d', 'nAcid', 'ATS5v', 'AATSC2p', 'SsssP', 'RASA', 'SpDiam_Dzv', 'ATS7s', 'ATSC5se', 'Xpc-6d', 'ATSC5v', 'MATS5p', 'AATSC7Z', 'AATS1s', 'EState_VSA1', 'ATS1p', 'nG12FaRing', 'ATS4are', 'n6ARing', 'n3aHRing', 'SlogP_VSA10', 'GATS4c', 'VE3_DzZ', 'ATS7m', 'SsssPbH', 'ZMIC1', 'ATSC2v', 'AATS7pe', 'AATS5m', 'ATS8p', 'SaaaC', 'nRot', 'Diameter', 'GeomShapeIndex', 'MDEC-23', 'n10aRing', 'NssCH2', 'ATSC7d', 'VR3_DzZ', 'ATS0p', 'ATS3s', 'ETA_alpha', 'SMR_VSA4', 'GATS6pe', 'SsSnH3', 'Sv', 'nFHRing', 'AATSC2are', 'RPCS', 'AATSC4s', 'ETA_epsilon_4', 'JGI7', 'SssssSn', 'NsGeH3', 'AATSC6dv', 'ATSC4c', 'SMR_VSA8', 'NssssC', 'MATS1p', 'NddsN', 'VR2_Dzp', 'ATS1se', 'AATSC1s', 'CIC2', 'n7FaRing', 'SaaS', 'NaaS', 'VE1_Dzare', 'n9HRing', 'JGI9', 'AMID_X', 'ATSC3v', 'ETA_eta_B']
                df = df[common_columns]
                dataframes[name] = df
            # Calculate the minimum count of non-null values required 
            min_count = int(((100 - threshold) / 100) * df.shape[0] + 1)
            # Drop columns with insufficient non-null values
            df_cleaned = df.dropna(axis=1, thresh=min_count)
            dataframes[name] = df_cleaned
        self.dataframes = dataframes

    def ImputeData(self, method="knn", class_specific = False):
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
                    if class_specific is True:
                        df_imputed = impute_class_specific(df, self.prediction_label)
                    else:
                        imputer = KNNImputer(n_neighbors=5)
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
                normalized_df =  pd.DataFrame(normalize(df, norm='l1'), columns=df.columns,index = df.index)
            elif normalization_type == 'L2':
                normalized_df =  pd.DataFrame(normalize(df, norm='l2'), columns=df.columns,index = df.index)
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




    def fuseFeatures(self, n_components, methods=["pca","CDI"],CDI_dim=8192,CDI_epochs=500,  CDI_k =[10,7,12,5,10,6],save_dir = "ChemicalDice_fusedData",**kwargs):
        """

        The fusion methods are as follows:
           - 'CDI': Chemical Dive Integrator, a autoencoder based feature fusion technique that cross reconstruct data from different modalities and make autoencoders to learn importent features from the data.  Finally the data is converted from its orignal dimention to reduced size of embedding  
           - 'pca': Principal Component Analysis, a linear dimensionality reduction technique that projects the data onto a lower-dimensional subspace that maximizes the variance.
           - 'ica': Independent Component Analysis, a linear dimensionality reduction technique that separates the data into independent sources based on the assumption of statistical independence.
           - 'ipca': Incremental Principal Component Analysis, a variant of PCA that allows for online updates of the components without requiring access to the entire dataset.
           - 'cca': Canonical Correlation Analysis, a linear dimensionality reduction technique that finds the linear combinations of two sets of variables that are maximally correlated with each other.
           - 'plsda': Partial Least Squares Discriminant Analysis, a supervised dimensionality reduction technique that finds the linear combinations of the features that best explain both the variance and the correlation with the target variable.
           - 'tsne': t-distributed Stochastic Neighbor Embedding, a non-linear dimensionality reduction technique that preserves the local structure of the data by embedding it in a lower-dimensional space with a probability distribution that matches the pairwise distances in the original space.
           - 'kpca': Kernel Principal Component Analysis, a non-linear extension of PCA that uses a kernel function to map the data into a higher-dimensional feature space where it is linearly separable.
           - 'rks': Random Kitchen Sinks, a non-linear dimensionality reduction technique that uses random projections to approximate a kernel function and map the data into a lower-dimensional feature space.
           - 'SEM': Spectral Embedding is a dimensionality reduction technique that uses the eigenvalues of a graph Laplacian to map data into a lower-dimensional space. It preserves the local and global structure of the data by capturing its intrinsic geometry. This method is particularly effective for non-linear dimensionality reduction and manifold learning
           - 'isomap': Isometric Mapping, a non-linear dimensionality reduction technique that preserves the global structure of the data by embedding it in a lower-dimensional space with a geodesic distance that approximates the shortest path between points in the original space.
           - 'lle': Locally Linear Embedding, a non-linear dimensionality reduction technique that preserves the local structure of the data by reconstructing each point as a linear combination of its neighbors and embedding it in a lower-dimensional space that minimizes the reconstruction error.
           - 'tensordecompose': Tensor Decomposition, a technique that decomposes a multi-dimensional array (tensor) into a set of lower-dimensional arrays (factors) that capture the latent structure and interactions of the data.
        
        Fuse the features of multiple dataframes using different methods.

        :param n_components: The number of components use for the fusion.
        :type n_components: int
        :param method: The method to use for the fusion. It can be one of these: 'pca', 'ica', 'ipca', 'cca', 'tsne', 'kpca', 'rks', 'SEM', 'isomap', 'lle', 'autoencoder', 'plsda', or 'tensordecompose'.
        :type methods: list
        :param CDI_dim: The output dimension for the 'CDI' method. This determines the size of the final embedding.
        :type CDI_dim: int
        :param CDI_epochs: The number of epochs for training the autoencoder in the 'CDI' method.
        :type CDI_epochs: int
        :param CDI_k: A list representing the reduction in the number of nodes from the input layer to the subsequent layers, for six input feature matrix in the 'CDI' method.
        :type CDI_k: list
        :param save_dir: The directory where the fused data will be saved.
        :type save_dir: str
        :param kwargs: Additional arguments for specific fusion methods.
        :type kwargs: dict

        :raises: ValueError if the method is not one of the valid options.
        """
        
        try:
            os.mkdir(save_dir)
        except:
            print(save_dir," already exists")
        

        methods_chemdices = ['pca', 'ica', 'ipca', 'cca', 'tsne', 'kpca', 'rks', 'SEM', 'autoencoder', 'tensordecompose', 'plsda',"CDI"]   
        valid_methods_chemdices = [method for method in methods if method in methods_chemdices]
        invalid_methods_chemdices = [method for method in methods if method not in methods_chemdices]
        methods_chemdices_text = ",".join(methods_chemdices)
        invalid_methods_chemdices_text = ",".join(invalid_methods_chemdices)
        if len(invalid_methods_chemdices):
            raise ValueError(f"These methods are invalid:{invalid_methods_chemdices_text}\n Valid methods are : {methods_chemdices_text}")
        dataframe = self.dataframes
        for method in valid_methods_chemdices:
            print(method)
            # Iterate through the dictionary and fusion DataFrame
            # train data fusion
            df_list = []
            for name, df in dataframe.items():
                df_list.append(df)
            if method in ['pca', 'ica', 'ipca']:
                merged_df = pd.concat(df_list, axis=1)
                fused_df1 = apply_analysis_linear1(merged_df, analysis_type=method, n_components=n_components, **kwargs)
                fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(n_components)+".csv"))
            elif method in ['cca']:
                all_combinations = []
                all_combinations.extend(combinations(df_list, 2))
                all_fused =[]
                n=0
                for df_listn in all_combinations:
                    fused_df_t = ccafuse(df_listn[0], df_listn[1],n_components)
                    all_fused.append(fused_df_t)
                fused_df1 = pd.concat(all_fused, axis=1, sort=False)
                fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(n_components)+".csv"))

            elif method in ['tsne', 'kpca', 'rks', 'SEM']:
                merged_df = pd.concat(df_list, axis=1)
                fused_df1 = apply_analysis_nonlinear1(merged_df,
                                                    analysis_type=method,
                                                    n_components=n_components,
                                                    **kwargs)
                fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(n_components)+".csv"))

            elif method in ['isomap', 'lle']:
                merged_df = pd.concat(df_list, axis=1)
                fused_df1 = apply_analysis_nonlinear2(merged_df, analysis_type=method, n_neighbors=5, n_components=n_components, **kwargs)
                fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(n_components)+".csv"))

            elif method in ['autoencoder']:
                merged_df = pd.concat(df_list, axis=1)
                fused_df1 = apply_analysis_nonlinear3(merged_df,
                                                    analysis_type=method,
                                                    lr=0.001,
                                                    num_epochs = 20, 
                                                    hidden_sizes=[128, 64, 36, 18],
                                                    **kwargs)
                fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(n_components)+".csv"))

            elif method in ['CDI']:
                scaler =StandardScaler()
                if self.training_AER_model is None or self.AER_model_embt_size != CDI_dim:
                    df_list2=[None,None,None,None,None,None]
                    for name, df in self.dataframes.items():
                        if name.lower() == "mopac":
                            df_list2[0] = df.copy()
                        elif name.lower() == "chemberta":
                            df_list2[1] = df.copy()
                        elif name.lower() ==  "mordred":
                            df_list2[2] = df.copy()
                        elif name.lower() ==  "signaturizer":
                            df_list2[3] = df.copy()
                        elif name.lower() ==  "imagemol":
                            df_list2[4] = df.copy()
                        elif name.lower() ==  "grover":
                            df_list2[5] = df.copy()
                    if type(CDI_dim) == list:
                        embd = 8192
                        #print(df_list2)
                        embeddings_8192 = AutoencoderReconstructor_training_8192(df_list2[0], df_list2[1], df_list2[2], df_list2[3],df_list2[4],df_list2[5],CDI_epochs,CDI_k)
                        fused_df_unstandardized = embeddings_8192
                        fused_df_unstandardized.set_index("id",inplace =True)
                        fused_df1 = pd.DataFrame(scaler.fit_transform(fused_df_unstandardized), index=fused_df_unstandardized.index, columns=fused_df_unstandardized.columns)
                        if 8192 in CDI_dim:
                            fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(embd)+".csv"))
                            CDI_dim.remove(8192)
                        for embd in CDI_dim:
                            embeddings_df = AutoencoderReconstructor_training_other(df_list2[0], df_list2[1], df_list2[2], df_list2[3],df_list2[4],df_list2[5], embd,CDI_epochs,CDI_k)
                            fused_df_unstandardized = embeddings_df
                            fused_df_unstandardized.set_index("id",inplace =True)
                            fused_df1 = pd.DataFrame(scaler.fit_transform(fused_df_unstandardized), index=fused_df_unstandardized.index, columns=fused_df_unstandardized.columns)
                            fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(embd)+".csv"))
                    elif type(CDI_dim) == int:
                        embd=CDI_dim
                        embeddings_df,model_wt = AutoencoderReconstructor_training_single(df_list2[0], df_list2[1], df_list2[2], df_list2[3],df_list2[4],df_list2[5],CDI_dim,CDI_epochs,CDI_k)
                        fused_df_unstandardized = embeddings_df
                        fused_df_unstandardized.set_index("id",inplace =True)
                        fused_df1 = pd.DataFrame(scaler.fit_transform(fused_df_unstandardized), index=fused_df_unstandardized.index, columns=fused_df_unstandardized.columns)
                        fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(embd)+".csv"))
                    else:
                        raise ValueError("CDI_dim should be  int or list")

                else:
                    print("CDI model Training")

                    
            prediction_label = self.prediction_label
            if method in ['plsda']:
                df_list = []
                for name, df in dataframe.items():
                    df_list.append(df)
                merged_df = pd.concat(df_list, axis=1)
                fused_df1, pls_canonical = apply_analysis_linear2(merged_df, prediction_label, analysis_type=method, n_components=n_components, **kwargs)
                self.pls_model = pls_canonical
                fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(n_components)+".csv"))

            ######################
                
            if method in ['tensordecompose']:
                df_list = []
                for name, df in dataframe.items():
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
                fused_df1.to_csv(os.path.join(save_dir,"fused_data_"+method+"_"+str(n_components)+".csv"))
            print("Data is fused and saved to  ChemicalDice_fusedData")


    def evaluate_fusion_models_nfold(self, folds, task_type, fused_data_path="ChemicalDice_fusedData", models = None,save_model = True):
        """
        Perform n-fold cross-validation on fusion models and save the evaluation metrics.This method evaluates the performance of various machine learning models on fused data obtained from ChemDice. It supports both classification and regression tasks and saves the performance metrics for each fold into a CSV file.
        :param folds: The number of folds to use for KFold cross-validation.
        :type folds: int
        :param task_type: The type of task to perform, either 'classification' or 'regression'.
        :type task_type: str
        :param fused_data_path: The path to the directory containing the fused data files, defaults to 'ChemicalDice_fusedData'.
        :type fused_data_path: str
        :param models: The list of model names to evaluate. If None, a default set of models will be used.
        :type models: list[str], optional
        :param save_model: Whether to save the trained models, defaults to True.
        :type save_model: bool, optional
        :raises ValueError: If the `task_type` is neither 'classification' nor 'regression'.
        :return: None

        Available Models for Classification:
            - "Logistic Regression"
            - "Decision Tree"
            - "Random Forest"
            - "Support Vector Machine"
            - "Naive Bayes"
            - "KNN"
            - "MLP"
            - "QDA"
            - "AdaBoost"
            - "Extra Trees"
            - "XGBoost"

        Available Models for Regression:
            - "Linear Regression"
            - "Ridge"
            - "Lasso"
            - "ElasticNet"
            - "Decision Tree"
            - "Random Forest"
            - "Gradient Boosting"
            - "AdaBoost"
            - "Support Vector Machine"
            - "K Neighbors"
            - "MLP"
            - "Gaussian Process"
            - "Kernel Ridge"

        .. note:: The method assumes that the prediction labels are stored in `self.prediction_label`.For classification, it evaluates models based on AUC, accuracy, precision, recall, f1 score, balanced accuracy, MCC, and kappa. For regression, it evaluates models based on R2 score, MSE, RMSE, and MAE. The results are saved in a CSV file named 'Accuracy_Metrics_{method_name}.csv' in a directory named '{folds}_fold_CV_results'.

        """
        
        list_of_files = os.listdir(fused_data_path)
        self.task_type = task_type
        self.folds = folds
        if task_type == "classification":
            classifiers = [
                ("Logistic Regression", LogisticRegression()),
                ("Decision Tree", DecisionTreeClassifier()),
                ("Random Forest", RandomForestClassifier()),
                ("Support Vector Machine", SVC(probability=True)),
                ("Naive Bayes", GaussianNB()),
                ("KNN", KNeighborsClassifier(leaf_size=1, n_neighbors=11, p=3, weights='distance')),
                ("MLP", MLPClassifier(alpha=1, max_iter=1000)),
                ("QDA", QuadraticDiscriminantAnalysis()),
                ("AdaBoost", AdaBoostClassifier()),
                ("Extra Trees", ExtraTreesClassifier()),
                ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
            ]
            if models == None:
                models =classifiers
            else:
                models = [clf for clf in classifiers if clf[0] in models]
        elif task_type == "regression":
                regressors = [
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
                if models == None:
                    models = regressors
                else:
                    models = [regg for regg in regressors if regg[0] in models]
        else:
            raise ValueError("task_type can be either 'classification' or 'regression'")
        
        if len(models) == 0 :
            raise ValueError("models given are invalid")

        try:
            os.mkdir(f"{folds}_fold_CV_results")
        except:
            print(f"{folds}_fold_CV_results already exists")

        for file in list_of_files:
            y = self.prediction_label
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)
            #print(file)
            #print(1)
            if task_type == "classification":
                metrics = {
                    "Model type": [],
                    "Fold": [],
                    "Model": [],
                    "AUC": [],
                    "Accuracy": [],
                    "Precision": [],
                    "Recall": [],
                    "f1 score": [],
                    "Balanced accuracy": [],
                    "MCC": [],
                    "Kappa": []
                }
            elif task_type == "regression":
                metrics = {
                    "Model type": [],
                    "Model": [],
                    "Fold": [],
                    "R2 Score": [],
                    "MSE": [],
                    "RMSE": [],
                    "MAE": []
                }


            fold_number = 0

            for train_index, test_index in kf.split(y):
                #print(2)
                fold_number += 1
                if file.startswith("fused_data"):
                    method_chemdice = file.replace("fused_data_", "")
                    method_chemdice= method_chemdice.replace(".csv","")
                    # print(method_chemdice)
                    if method_chemdice.startswith("plsda"):
                        #print("Method name",method_chemdice)
                        n_components = method_chemdice.replace("plsda_","")
                        n_components = int(n_components.replace(".csv",""))
                        train_dataframes, test_dataframes, train_label, test_label  = save_train_test_data_n_fold(self.dataframes, self.prediction_label, train_index, test_index, output_dir="comaprision_data_fold_"+str(fold_number)+"_of_"+str(fold_number))
                        X_train = self.fuseFeaturesTrain_plsda(n_components = n_components,  method=method_chemdice, train_dataframes = train_dataframes, train_label =train_label)
                        y_train = train_label
                        X_test = self.fuseFeaturesTest_plsda(n_components = n_components,  method=method_chemdice,test_dataframes = test_dataframes)
                        y_test = test_label
                        # print("y_test")
                        # print(y_test)
                    elif method_chemdice.startswith("tensordecompose"):
                        #print("Method name",method_chemdice)
                        n_components = method_chemdice.replace("tensordecompose_","")
                        n_components = int(n_components.replace(".csv",""))
                        train_dataframes, test_dataframes, train_label, test_label  = save_train_test_data_n_fold(self.dataframes, self.prediction_label, train_index, test_index, output_dir="comaprision_data_fold_"+str(fold_number)+"_of_"+str(fold_number))
                        X_train = self.fuseFeaturesTrain_td(n_components = n_components,  method=method_chemdice, train_dataframes = train_dataframes, train_label =train_label)
                        y_train = train_label
                        X_test = self.fuseFeaturesTest_td(n_components = n_components,  method=method_chemdice,test_dataframes = test_dataframes)
                        y_test = test_label
                        # print("y_test")
                        # print(y_test)
                    
                    else:
                        data = pd.read_csv(os.path.join(fused_data_path, file),index_col=0)
                        X = data
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y[train_index], y[test_index]

                    for name, model in models:


                        if method_chemdice.startswith(('pca', 'ica', 'ipca', 'cca', 'plsda')):
                            metrics["Model type"].append("linear")
                        else:
                            metrics["Model type"].append("Non-linear")
                        

                        name = f"{method_chemdice} {name}"
                        # print(X_train)
                        # print(y_train)
                        # Fit the model on the train set
                        model.fit(X_train, y_train)
                        # if save_model ==True:
                        #     pickle.dump(model, open(f"{folds}_fold_CV_results/Model_{name.replace(" ","_")}.pkl", 'wb'))

                        if task_type == "classification":
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
                            #print(metrics)
                        else:
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
                            #print(metrics)
            #print(metrics)
            #print(metrics)
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(f"{folds}_fold_CV_results/Accuracy_Metrics_{method_chemdice}.csv", index=False)
            print("Done")
            print(f"{folds}_fold_CV_results/Accuracy_Metrics_{method_chemdice}.csv saved")

    

    def evaluate_fusion_models_scaffold_split(self, split_type, task_type, fused_data_path="ChemicalDice_fusedData",models=None,save_model =True):
        """
        Perform n-fold cross-validation on fusion models and save the evaluation metrics. This method evaluates the performance of various machine learning models on fused data obtained from ChemDice. It supports both classification and regression tasks and saves the performance metrics for each fold into a CSV file.
        
        :param split_type: The type scaffold dsta split to perform. Three types available  'random', 'balanced' or 'simple'
        :type folds: str
        :param task_type: The type of task to perform, either 'classification' or 'regression'.
        :type task_type: str
        :param fused_data_path: The path to the directory containing the fused data files, defaults to 'ChemicalDice_fusedData'.
        :type fused_data_path: str
        :param models: The list of model names to evaluate. If None, a default set of models will be used.
        :type models: list[str], optional
        :param save_model: Whether to save the trained models, defaults to True.
        :type save_model: bool, optional
        :raises ValueError: If the `task_type` is neither 'classification' nor 'regression'.
        :return: None

        Available Models for Classification:
            - "Logistic Regression"
            - "Decision Tree"
            - "Random Forest"
            - "Support Vector Machine"
            - "Naive Bayes"
            - "KNN"
            - "MLP"
            - "QDA"
            - "AdaBoost"
            - "Extra Trees"
            - "XGBoost"

        Available Models for Regression:
            - "Linear Regression"
            - "Ridge"
            - "Lasso"
            - "ElasticNet"
            - "Decision Tree"
            - "Random Forest"
            - "Gradient Boosting"
            - "AdaBoost"
            - "Support Vector Machine"
            - "K Neighbors"
            - "MLP"
            - "Gaussian Process"
            - "Kernel Ridge"
            
        .. note:: The method assumes that the prediction labels are stored in `self.prediction_label`. For classification, it evaluates models based on AUC, accuracy, precision, recall, f1 score, balanced accuracy, MCC, and kappa. For regression, it evaluates models based on R2 score, MSE, RMSE, and MAE. The results are saved in a CSV file named 'Accuracy_Metrics_{method_chemdice}.csv' in a directory named 'scaffold_split_results'.


        """
        list_of_files = os.listdir(fused_data_path)
        self.task_type = task_type
        try:
            os.mkdir("scaffold_split_results")
        except:
            print("scaffold_split_results already exist")

        if task_type == "classification":
            classifiers = [
                ("Logistic Regression", LogisticRegression(), {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['saga']
                }),
                ("Decision Tree", DecisionTreeClassifier(), {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }),
                ("Random Forest", RandomForestClassifier(), {
                    'n_estimators': [50, 100, 200, 300, 400],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }),
                ("Support Vector Machine", SVC(probability=True), {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': [2, 3, 4],
                    'gamma': ['scale', 'auto']
                }),
                ("Naive Bayes", GaussianNB(), {
                    'var_smoothing': [1e-9, 1e-8, 1e-7]
                }),
                ("KNN", KNeighborsClassifier(leaf_size=1, n_neighbors=11, p=3, weights='distance'), {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'leaf_size': [1, 10, 30],
                    'p': [1, 2, 3]
                }),
                ("MLP", MLPClassifier(alpha=1, max_iter=1000), {
                    'hidden_layer_sizes': [(10,), (20,), (50,)],
                    'activation': ['relu', 'logistic', 'tanh'],
                    'solver': ['adam', 'sgd'],
                    'alpha': [0.0001, 0.001, 0.01, 1]
                }),
                ("QDA", QuadraticDiscriminantAnalysis(), {
                    'reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                }),
                ("AdaBoost", AdaBoostClassifier(), {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1]
                }),
                ("Extra Trees", ExtraTreesClassifier(), {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'max_features': ['sqrt', 'log2', None]
                }),
                ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.5, 0.7, 1.0],
                    'colsample_bytree': [0.5, 0.7, 1.0]
                })
            ]
            if models ==None:
                models = classifiers
            else:
                models = [clf for clf in classifiers if clf[0] in models]
        # Define models and their respective parameter grids
        elif task_type == "regression":
            regressors = [
                ("MLP", MLPRegressor(), {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001, 0.01]}),
                ("Kernel Ridge", KernelRidge(), {'alpha': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': [0.1, 1.0, 10.0]}),
                ("Linear Regression", LinearRegression(), {'fit_intercept': [True, False]}),
                ("Ridge", Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
                ("Lasso", Lasso(), {'alpha': [0.1, 1.0, 10.0]}),
                ("ElasticNet", ElasticNet(), {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]}),
                ("Decision Tree", DecisionTreeRegressor(), {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}),
                ("Random Forest", RandomForestRegressor(), {'n_estimators': [50, 100, 200, 300, 400], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}),
                ("Gradient Boosting", GradientBoostingRegressor(), {'n_estimators': [50, 100, 200, 300, 400], 'max_depth': [3, 5, 7]}),
                ("AdaBoost", AdaBoostRegressor(), {'n_estimators': [50, 100, 200, 300, 400], 'learning_rate': [0.01, 0.1, 1.0]}),
                ("Support Vector Machine", SVR(), {'kernel': ['linear', 'rbf'], 'C': [0.1, 1.0, 10.0], 'epsilon': [0.1, 0.01, 0.001]}),
                ("K Neighbors", KNeighborsRegressor(), {'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance']}),
                ("Gaussian Process", GaussianProcessRegressor(), {'alpha': [1e-10, 1e-9, 1e-8],'kernel': [RBF(), Matern()] })
            ]
            if models==None:
                models = regressors
            else:
                models = [regg for regg in regressors if regg[0] in models]
        else:
            raise ValueError("task_type can be either 'classification' or 'regression'")
        
        print(len(models))
        if len(models) == 0:
            raise ValueError("models provided are not valid")

        for file in list_of_files:
            method_chemdice = file.replace("fused_data_", "")
            method_chemdice= method_chemdice.replace(".csv","")
            # print(method_chemdice)
            y = self.prediction_label

            if task_type == "classification":
                test_metrics = {
                    "Model": [],
                    "Model type":[],
                    "AUC": [],
                    "Accuracy": [],
                    "Precision": [],
                    "Recall": [],
                    "f1 score":[],
                    "Balanced accuracy":[],
                    "MCC":[],
                    "Kappa":[],
                }

                train_metrics = {
                    "Model": [],
                    "Model type":[],
                    "AUC": [],
                    "Accuracy": [],
                    "Precision": [],
                    "Recall": [],
                    "f1 score":[],
                    "Balanced accuracy":[],
                    "MCC":[],
                    "Kappa":[],
                }


                val_metrics = {
                    "Model": [],
                    "Model type":[],
                    "AUC": [],
                    "Accuracy": [],
                    "Precision": [],
                    "Recall": [],
                    "f1 score":[],
                    "Balanced accuracy":[],
                    "MCC":[],
                    "Kappa":[],
                }
            elif task_type == "regression":
                test_metrics = {
                    "Model": [],
                    "Model type":[],
                    "R2 Score": [],
                    "MSE": [],
                    "RMSE":[],
                    "MAE":[]
                    }
                
                train_metrics = {
                    "Model": [],
                    "Model type":[],
                    "R2 Score": [],
                    "MSE": [],
                    "RMSE":[],
                    "MAE":[]
                    }

                val_metrics = {
                    "Model": [],
                    "Model type":[],
                    "R2 Score": [],
                    "MSE": [],
                    "RMSE":[],
                    "MAE":[]
                    }


            if split_type == "random":
                train_index, val_index, test_index  = random_scaffold_split_train_val_test(index = y.index.to_list(), smiles_list = self.smiles_col.to_list(), seed=0)
            elif split_type == "balanced":
                train_index, val_index, test_index  = scaffold_split_balanced_train_val_test(index = y.index.to_list(), smiles_list = self.smiles_col.to_list(), seed=0)
            elif split_type == "simple":
                train_index, val_index, test_index  = scaffold_split_train_val_test(index = y.index.to_list(), smiles_list = self.smiles_col.to_list(), seed=0)

            print("Method name",method_chemdice)
            if method_chemdice.startswith("plsda"):
                # print("Method name",method_chemdice)
                n_components = method_chemdice.replace("plsda_","")
                n_components = int(n_components.replace(".csv",""))
                train_dataframes,val_dataframe, test_dataframes, train_label, val_label, test_label  = save_train_test_data_s_fold(self.dataframes, self.prediction_label, train_index,val_index, test_index)
                X_train = self.fuseFeaturesTrain_plsda(n_components = n_components,  method=method_chemdice, train_dataframes = train_dataframes, train_label =train_label)
                y_train = train_label
                X_test = self.fuseFeaturesTest_plsda(n_components = n_components,  method=method_chemdice,test_dataframes = test_dataframes)
                y_test = test_label
                X_val = self.fuseFeaturesTest_plsda(n_components = n_components,  method=method_chemdice,test_dataframes = val_dataframe)
                y_val = val_label
            elif  method_chemdice.startswith("tensordecompose"):
                # print("Method name",method_chemdice)
                n_components = method_chemdice.replace("tensordecompose_","")
                n_components = int(n_components.replace(".csv",""))
                train_dataframes,val_dataframe, test_dataframes, train_label, val_label, test_label  = save_train_test_data_s_fold(self.dataframes, self.prediction_label, train_index,val_index, test_index)
                X_train = self.fuseFeaturesTrain_td(n_components = n_components,  method=method_chemdice, train_dataframes = train_dataframes, train_label =train_label)
                y_train = train_label
                X_test = self.fuseFeaturesTest_td(n_components = n_components,  method=method_chemdice,test_dataframes = test_dataframes)
                y_test = test_label
                X_val = self.fuseFeaturesTest_td(n_components = n_components,  method=method_chemdice,test_dataframes = val_dataframe)
                y_val = val_label
            else:
                # print("Method name",method_chemdice)
                data = pd.read_csv(os.path.join(fused_data_path, file),index_col=0)
                X = data

                X_train, X_test, X_val = X.loc[train_index], X.loc[test_index], X.loc[val_index]
                y_train, y_test, y_val = y[train_index], y[test_index], y[val_index]


            parameters_dict = {model_name: [] for model_name, _, _ in models}
            best_parameters_dict = {}


            # Iterate over models
            for model_name, model, param_grid in models:
                # Iterate over hyperparameters
                for params in ParameterGrid(param_grid):
                    # Initialize and train the model with the current hyperparameters
                    model.set_params(**params)
                    model.fit(X_train, y_train)
                    
                    # Evaluate the model on the validation set
                    if  task_type == "classification":
                        y_pred_proba = model.predict_proba(X_val)[:, 1]
                        # Compute the metrics

                        auc = roc_auc_score(y_val, y_pred_proba)
                        accuracy = accuracy_score(y_val, y_pred_proba > 0.5)
                        precision = precision_score(y_val, y_pred_proba > 0.5)
                        recall = recall_score(y_val, y_pred_proba > 0.5)
                        f1 = f1_score(y_val, y_pred_proba > 0.5)
                        baccuracy = balanced_accuracy_score(y_val, y_pred_proba > 0.5)
                        mcc = matthews_corrcoef(y_val, y_pred_proba > 0.5)
                        kappa = cohen_kappa_score(y_val, y_pred_proba > 0.5)
                        # Store metrics and parameters
                        parameters_dict[model_name].append({**params,  
                                                            "auc":auc, 
                                                            'accuracy':accuracy,
                                                            "precision":precision,
                                                            "recall":recall,
                                                            "f1":f1,
                                                            "baccuracy":baccuracy,
                                                            "mcc":mcc,
                                                            "kappa":kappa })

                    elif task_type == "regression":
                        y_pred_val = model.predict(X_val)

                        mse = mean_squared_error(y_val, y_pred_val)
                        mae = mean_absolute_error(y_val, y_pred_val)
                        r2 = r2_score(y_val, y_pred_val)
                        rmse = np.sqrt(mse)

                        # Store metrics and parameters
                        parameters_dict[model_name].append({**params, 'mse': mse, 'mae': mae, 'r2': r2, 'rmse':rmse})

            # Display metrics for each model
            
            for model_name, params_list in parameters_dict.items():
                # for params in params_list:
                #     print(params)

                if task_type == "regression":
                    best_mse_index = max(range(len(params_list)), key=lambda i: params_list[i]['r2'])
                    best_parameters_dict[model_name] = params_list[best_mse_index]
                    measures=  ['mse' , 'mae', 'r2','rmse']
                elif task_type == "classification":
                    best_mse_index = max(range(len(params_list)), key=lambda i: params_list[i]['auc'])
                    best_parameters_dict[model_name] = params_list[best_mse_index]
                    measures = ["auc",
                            'accuracy',
                            "precision",
                            "recall",
                            "f1",
                            "baccuracy",
                            "mcc",
                            "kappa"]
                
                for measure in measures:
                    del best_parameters_dict[model_name][measure]
                    

                print("Best parameters of: " , model_name)
                # print(best_parameters_dict[model_name])
                

            print("Testing with best parameters")
            #print(models)
            for name, model,_ in models:
                
                
                best_parameter = best_parameters_dict[name]
                
                if method_chemdice.startswith(('pca', 'ica', 'ipca', 'cca', 'plsda')):
                    train_metrics["Model type"].append("linear")
                    test_metrics["Model type"].append("linear")
                    val_metrics["Model type"].append("linear")
                else:
                    train_metrics["Model type"].append("Non-linear")
                    test_metrics["Model type"].append("Non-linear")
                    val_metrics["Model type"].append("Non-linear")




                name = method_chemdice+"_"+name
                #print("**************************   "+name + "    **************************")
                
                #print(best_parameter)
                model.set_params(**best_parameter)
                model.fit(X_train, y_train)
                
                # if save_model ==True:
                #     pickle.dump(model, open("scaffold_split_results/Model_"+name.replace(" ","_")+".pkl", 'wb'))
                
                # some time later...
                
                # # load the model from disk
                # loaded_model = pickle.load(open(filename, 'rb'))
                # result = loaded_model.score(X_test, Y_test)
                # print(result)

                if task_type == "regression":

                    #print(" ######## Training data #########")
                    y_pred = model.predict(X_train)
                    # Compute the metrics
                    mse = mean_squared_error(y_train, y_pred)
                    r2 = r2_score(y_train, y_pred)
                    mae = mean_absolute_error(y_train, y_pred)
                    rmse = np.sqrt(mse)
                    # Append the metrics to the dictionary
                    train_metrics["Model"].append(name)
                    train_metrics["MSE"].append(mse)
                    train_metrics["RMSE"].append(rmse)
                    train_metrics["MAE"].append(mae)
                    train_metrics["R2 Score"].append(r2)
                
                    #print(" ######## Validation data #########")
                    y_pred = model.predict(X_val)
                    # Compute the metrics
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    mae = mean_absolute_error(y_val, y_pred)
                    rmse = np.sqrt(mse)
                    # Append the metrics to the dictionary
                    val_metrics["Model"].append(name)
                    val_metrics["MSE"].append(mse)
                    val_metrics["RMSE"].append(rmse)
                    val_metrics["MAE"].append(mae)
                    val_metrics["R2 Score"].append(r2)



                    #print(" ######## Test data #########")
                    y_pred = model.predict(X_test)
                    # Compute the metrics
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    # Append the metrics to the dictionary
                    test_metrics["Model"].append(name)
                    test_metrics["MSE"].append(mse)
                    test_metrics["RMSE"].append(rmse)
                    test_metrics["MAE"].append(mae)
                    test_metrics["R2 Score"].append(r2)
                    # print(test_metrics)

                
                elif task_type == "classification":
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
                    # Store metrics and parameters
                    train_metrics["Model"].append(name)
                    train_metrics["AUC"].append(auc)
                    train_metrics["Accuracy"].append(accuracy)
                    train_metrics["Precision"].append(precision)
                    train_metrics["Recall"].append(recall)
                    train_metrics["f1 score"].append(f1)
                    train_metrics["Balanced accuracy"].append(baccuracy)
                    train_metrics["MCC"].append(mcc)
                    train_metrics["Kappa"].append(kappa)

                    predictions = model.predict(X_val)
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    # Compute the val_metrics
                    auc = roc_auc_score(y_val, y_pred_proba)
                    accuracy = accuracy_score(y_val, y_pred_proba > 0.5)
                    precision = precision_score(y_val, y_pred_proba > 0.5)
                    recall = recall_score(y_val, y_pred_proba > 0.5)
                    f1 = f1_score(y_val, y_pred_proba > 0.5)
                    baccuracy = balanced_accuracy_score(y_val, y_pred_proba > 0.5)
                    mcc = matthews_corrcoef(y_val, y_pred_proba > 0.5)
                    kappa = cohen_kappa_score(y_val, y_pred_proba > 0.5)
                    val_metrics["Model"].append(name)
                    val_metrics["AUC"].append(auc)
                    val_metrics["Accuracy"].append(accuracy)
                    val_metrics["Precision"].append(precision)
                    val_metrics["Recall"].append(recall)
                    val_metrics["f1 score"].append(f1)
                    val_metrics["Balanced accuracy"].append(baccuracy)
                    val_metrics["MCC"].append(mcc)
                    val_metrics["Kappa"].append(kappa)

                    #print(" ######## Validation data #########")
                    # predictions_df = pd.DataFrame({'Predictions': predictions, 'Actual': y_test})
                    # display(predictions_df)
                    # fp_df = predictions_df[(predictions_df['Predictions'] == 1) & (predictions_df['Actual'] == 0)]
                    # print("False Positive")
                    # display(fp_df)
                    # fp_prediction_list.append(fp_df.index.to_list())
                    # # False Negatives (FN): Predicted 0 but Actual 1
                    # fn_df = predictions_df[(predictions_df['Predictions'] == 0) & (predictions_df['Actual'] == 1)]
                    # print("False Negative")
                    # display(fn_df)
                    # fn_prediction_list.append(fn_df.index.to_list())
                    # predictions_dict.update({name:predictions_df})


                    #print(" ######## Test data #########")
                    predictions = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    # Compute the test_metrics
                    auc = roc_auc_score(y_test, y_pred_proba)
                    accuracy = accuracy_score(y_test, y_pred_proba > 0.5)
                    precision = precision_score(y_test, y_pred_proba > 0.5)
                    recall = recall_score(y_test, y_pred_proba > 0.5)
                    f1 = f1_score(y_test, y_pred_proba > 0.5)
                    baccuracy = balanced_accuracy_score(y_test, y_pred_proba > 0.5)
                    mcc = matthews_corrcoef(y_test, y_pred_proba > 0.5)
                    kappa = cohen_kappa_score(y_test, y_pred_proba > 0.5)
                    test_metrics["Model"].append(name)
                    test_metrics["AUC"].append(auc)
                    test_metrics["Accuracy"].append(accuracy)
                    test_metrics["Precision"].append(precision)
                    test_metrics["Recall"].append(recall)
                    test_metrics["f1 score"].append(f1)
                    test_metrics["Balanced accuracy"].append(baccuracy)
                    test_metrics["MCC"].append(mcc)
                    test_metrics["Kappa"].append(kappa)

                    #print("done")
            # print(test_metrics)
            # Convert dictionaries to pandas DataFrames
            test_df = pd.DataFrame(test_metrics)
            train_df = pd.DataFrame(train_metrics)
            val_df = pd.DataFrame(val_metrics)

            
            if task_type == "classification":
                test_df.sort_values(by="AUC", inplace=True, ascending=False)
            else:
                test_df.sort_values(by="R2 Score", inplace=True, ascending=False)

            train_df = train_df.loc[test_df.index]
            val_df = val_df.loc[test_df.index]

            test_df['Data'] = 'test'
            train_df['Data'] = 'train'
            val_df['Data'] = 'valid'

            Accuracy_metrics = pd.concat([test_df, train_df, val_df])


            Accuracy_metrics.to_csv(f"scaffold_split_results/Accuracy_Metrics_{method_chemdice}.csv", index=False)
            print("Done")
            print(f"scaffold_split_results/Accuracy_Metrics_{method_chemdice}.csv saved") 
            print()       



    def get_accuracy_metrics(self,result_dir):
        """
        Retrieve and compile accuracy metrics from cross-validation results.

        This method aggregates the performance metrics from multiple CSV files generated by n-fold cross-validation or scafoold split.
        It calculates the mean performance metrics for each model and sorts the results based on the performance metric
        relevant to the task type (AUC for classification, R2 Score for regression).

        :return: For cross validation result : A tuple containing two DataFrames: `mean_accuracy_metrics` with the mean performance metrics for each model,
                and `Accuracy_metrics` with all the individual fold results. For scaffold split : A tuple containing three DataFrames: `test_metrics`, 
                `train_metrics` and `val_metrics`. 
        :rtype: tuple


        """
        list_of_files = os.listdir(result_dir)
        #print(list_of_files)
        dataframes = []
        for files in list_of_files:
            if files.startswith("Accuracy_Metrics"):
                # print(files)
                files = result_dir+"/"+files
                df = pd.read_csv(files)
                dataframes.append(df)
        # if len(dataframes) == 0:
        #     metrics_df = dataframes[0]
        # else:
        #     pass
        metrics_df = pd.concat(dataframes, ignore_index=True )
        if 'Fold' not in metrics_df.columns:
            # Define the folder path
            # Combine all dataframes into a single dataframe
            
            test_df = metrics_df.loc[metrics_df['Data']=="test"]
            train_df = metrics_df.loc[metrics_df['Data']=="train"]
            val_df = metrics_df.loc[metrics_df['Data']=="valid"]

            test_df = test_df.reset_index(drop=True)
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)

            if 'AUC' in val_df.columns:
                top_models_test = pd.DataFrame()
                test_df.sort_values(by="AUC", inplace=True, ascending=False)
                train_df = train_df.loc[test_df.index]
                val_df = val_df.loc[test_df.index]
                test_df['Method'] = test_df['Model'].apply(lambda x: re.split('_|\d', x)[0])
                methods = test_df['Method'].unique()

                for method in methods:
                    # Filter rows where Model contains the method
                    method_data = test_df[test_df['Model'].str.contains(method)]
                    # Sort by R2 Score in descending order and get the top 3
                    top_method_models = method_data.nlargest(1, 'AUC')
                    # Append to the top_models DataFrame
                    top_models_test = pd.concat([top_models_test, top_method_models])

                top_models_test = top_models_test.sort_values(by='AUC', ascending=False)
                top_models_train = train_df.loc[top_models_test.index]
                top_models_val = val_df.loc[top_models_test.index]

                plot_models_barplot((top_models_train,top_models_val, top_models_test), save_dir=result_dir)
                print("plot saved to ", result_dir)
                return train_df, val_df, test_df
            
            elif 'RMSE' in val_df.columns:
                top_models_test = pd.DataFrame()
                test_df.sort_values(by="R2 Score", inplace=True, ascending=False)
                train_df = train_df.loc[test_df.index]
                val_df = val_df.loc[test_df.index]
                test_df['Method'] = test_df['Model'].apply(lambda x: re.split('_|\d', x)[0])
                methods = test_df['Method'].unique()

                for method in methods:
                    # Filter rows where Model contains the method
                    method_data = test_df[test_df['Model'].str.contains(method)]
                    # Sort by R2 Score in descending order and get the top 3
                    top_method_models = method_data.nlargest(1, 'R2 Score')
                    # Append to the top_models DataFrame
                    top_models_test = pd.concat([top_models_test, top_method_models])

                top_models_test = top_models_test.sort_values(by='R2 Score', ascending=False)
                top_models_train = train_df.loc[top_models_test.index]
                top_models_val = val_df.loc[top_models_test.index]

                plot_models_barplot((top_models_train,top_models_val, top_models_test), save_dir=result_dir)
                print("plot saved to ", result_dir)
                return train_df, val_df, test_df
        else:

            if 'AUC' in metrics_df.columns:
                top_models_test = pd.DataFrame()
                grouped_means = metrics_df.groupby('Model')["AUC"].mean()
                metrics_df = metrics_df.sort_values(by=['Model', 'Fold'], key=lambda x: x.map(grouped_means),ascending=[False, True])
                Accuracy_metrics = metrics_df

                grouped_means = metrics_df.groupby('Model').mean(numeric_only = True).reset_index()
                #metrics_df = grouped_means.drop(columns=['Fold'])
                metrics_df = metrics_df.sort_values(by="AUC",ascending = False)
                mean_accuracy_metrics = metrics_df

                mean_accuracy_metrics['Method'] = mean_accuracy_metrics['Model'].apply(lambda x: re.split('_|\d', x)[0])
                methods = mean_accuracy_metrics['Method'].unique()

                for method in methods:
                    # Filter rows where Model contains the method
                    method_data = mean_accuracy_metrics[mean_accuracy_metrics['Model'].str.contains(method)]
                    # Sort by R2 Score in descending order and get the top 3
                    top_method_models = method_data.nlargest(1, 'AUC')
                    # Append to the top_models DataFrame
                    top_models_test = pd.concat([top_models_test, top_method_models])

                top_models_test = top_models_test.sort_values(by='AUC', ascending=False)
                top_Accuracy_metrics = Accuracy_metrics[ Accuracy_metrics['Model'].isin(top_models_test['Model'].to_list())] 


                plot_models_boxplot(Accuracy_metrics, save_dir=result_dir)
                print("plot saved to ", result_dir)
                return mean_accuracy_metrics, Accuracy_metrics

            elif 'RMSE' in metrics_df.columns:
                top_models_test = pd.DataFrame()
                # Group by 'Model' and calculate the mean of 'R2 Score'
                grouped_means = metrics_df.groupby('Model')["R2 Score"].mean()
                metrics_df = metrics_df.sort_values(by=['Model', 'Fold'], key=lambda x: x.map(grouped_means),ascending=[False, True])
                Accuracy_metrics = metrics_df


                grouped_means = metrics_df.groupby('Model').mean(numeric_only = True).reset_index()
                #metrics_df = grouped_means.drop(columns=['Fold'])
                metrics_df = metrics_df.sort_values(by="R2 Score",ascending = False)
                mean_accuracy_metrics = metrics_df

                mean_accuracy_metrics['Method'] = mean_accuracy_metrics['Model'].apply(lambda x: re.split('_|\d', x)[0])
                methods = mean_accuracy_metrics['Method'].unique()

                for method in methods:
                    # Filter rows where Model contains the method
                    method_data = mean_accuracy_metrics[mean_accuracy_metrics['Model'].str.contains(method)]
                    # Sort by R2 Score in descending order and get the top 3
                    top_method_models = method_data.nlargest(1, 'R2 Score')
                    # Append to the top_models DataFrame
                    top_models_test = pd.concat([top_models_test, top_method_models])

                top_models_test = top_models_test.sort_values(by='R2 Score', ascending=False)
                top_Accuracy_metrics = Accuracy_metrics[ Accuracy_metrics['Model'].isin(top_models_test['Model'].to_list())] 

                plot_models_boxplot(top_Accuracy_metrics, save_dir=result_dir)
                print("plot saved to ", result_dir)
                return mean_accuracy_metrics, Accuracy_metrics       

    def fuseFeaturesTrain_plsda(self, n_components , method, train_dataframes, train_label,**kwargs):
        """
        Internal function for training of plsda fusion method.
        
        """
        prediction_label = train_label
        dataframes1 = train_dataframes
        df_list = []
        for name, df in dataframes1.items():
            df_list.append(df)
        merged_df = pd.concat(df_list, axis=1)
        
        pls_canonical = PLSRegression(n_components=n_components, **kwargs)
        #print(traindata.shape[0])
        #print(traindata.shape[1])
        #print(prediction_label.shape)
        pls_canonical.fit(merged_df, prediction_label)
        fused_df1 = pd.DataFrame(pls_canonical.transform(merged_df),
                                        columns=[f'PLS{i+1}' for i in range(pls_canonical.n_components)],
                                        index = merged_df.index) 
        self.training_pls_model = pls_canonical
        #print("Training data is fused. ")
        return fused_df1

    
    def fuseFeaturesTest_plsda(self, n_components, method, test_dataframes, **kwargs):
        """
        Internal function for testing of plsda fusion method.
        
        """
        # Iterate through the dictionary and fusion DataFrame
        # train data fusion
        dataframes1 = test_dataframes
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
        fused_df1
        #print("Testing data is fused. ")
        return fused_df1

    def fuseFeaturesTrain_td(self, n_components , method, train_dataframes, train_label,**kwargs):
        """
        Internal function for training of tensordecompose fusion method.
        
        """
        df_list = []
        dataframes1 = train_dataframes
        for name, df in dataframes1.items():
            df_list.append(df)
        df_list_selected=[]
        top_features = []
        for df in df_list:
            num_features = 100
            fs = SelectKBest(score_func=f_regression, k=num_features)
            #print(df)
            X_selected = fs.fit_transform(df, train_label)
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
        return fused_df1

    def fuseFeaturesTest_td(self, n_components, method, test_dataframes, **kwargs):
        """
        Internal function for testing of tensordecompose fusion method.
        
        """
        dataframes1 = test_dataframes
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
        #print("Testing data is fused. ")
        return fused_df1