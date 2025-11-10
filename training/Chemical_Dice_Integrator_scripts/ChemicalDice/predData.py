
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
from ChemicalDice.getEmbeddings import AutoencoderReconstructor_training , AutoencoderReconstructor_testing
import time
from ChemicalDice.splitting import random_scaffold_split_train_val_test


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid




class predData:
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
    def __init__(self, data_paths, smile_file_path, id_column = "id", label_column = "labels"):

        """

        Initialize fusionData object with provided data paths,label file path, ID column name,
        and prediction label column name.

        Args:
            data_paths (dict): Dictionary of file paths containing the data.
            label_file_path (str): file paths containing the label data.
            id_column (str, optional): Name of the column representing unique identifiers.
                Defaults to "id".
            prediction_label_column (str, optional): Name of the column containing prediction labels.
                Defaults to "labels".

        """


        self.data_paths = data_paths
        loaded_data_dict = clear_and_process_data(data_paths, id_column)



        smile_df = pd.read_csv(smile_file_path)
        smile_df.set_index(id_column, inplace=True)
        self.smiles_col = smile_df["SMILES"]

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
                common_columns = ['GeomDiameter', 'GeomRadius', 'GeomShapeIndex', 'GeomPetitjeanIndex', 'IC0', 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'TIC0', 'TIC1', 'TIC2', 'TIC3', 'TIC4', 'TIC5', 'SIC0', 'SIC1', 'SIC2', 'SIC3', 'SIC4', 'SIC5', 'BIC0', 'BIC1', 'BIC2', 'BIC3', 'BIC4', 'BIC5', 'CIC0', 'CIC1', 'CIC2', 'CIC3', 'CIC4', 'CIC5', 'MIC0', 'MIC1', 'MIC2', 'MIC3', 'MIC4', 'MIC5', 'ZMIC0', 'ZMIC1', 'ZMIC2', 'ZMIC3', 'ZMIC4', 'ZMIC5', 'nRing', 'n3Ring', 'n4Ring', 'n5Ring', 'n6Ring', 'n7Ring', 'n8Ring', 'n9Ring', 'n10Ring', 'n11Ring', 'n12Ring', 'nG12Ring', 'nHRing', 'n3HRing', 'n4HRing', 'n5HRing', 'n6HRing', 'n7HRing', 'n8HRing', 'n9HRing', 'n10HRing', 'n11HRing', 'n12HRing', 'nG12HRing', 'naRing', 'n3aRing', 'n4aRing', 'n5aRing', 'n6aRing', 'n7aRing', 'n8aRing', 'n9aRing', 'n10aRing', 'n11aRing', 'n12aRing', 'nG12aRing', 'naHRing', 'n3aHRing', 'n4aHRing', 'n5aHRing', 'n6aHRing', 'n7aHRing', 'n8aHRing', 'n9aHRing', 'n10aHRing', 'n11aHRing', 'n12aHRing', 'nG12aHRing', 'nARing', 'n3ARing', 'n4ARing', 'n5ARing', 'n6ARing', 'n7ARing', 'n8ARing', 'n9ARing', 'n10ARing', 'n11ARing', 'n12ARing', 'nG12ARing', 'nAHRing', 'n3AHRing', 'n4AHRing', 'n5AHRing', 'n6AHRing', 'n7AHRing', 'n8AHRing', 'n9AHRing', 'n10AHRing', 'n11AHRing', 'n12AHRing', 'nG12AHRing', 'nFRing', 'n4FRing', 'n5FRing', 'n6FRing', 'n7FRing', 'n8FRing', 'n9FRing', 'n10FRing', 'n11FRing', 'n12FRing', 'nG12FRing', 'nFHRing', 'n4FHRing', 'n5FHRing', 'n6FHRing', 'n7FHRing', 'n8FHRing', 'n9FHRing', 'n10FHRing', 'n11FHRing', 'n12FHRing', 'nG12FHRing', 'nFaRing', 'n4FaRing', 'n5FaRing', 'n6FaRing', 'n7FaRing', 'n8FaRing', 'n9FaRing', 'n10FaRing', 'n11FaRing', 'n12FaRing', 'nG12FaRing', 'nFaHRing', 'n4FaHRing', 'n5FaHRing', 'n6FaHRing', 'n7FaHRing', 'n8FaHRing', 'n9FaHRing', 'n10FaHRing', 'n11FaHRing', 'n12FaHRing', 'nG12FaHRing', 'nFARing', 'n4FARing', 'n5FARing', 'n6FARing', 'n7FARing', 'n8FARing', 'n9FARing', 'n10FARing', 'n11FARing', 'n12FARing', 'nG12FARing', 'nFAHRing', 'n4FAHRing', 'n5FAHRing', 'n6FAHRing', 'n7FAHRing', 'n8FAHRing', 'n9FAHRing', 'n10FAHRing', 'n11FAHRing', 'n12FAHRing', 'nG12FAHRing', 'MOMI-X', 'MOMI-Y', 'MOMI-Z', 'VMcGowan', 'WPath', 'WPol', 'C1SP1', 'C2SP1', 'C1SP2', 'C2SP2', 'C3SP2', 'C1SP3', 'C2SP3', 'C3SP3', 'C4SP3', 'HybRatio', 'Zagreb1', 'Zagreb2', 'mZagreb1', 'mZagreb2', 'SpAbs_DzZ', 'SpMax_DzZ', 'SpDiam_DzZ', 'SpAD_DzZ', 'SpMAD_DzZ', 'LogEE_DzZ', 'SM1_DzZ', 'VE1_DzZ', 'VE2_DzZ', 'VE3_DzZ', 'VR1_DzZ', 'VR2_DzZ', 'VR3_DzZ', 'SpAbs_Dzm', 'SpMax_Dzm', 'SpDiam_Dzm', 'SpAD_Dzm', 'SpMAD_Dzm', 'LogEE_Dzm', 'SM1_Dzm', 'VE1_Dzm', 'VE2_Dzm', 'VE3_Dzm', 'VR1_Dzm', 'VR2_Dzm', 'VR3_Dzm', 'SpAbs_Dzv', 'SpMax_Dzv', 'SpDiam_Dzv', 'SpAD_Dzv', 'SpMAD_Dzv', 'LogEE_Dzv', 'SM1_Dzv', 'VE1_Dzv', 'VE2_Dzv', 'VE3_Dzv', 'VR1_Dzv', 'VR2_Dzv', 'VR3_Dzv', 'SpAbs_Dzse', 'SpMax_Dzse', 'SpDiam_Dzse', 'SpAD_Dzse', 'SpMAD_Dzse', 'LogEE_Dzse', 'SM1_Dzse', 'VE1_Dzse', 'VE2_Dzse', 'VE3_Dzse', 'VR1_Dzse', 'VR2_Dzse', 'VR3_Dzse', 'SpAbs_Dzpe', 'SpMax_Dzpe', 'SpDiam_Dzpe', 'SpAD_Dzpe', 'SpMAD_Dzpe', 'LogEE_Dzpe', 'SM1_Dzpe', 'VE1_Dzpe', 'VE2_Dzpe', 'VE3_Dzpe', 'VR1_Dzpe', 'VR2_Dzpe', 'VR3_Dzpe', 'SpAbs_Dzare', 'SpMax_Dzare', 'SpDiam_Dzare', 'SpAD_Dzare', 'SpMAD_Dzare', 'LogEE_Dzare', 'SM1_Dzare', 'VE1_Dzare', 'VE2_Dzare', 'VE3_Dzare', 'VR1_Dzare', 'VR2_Dzare', 'VR3_Dzare', 'SpAbs_Dzp', 'SpMax_Dzp', 'SpDiam_Dzp', 'SpAD_Dzp', 'SpMAD_Dzp', 'LogEE_Dzp', 'SM1_Dzp', 'VE1_Dzp', 'VE2_Dzp', 'VE3_Dzp', 'VR1_Dzp', 'VR2_Dzp', 'VR3_Dzp', 'SpAbs_Dzi', 'SpMax_Dzi', 'SpDiam_Dzi', 'SpAD_Dzi', 'SpMAD_Dzi', 'LogEE_Dzi', 'SM1_Dzi', 'VE1_Dzi', 'VE2_Dzi', 'VE3_Dzi', 'VR1_Dzi', 'VR2_Dzi', 'VR3_Dzi', 'nAtom', 'nHeavyAtom', 'nSpiro', 'nBridgehead', 'nH', 'nB', 'nC', 'nN', 'nO', 'nS', 'nP', 'nF', 'nCl', 'nBr', 'nI', 'nX', 'MDEC-22', 'MDEC-23', 'MDEC-33', 'BalabanJ', 'ETA_alpha', 'AETA_alpha', 'ETA_shape_p', 'ETA_shape_y', 'ETA_shape_x', 'ETA_beta', 'AETA_beta', 'ETA_beta_s', 'AETA_beta_s', 'ETA_beta_ns', 'AETA_beta_ns', 'ETA_beta_ns_d', 'AETA_beta_ns_d', 'ETA_eta', 'AETA_eta', 'ETA_eta_L', 'AETA_eta_L', 'ETA_eta_R', 'AETA_eta_R', 'ETA_eta_RL', 'AETA_eta_RL', 'ETA_eta_F', 'AETA_eta_F', 'ETA_eta_FL', 'AETA_eta_FL', 'ETA_eta_B', 'AETA_eta_B', 'ETA_eta_BR', 'AETA_eta_BR', 'ETA_dAlpha_A', 'ETA_dAlpha_B', 'ETA_epsilon_1', 'ETA_epsilon_2', 'ETA_epsilon_3', 'ETA_epsilon_4', 'ETA_epsilon_5', 'ETA_dEpsilon_A', 'ETA_dEpsilon_B', 'ETA_dEpsilon_C', 'ETA_dEpsilon_D', 'ETA_dBeta', 'AETA_dBeta', 'ETA_psi_1', 'ETA_dPsi_A', 'ETA_dPsi_B', 'MID', 'AMID', 'MID_h', 'AMID_h', 'MID_C', 'AMID_C', 'MID_N', 'AMID_N', 'MID_O', 'AMID_O', 'MID_X', 'AMID_X', 'GGI1', 'GGI2', 'GGI3', 'GGI4', 'GGI5', 'GGI6', 'GGI7', 'GGI8', 'GGI9', 'GGI10', 'JGI1', 'JGI2', 'JGI3', 'JGI4', 'JGI5', 'JGI6', 'JGI7', 'JGI8', 'JGI9', 'JGI10', 'JGT10', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10', 'SlogP_VSA11', 'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10', 'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'fMF', 'SZ', 'Sm', 'Sv', 'Sse', 'Spe', 'Sare', 'Sp', 'Si', 'MZ', 'Mm', 'Mv', 'Mse', 'Mpe', 'Mare', 'Mp', 'Mi', 'nRot', 'RotRatio', 'NsLi', 'NssBe', 'NssssBe', 'NssBH', 'NsssB', 'NssssB', 'NsCH3', 'NdCH2', 'NssCH2', 'NtCH', 'NdsCH', 'NaaCH', 'NsssCH', 'NddC', 'NtsC', 'NdssC', 'NaasC', 'NaaaC', 'NssssC', 'NsNH3', 'NsNH2', 'NssNH2', 'NdNH', 'NssNH', 'NaaNH', 'NtN', 'NsssNH', 'NdsN', 'NaaN', 'NsssN', 'NddsN', 'NaasN', 'NssssN', 'NsOH', 'NdO', 'NssO', 'NaaO', 'NsF', 'NsSiH3', 'NssSiH2', 'NsssSiH', 'NssssSi', 'NsPH2', 'NssPH', 'NsssP', 'NdsssP', 'NsssssP', 'NsSH', 'NdS', 'NssS', 'NaaS', 'NdssS', 'NddssS', 'NsCl', 'NsGeH3', 'NssGeH2', 'NsssGeH', 'NssssGe', 'NsAsH2', 'NssAsH', 'NsssAs', 'NsssdAs', 'NsssssAs', 'NsSeH', 'NdSe', 'NssSe', 'NaaSe', 'NdssSe', 'NddssSe', 'NsBr', 'NsSnH3', 'NssSnH2', 'NsssSnH', 'NssssSn', 'NsI', 'NsPbH3', 'NssPbH2', 'NsssPbH', 'NssssPb', 'SsLi', 'SssBe', 'SssssBe', 'SssBH', 'SsssB', 'SssssB', 'SsCH3', 'SdCH2', 'SssCH2', 'StCH', 'SdsCH', 'SaaCH', 'SsssCH', 'SddC', 'StsC', 'SdssC', 'SaasC', 'SaaaC', 'SssssC', 'SsNH3', 'SsNH2', 'SssNH2', 'SdNH', 'SssNH', 'SaaNH', 'StN', 'SsssNH', 'SdsN', 'SaaN', 'SsssN', 'SddsN', 'SaasN', 'SssssN', 'SsOH', 'SdO', 'SssO', 'SaaO', 'SsF', 'SsSiH3', 'SssSiH2', 'SsssSiH', 'SssssSi', 'SsPH2', 'SssPH', 'SsssP', 'SdsssP', 'SsssssP', 'SsSH', 'SdS', 'SssS', 'SaaS', 'SdssS', 'SddssS', 'SsCl', 'SsGeH3', 'SssGeH2', 'SsssGeH', 'SssssGe', 'SsAsH2', 'SssAsH', 'SsssAs', 'SsssdAs', 'SsssssAs', 'SsSeH', 'SdSe', 'SssSe', 'SaaSe', 'SdssSe', 'SddssSe', 'SsBr', 'SsSnH3', 'SssSnH2', 'SsssSnH', 'SssssSn', 'SsI', 'SsPbH3', 'SssPbH2', 'SsssPbH', 'SssssPb', 'PNSA1', 'PNSA2', 'PNSA3', 'PNSA4', 'PNSA5', 'PPSA1', 'PPSA2', 'PPSA3', 'PPSA4', 'PPSA5', 'DPSA1', 'DPSA2', 'DPSA3', 'DPSA4', 'DPSA5', 'FNSA1', 'FNSA2', 'FNSA3', 'FNSA4', 'FNSA5', 'FPSA1', 'FPSA2', 'FPSA3', 'FPSA4', 'FPSA5', 'WNSA1', 'WNSA2', 'WNSA3', 'WNSA4', 'WNSA5', 'WPSA1', 'WPSA2', 'WPSA3', 'WPSA4', 'WPSA5', 'RNCG', 'RPCG', 'RNCS', 'RPCS', 'TASA', 'TPSA', 'RASA', 'RPSA', 'BertzCT', 'nBonds', 'nBondsO', 'nBondsS', 'nBondsD', 'nBondsT', 'nBondsA', 'nBondsM', 'nBondsKS', 'nBondsKD', 'GRAV', 'GRAVp', 'Xch-3d', 'Xch-4d', 'Xch-5d', 'Xch-6d', 'Xch-7d', 'Xch-3dv', 'Xch-4dv', 'Xch-5dv', 'Xch-6dv', 'Xch-7dv', 'Xc-3d', 'Xc-4d', 'Xc-5d', 'Xc-6d', 'Xc-3dv', 'Xc-4dv', 'Xc-5dv', 'Xc-6dv', 'Xpc-4d', 'Xpc-5d', 'Xpc-6d', 'Xpc-4dv', 'Xpc-5dv', 'Xpc-6dv', 'Xp-0d', 'Xp-1d', 'Xp-2d', 'Xp-3d', 'Xp-4d', 'Xp-5d', 'Xp-6d', 'Xp-7d', 'AXp-0d', 'AXp-1d', 'AXp-2d', 'AXp-3d', 'AXp-4d', 'AXp-5d', 'AXp-6d', 'AXp-7d', 'Xp-0dv', 'Xp-1dv', 'Xp-2dv', 'Xp-3dv', 'Xp-4dv', 'Xp-5dv', 'Xp-6dv', 'Xp-7dv', 'AXp-0dv', 'AXp-1dv', 'AXp-2dv', 'AXp-3dv', 'AXp-4dv', 'AXp-5dv', 'AXp-6dv', 'AXp-7dv', 'nAcid', 'nBase', 'ECIndex', 'SLogP', 'SMR', 'Diameter', 'Radius', 'TopoShapeIndex', 'PetitjeanIndex', 'SpAbs_D', 'SpMax_D', 'SpDiam_D', 'SpAD_D', 'SpMAD_D', 'LogEE_D', 'SM1_D', 'VE1_D', 'VE2_D', 'VE3_D', 'VR1_D', 'VR2_D', 'VR3_D', 'TopoPSA(NO)', 'TopoPSA', 'nAromAtom', 'nAromBond', 'MWC01', 'MWC02', 'MWC03', 'MWC04', 'MWC05', 'MWC06', 'MWC07', 'MWC08', 'MWC09', 'MWC10', 'TMWC10', 'SRW02', 'SRW03', 'SRW04', 'SRW05', 'SRW06', 'SRW07', 'SRW08', 'SRW09', 'SRW10', 'TSRW10', 'VAdjMat', 'MW', 'AMW', 'BCUTc-1h', 'BCUTc-1l', 'BCUTdv-1h', 'BCUTdv-1l', 'BCUTd-1h', 'BCUTd-1l', 'BCUTs-1h', 'BCUTs-1l', 'BCUTZ-1h', 'BCUTZ-1l', 'BCUTm-1h', 'BCUTm-1l', 'BCUTv-1h', 'BCUTv-1l', 'BCUTse-1h', 'BCUTse-1l', 'BCUTpe-1h', 'BCUTpe-1l', 'BCUTare-1h', 'BCUTare-1l', 'BCUTp-1h', 'BCUTp-1l', 'BCUTi-1h', 'BCUTi-1l', 'fragCpx', 'apol', 'bpol', 'SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A', 'LogEE_A', 'SM1_A', 'VE1_A', 'VE2_A', 'VE3_A', 'VR1_A', 'VR2_A', 'VR3_A', 'Kier1', 'Kier2', 'Kier3', 'MPC2', 'MPC3', 'MPC4', 'MPC5', 'MPC6', 'MPC7', 'MPC8', 'MPC9', 'MPC10', 'TMPC10', 'piPC1', 'piPC2', 'piPC3', 'piPC4', 'piPC5', 'piPC6', 'piPC7', 'piPC8', 'piPC9', 'piPC10', 'TpiPC10', 'nHBAcc', 'nHBDon', 'ATS0dv', 'ATS1dv', 'ATS2dv', 'ATS3dv', 'ATS4dv', 'ATS5dv', 'ATS6dv', 'ATS7dv', 'ATS8dv', 'ATS0d', 'ATS1d', 'ATS2d', 'ATS3d', 'ATS4d', 'ATS5d', 'ATS6d', 'ATS7d', 'ATS8d', 'ATS0s', 'ATS1s', 'ATS2s', 'ATS3s', 'ATS4s', 'ATS5s', 'ATS6s', 'ATS7s', 'ATS8s', 'ATS0Z', 'ATS1Z', 'ATS2Z', 'ATS3Z', 'ATS4Z', 'ATS5Z', 'ATS6Z', 'ATS7Z', 'ATS8Z', 'ATS0m', 'ATS1m', 'ATS2m', 'ATS3m', 'ATS4m', 'ATS5m', 'ATS6m', 'ATS7m', 'ATS8m', 'ATS0v', 'ATS1v', 'ATS2v', 'ATS3v', 'ATS4v', 'ATS5v', 'ATS6v', 'ATS7v', 'ATS8v', 'ATS0se', 'ATS1se', 'ATS2se', 'ATS3se', 'ATS4se', 'ATS5se', 'ATS6se', 'ATS7se', 'ATS8se', 'ATS0pe', 'ATS1pe', 'ATS2pe', 'ATS3pe', 'ATS4pe', 'ATS5pe', 'ATS6pe', 'ATS7pe', 'ATS8pe', 'ATS0are', 'ATS1are', 'ATS2are', 'ATS3are', 'ATS4are', 'ATS5are', 'ATS6are', 'ATS7are', 'ATS8are', 'ATS0p', 'ATS1p', 'ATS2p', 'ATS3p', 'ATS4p', 'ATS5p', 'ATS6p', 'ATS7p', 'ATS8p', 'ATS0i', 'ATS1i', 'ATS2i', 'ATS3i', 'ATS4i', 'ATS5i', 'ATS6i', 'ATS7i', 'ATS8i', 'AATS0dv', 'AATS1dv', 'AATS2dv', 'AATS3dv', 'AATS4dv', 'AATS5dv', 'AATS6dv', 'AATS7dv', 'AATS0d', 'AATS1d', 'AATS2d', 'AATS3d', 'AATS4d', 'AATS5d', 'AATS6d', 'AATS7d', 'AATS0s', 'AATS1s', 'AATS2s', 'AATS3s', 'AATS4s', 'AATS5s', 'AATS6s', 'AATS7s', 'AATS0Z', 'AATS1Z', 'AATS2Z', 'AATS3Z', 'AATS4Z', 'AATS5Z', 'AATS6Z', 'AATS7Z', 'AATS0m', 'AATS1m', 'AATS2m', 'AATS3m', 'AATS4m', 'AATS5m', 'AATS6m', 'AATS7m', 'AATS0v', 'AATS1v', 'AATS2v', 'AATS3v', 'AATS4v', 'AATS5v', 'AATS6v', 'AATS7v', 'AATS0se', 'AATS1se', 'AATS2se', 'AATS3se', 'AATS4se', 'AATS5se', 'AATS6se', 'AATS7se', 'AATS0pe', 'AATS1pe', 'AATS2pe', 'AATS3pe', 'AATS4pe', 'AATS5pe', 'AATS6pe', 'AATS7pe', 'AATS0are', 'AATS1are', 'AATS2are', 'AATS3are', 'AATS4are', 'AATS5are', 'AATS6are', 'AATS7are', 'AATS0p', 'AATS1p', 'AATS2p', 'AATS3p', 'AATS4p', 'AATS5p', 'AATS6p', 'AATS7p', 'AATS0i', 'AATS1i', 'AATS2i', 'AATS3i', 'AATS4i', 'AATS5i', 'AATS6i', 'AATS7i', 'ATSC0c', 'ATSC1c', 'ATSC2c', 'ATSC3c', 'ATSC4c', 'ATSC5c', 'ATSC6c', 'ATSC7c', 'ATSC8c', 'ATSC0dv', 'ATSC1dv', 'ATSC2dv', 'ATSC3dv', 'ATSC4dv', 'ATSC5dv', 'ATSC6dv', 'ATSC7dv', 'ATSC8dv', 'ATSC0d', 'ATSC1d', 'ATSC2d', 'ATSC3d', 'ATSC4d', 'ATSC5d', 'ATSC6d', 'ATSC7d', 'ATSC8d', 'ATSC0s', 'ATSC1s', 'ATSC2s', 'ATSC3s', 'ATSC4s', 'ATSC5s', 'ATSC6s', 'ATSC7s', 'ATSC8s', 'ATSC0Z', 'ATSC1Z', 'ATSC2Z', 'ATSC3Z', 'ATSC4Z', 'ATSC5Z', 'ATSC6Z', 'ATSC7Z', 'ATSC8Z', 'ATSC0m', 'ATSC1m', 'ATSC2m', 'ATSC3m', 'ATSC4m', 'ATSC5m', 'ATSC6m', 'ATSC7m', 'ATSC8m', 'ATSC0v', 'ATSC1v', 'ATSC2v', 'ATSC3v', 'ATSC4v', 'ATSC5v', 'ATSC6v', 'ATSC7v', 'ATSC8v', 'ATSC0se', 'ATSC1se', 'ATSC2se', 'ATSC3se', 'ATSC4se', 'ATSC5se', 'ATSC6se', 'ATSC7se', 'ATSC8se', 'ATSC0pe', 'ATSC1pe', 'ATSC2pe', 'ATSC3pe', 'ATSC4pe', 'ATSC5pe', 'ATSC6pe', 'ATSC7pe', 'ATSC8pe', 'ATSC0are', 'ATSC1are', 'ATSC2are', 'ATSC3are', 'ATSC4are', 'ATSC5are', 'ATSC6are', 'ATSC7are', 'ATSC8are', 'ATSC0p', 'ATSC1p', 'ATSC2p', 'ATSC3p', 'ATSC4p', 'ATSC5p', 'ATSC6p', 'ATSC7p', 'ATSC8p', 'ATSC0i', 'ATSC1i', 'ATSC2i', 'ATSC3i', 'ATSC4i', 'ATSC5i', 'ATSC6i', 'ATSC7i', 'ATSC8i', 'AATSC0c', 'AATSC1c', 'AATSC2c', 'AATSC3c', 'AATSC4c', 'AATSC5c', 'AATSC6c', 'AATSC7c', 'AATSC0dv', 'AATSC1dv', 'AATSC2dv', 'AATSC3dv', 'AATSC4dv', 'AATSC5dv', 'AATSC6dv', 'AATSC7dv', 'AATSC0d', 'AATSC1d', 'AATSC2d', 'AATSC3d', 'AATSC4d', 'AATSC5d', 'AATSC6d', 'AATSC7d', 'AATSC0s', 'AATSC1s', 'AATSC2s', 'AATSC3s', 'AATSC4s', 'AATSC5s', 'AATSC6s', 'AATSC7s', 'AATSC0Z', 'AATSC1Z', 'AATSC2Z', 'AATSC3Z', 'AATSC4Z', 'AATSC5Z', 'AATSC6Z', 'AATSC7Z', 'AATSC0m', 'AATSC1m', 'AATSC2m', 'AATSC3m', 'AATSC4m', 'AATSC5m', 'AATSC6m', 'AATSC7m', 'AATSC0v', 'AATSC1v', 'AATSC2v', 'AATSC3v', 'AATSC4v', 'AATSC5v', 'AATSC6v', 'AATSC7v', 'AATSC0se', 'AATSC1se', 'AATSC2se', 'AATSC3se', 'AATSC4se', 'AATSC5se', 'AATSC6se', 'AATSC7se', 'AATSC0pe', 'AATSC1pe', 'AATSC2pe', 'AATSC3pe', 'AATSC4pe', 'AATSC5pe', 'AATSC6pe', 'AATSC7pe', 'AATSC0are', 'AATSC1are', 'AATSC2are', 'AATSC3are', 'AATSC4are', 'AATSC5are', 'AATSC6are', 'AATSC7are', 'AATSC0p', 'AATSC1p', 'AATSC2p', 'AATSC3p', 'AATSC4p', 'AATSC5p', 'AATSC6p', 'AATSC7p', 'AATSC0i', 'AATSC1i', 'AATSC2i', 'AATSC3i', 'AATSC4i', 'AATSC5i', 'AATSC6i', 'AATSC7i', 'MATS1c', 'MATS2c', 'MATS3c', 'MATS4c', 'MATS5c', 'MATS6c', 'MATS7c', 'MATS1dv', 'MATS2dv', 'MATS3dv', 'MATS4dv', 'MATS5dv', 'MATS6dv', 'MATS7dv', 'MATS1d', 'MATS2d', 'MATS3d', 'MATS4d', 'MATS5d', 'MATS6d', 'MATS7d', 'MATS1s', 'MATS2s', 'MATS3s', 'MATS4s', 'MATS5s', 'MATS6s', 'MATS7s', 'MATS1Z', 'MATS2Z', 'MATS3Z', 'MATS4Z', 'MATS5Z', 'MATS6Z', 'MATS7Z', 'MATS1m', 'MATS2m', 'MATS3m', 'MATS4m', 'MATS5m', 'MATS6m', 'MATS7m', 'MATS1v', 'MATS2v', 'MATS3v', 'MATS4v', 'MATS5v', 'MATS6v', 'MATS7v', 'MATS1se', 'MATS2se', 'MATS3se', 'MATS4se', 'MATS5se', 'MATS6se', 'MATS7se', 'MATS1pe', 'MATS2pe', 'MATS3pe', 'MATS4pe', 'MATS5pe', 'MATS6pe', 'MATS7pe', 'MATS1are', 'MATS2are', 'MATS3are', 'MATS4are', 'MATS5are', 'MATS6are', 'MATS7are', 'MATS1p', 'MATS2p', 'MATS3p', 'MATS4p', 'MATS5p', 'MATS6p', 'MATS7p', 'MATS1i', 'MATS2i', 'MATS3i', 'MATS4i', 'MATS5i', 'MATS6i', 'MATS7i', 'GATS1c', 'GATS2c', 'GATS3c', 'GATS4c', 'GATS5c', 'GATS6c', 'GATS7c', 'GATS1dv', 'GATS2dv', 'GATS3dv', 'GATS4dv', 'GATS5dv', 'GATS6dv', 'GATS7dv', 'GATS1d', 'GATS2d', 'GATS3d', 'GATS4d', 'GATS5d', 'GATS6d', 'GATS7d', 'GATS1s', 'GATS2s', 'GATS3s', 'GATS4s', 'GATS5s', 'GATS6s', 'GATS7s', 'GATS1Z', 'GATS2Z', 'GATS3Z', 'GATS4Z', 'GATS5Z', 'GATS6Z', 'GATS7Z', 'GATS1m', 'GATS2m', 'GATS3m', 'GATS4m', 'GATS5m', 'GATS6m', 'GATS7m', 'GATS1v', 'GATS2v', 'GATS3v', 'GATS4v', 'GATS5v', 'GATS6v', 'GATS7v', 'GATS1se', 'GATS2se', 'GATS3se', 'GATS4se', 'GATS5se', 'GATS6se', 'GATS7se', 'GATS1pe', 'GATS2pe', 'GATS3pe', 'GATS4pe', 'GATS5pe', 'GATS6pe', 'GATS7pe', 'GATS1are', 'GATS2are', 'GATS3are', 'GATS4are', 'GATS5are', 'GATS6are', 'GATS7are', 'GATS1p', 'GATS2p', 'GATS3p', 'GATS4p', 'GATS5p', 'GATS6p', 'GATS7p', 'GATS1i', 'GATS2i', 'GATS3i', 'GATS4i', 'GATS5i', 'GATS6i', 'GATS7i']
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



    def fuseFeaturesTrain(self, n_components, AER_dim , method="pca",**kwargs):
        # The rest of the code is unchanged

        # Iterate through the dictionary and fusion DataFrame
        # train data fusion
        dataframes1 = self.train_dataframes
        #print(dataframes1)
        df_list = []
        for name, df in dataframes1.items():
            df_list.append(df)
        #print(df_list)
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
        elif method in ['AER']:
            if self.training_AER_model is None or self.AER_model_embt_size != AER_dim:
                if AER_dim in [256, 512, 1024, 4096, 8192]:
                    embd = AER_dim
                else:
                    embd = 4096
                
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
                #print(df_list2)
                all_embeddings, model_state = AutoencoderReconstructor_training(df_list2[0], df_list2[1], df_list2[2], df_list2[3],df_list2[4],df_list2[5],embedding_sizes=[embd])
                mopac_feature_wt = model_state['weights.0'].cpu().numpy()
                chemberta_feature_wt = model_state['weights.1'].cpu().numpy()
                mordred_feature_wt = model_state['weights.2'].cpu().numpy()
                signaturizer_feature_wt = model_state['weights.3'].cpu().numpy()
                imagemol_feature_wt = model_state['weights.4'].cpu().numpy()
                grover_feature_wt = model_state['weights.5'].cpu().numpy()
                explanibility = {}

                for name, df in self.dataframes.items():
                    if name.lower() == "mopac":
                        column_name = df.columns
                        explanibility['name'] = pd.DataFrame({'Feature':column_name,'weights':mopac_feature_wt})
                    elif name.lower() == "chemberta":
                        column_name = df.columns
                        explanibility['name'] = pd.DataFrame({'Feature':column_name,'weights':chemberta_feature_wt})
                    elif name.lower() ==  "mordred":
                        column_name = df.columns
                        explanibility['name'] = pd.DataFrame({'Feature':column_name,'weights':mordred_feature_wt})
                    elif name.lower() ==  "signaturizer":
                        column_name = df.columns
                        explanibility['name'] = pd.DataFrame({'Feature':column_name,'weights':signaturizer_feature_wt})
                    elif name.lower() ==  "imagemol":
                        column_name = df.columns
                        explanibility['name'] = pd.DataFrame({'Feature':column_name,'weights':imagemol_feature_wt})
                    elif name.lower() ==  "grover":
                        column_name = df.columns
                        explanibility['name'] = pd.DataFrame({'Feature':column_name,'weights':grover_feature_wt})
                self.training_AER_model = model_state
                self.AER_model_embt_size = embd
                self.AER_model_explainability = explanibility
            else:
                print("AER model Training")
            if AER_dim in [256, 512, 1024, 4096, 8192]:
                embd = AER_dim
            else:
                print("AER_embed size should be  ",256, 512, 1024, 4096, " or ", 8192)
                embd = 4096
            
            df_list2=[None,None,None,None,None,None]
            for name, df in dataframes1.items():
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
            
            model_state = self.training_AER_model
            all_embeddings = AutoencoderReconstructor_testing(df_list2[0],df_list2[1],df_list2[2],df_list2[3],df_list2[4],df_list2[5],embedding_sizes=[embd], model_state=model_state)
            
            scaler = StandardScaler()
            fused_df_unstandardized = all_embeddings[0]
            fused_df_unstandardized.set_index("id",inplace =True)
            fused_df1 = pd.DataFrame(scaler.fit_transform(fused_df_unstandardized), index=fused_df_unstandardized.index, columns=fused_df_unstandardized.columns)

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

    
    def fuseFeaturesPredict(self, n_components, AER_dim, method="pca", **kwargs):
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
        elif method in ['AER']:
            if AER_dim in [256, 512, 1024, 4096, 8192]:
                embd = AER_dim
            else:
                embd = 4096
            
            df_list2=[None,None,None,None,None,None]
            for name, df in dataframes1.items():
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
            model_state = self.training_AER_model
            all_embeddings = AutoencoderReconstructor_testing(df_list2[0],df_list2[1],df_list2[2],df_list2[3],df_list2[4],df_list2[5],embedding_sizes=[embd], model_state=model_state)
            
            scaler = StandardScaler()
            fused_df_unstandardized = all_embeddings[0]
            fused_df_unstandardized.set_index("id",inplace =True)
            fused_df1 = pd.DataFrame(scaler.fit_transform(fused_df_unstandardized), index=fused_df_unstandardized.index, columns=fused_df_unstandardized.columns)
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



    def predict_fusion_model(self, fd, n_components=10,regression = False, AER_dim=4096, **kwargs):
        """
        Evaluate n-fold cross validations of different fusion models on the dataframes and prediction labels. 

        The fusion methods are as follows: 

        - 'AER': Autoencoder Reconstruction Analysis
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
        method_name = fd.Accuracy_metrics['Model'].to_list()[0]

        if regression == False:
            # Define a list of models
            models = [
                ("Logistic Regression", LogisticRegression()),
                ("Decision Tree", DecisionTreeClassifier()),
                ("Random Forest", RandomForestClassifier()),
                ("Support Vector Machine", SVC(probability=True)),
                ("Naive Bayes", GaussianNB())
            ]

            best_fusion_method = method_name.split(" ")[0]
            best_model_name = method_name.replace(best_fusion_method+" ","")
            for name, model in models:
                if name == best_model_name:
                    best_model = model


            self.train_dataframes = fd.dataframes
            self.train_label = fd.prediction_label

            self.test_dataframes = self.dataframes
            
            self.fuseFeaturesTrain(n_components = n_components,  method=best_fusion_method, AER_dim=AER_dim)
            X_train = self.fusedData_train
            y_train = self.train_label
            self.fuseFeaturesPredict(n_components = n_components,  method=best_fusion_method, AER_dim=AER_dim)
            X_test = self.fusedData_test
            best_model.fit(X_train, y_train)
            results = best_model.predict(X_test)
            results


        else:

            method_name = fd.Accuracy_metrics['Model'].to_list()[0]
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


            best_fusion_method = method_name.split(" ")[0]
            best_model_name = method_name.replace(best_fusion_method+" ","")
            for name, model in models:
                if name == best_model_name:
                    best_model = model


            #self.fuseFeaturesTrain(n_components = n_components,  method=method_chemdice, AER_dim=AER_dim)

            self.train_dataframes = fd.dataframes
            self.train_label = fd.prediction_label

            self.test_dataframes = self.dataframes
            
            self.fuseFeaturesTrain(n_components = n_components,  method=best_fusion_method, AER_dim=AER_dim)
            X_train = self.fusedData_train
            y_train = self.train_label
            self.fuseFeaturesPredict(n_components = n_components,  method=best_fusion_method, AER_dim=AER_dim)
            X_test = self.fusedData_test
            best_model.fit(X_train, y_train)
            results = best_model.predict(X_test)
            results

