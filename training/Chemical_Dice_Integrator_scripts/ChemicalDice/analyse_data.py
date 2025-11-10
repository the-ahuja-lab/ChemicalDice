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
from sklearn.metrics import mean_squared_error, r2_score
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



def ccafuse(trainX, trainY, n_components, mode="concat"):
    """
    This function performs Canonical Correlation Analysis (CCA) fusion on the input data.

    Parameters:
    trainX (pd.DataFrame): The first input dataset.
    trainY (pd.DataFrame): The second input dataset.
    n_components (int): The number of components for PCA and CCA.
    mode (str, optional): The mode of fusion, either 'concat' for concatenation or any other string for summation. Defaults to 'concat'.

    Returns:
    pd.DataFrame: The transformed data after CCA fusion.

    The function first applies PCA to both input datasets, reducing their dimensions to 'n_components'. 
    Then, it applies CCA to find a basis that maximizes the correlation of the projections of the datasets onto this basis.
    Depending on the 'mode', it either concatenates ('concat') or sums (any other string) the CCA-transformed datasets.
    The resulting dataset is returned as a DataFrame with columns named 'CCA1', 'CCA2', ..., 'CCAn' and the same index as 'trainX'.

    """

    sample_names=trainX.index
    pca = PCA(n_components=50)
    trainX = pca.fit_transform(trainX)
    pca = PCA(n_components=n_components)
    trainY = pca.fit_transform(trainY)
    cca = CCA(n_components=n_components)
    cca.fit(trainX, trainY)
    trainXcca,trainYcca = cca.transform(trainX, trainY)
    # Assuming trainXcca, trainYcca, testXcca, and testYcca are your transformed data
    if mode == 'concat':  # Fusion by concatenation (Z1)
        trainZ = np.concatenate((trainXcca, trainYcca), axis=1)
    else:  # Fusion by summation (Z2)
        trainZ = trainXcca + trainYcca
    col_trainZ = trainZ.shape[1]
    transformed_data = pd.DataFrame(trainZ, columns=[f'CCA{i+1}' for i in range(col_trainZ)],index = sample_names)
    return transformed_data

def apply_analysis_nonlinear1(data, analysis_type, n_components=None, **kwargs):
    """

    Apply various nonlinear fusion techniques.

    :param data: Input data for analysis.
    :type data: pandas.DataFrame
    :param analysis_type: The type of analysis to be applied.
    :type analysis_type: str
    :param n_components: The number of components to extract (for tsne, kpca, lle, isomap, rks, SpectralEmbedding).
    :type n_components: int, optional
    :param kwargs: Additional parameters for specific analysis methods.
    :type kwargs: dict, optional
    :return: Transformed pandas DataFrame.
    :rtype: pandas.DataFrame

    """
    analysis_types = ['tsne', 'kpca', 'isomap', 'rks', 'sem']
    if analysis_type.lower() not in analysis_types:
        raise ValueError(f"Supported analysis types are: {', '.join(analysis_types)}")
    if analysis_type.lower() == 'tsne':
        tsne = TSNE(n_components=3, random_state=42, **kwargs)
        transformed_data = pd.DataFrame(tsne.fit_transform(data), columns=[f'tSNE{i+1}' for i in range(tsne.n_components)],index = data.index)
    elif analysis_type.lower() == 'kpca':
        kpca = KernelPCA(n_components=n_components, kernel='linear', **kwargs)
        transformed_data = pd.DataFrame(kpca.fit_transform(data), columns=[f'KPC{i+1}' for i in range(kpca.n_components)],index = data.index)
    elif analysis_type.lower() == 'rks':
        rks = RBFSampler(n_components=n_components, random_state=42, **kwargs)
        transformed_data = pd.DataFrame(rks.fit_transform(data), columns=[f'rks{i+1}' for i in range(rks.n_components)],index = data.index)
    elif analysis_type.lower() == 'sem':
        spectral_embedding = SpectralEmbedding(n_components=n_components, random_state=42,**kwargs)
        transformed_data = pd.DataFrame(spectral_embedding.fit_transform(data), columns=[f'SE{i+1}' for i in range(spectral_embedding.n_components)],index = data.index)
    return transformed_data


def apply_analysis_nonlinear2(data, analysis_type,n_neighbors, n_components=None, **kwargs):
    """

    Apply various nonlinear fusion techniques.

    :param data: Input data for analysis.
    :type data: pandas.DataFrame
    :param analysis_type: The type of analysis to be applied.
    :type analysis_type: str
    :param n_components: The number of components to extract (for isomap).
    :type n_components: int, optional
    :param kwargs: Additional parameters for specific analysis methods.
    :type kwargs: dict, optional
    :return: Transformed pandas DataFrame.
    :rtype: pandas.DataFrame

    """
    if analysis_type.lower() == 'isomap':
        isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components, **kwargs)
        transformed_data = pd.DataFrame(isomap.fit_transform(data), columns=[f'isomap{i+1}' for i in range(isomap.n_components)],index = data.index)
    elif analysis_type.lower() == 'lle':
        lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, **kwargs)
        transformed_data = pd.DataFrame(lle.fit_transform(data), columns=[f'LLE{i+1}' for i in range(lle.n_components)],index = data.index)
    return transformed_data




class DynamicAutoencoder(nn.Module):
    """

    A dynamic autoencoder class that can be used to encode and decode data.

    :ivar encoder: The encoder part of the autoencoder. It is a sequence of Linear and ReLU layers.
    :vartype encoder: torch.nn.Sequential
    :ivar decoder: The decoder part of the autoencoder. It is a sequence of Linear and ReLU layers.
    :vartype decoder: torch.nn.Sequential

    :param input_size: The size of the input data.
    :type input_size: int
    :param hidden_sizes: The sizes of the hidden layers in the autoencoder.
    :type hidden_sizes: list

    :method forward(x): Encodes and then decodes the input data.

    """

    def __init__(self, input_size, hidden_sizes):
        super(DynamicAutoencoder, self).__init__()
        # Calculate the sizes of encoder and decoder layers dynamically
        sizes = [input_size] + hidden_sizes + [input_size]
        # Create a list of encoder layers
        encoder_layers = []
        for i in range(len(sizes) - 1):
            encoder_layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                encoder_layers.append(nn.ReLU())
        # Create a list of decoder layers
        decoder_layers = []
        for i in range(len(sizes) - 1, 0, -1):
            decoder_layers.append(nn.Linear(sizes[i], sizes[i - 1]))
            if i > 1:
                decoder_layers.append(nn.ReLU())
        # Define the encoder and decoder as Sequential models
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
    def forward(self, x):
        # Flatten the input if it's not already flattened
        x = x.view(x.size(0), -1)
        # Encode and decode
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def apply_analysis_nonlinear3(data, analysis_type, lr=0.0001, num_epochs = 50, hidden_sizes=None, **kwargs):
    """

    Applies a specified nonlinear analysis to the input data.

    :param data: The input data to be analyzed.
    :type data: pandas.DataFrame
    :param analysis_type: The type of analysis to be applied. Currently, only 'autoencoder' is supported.
    :type analysis_type: str
    :param lr: The learning rate for the optimizer. Default is 0.0001.
    :type lr: float, optional
    :param num_epochs: The number of epochs for training the autoencoder. Default is 50.
    :type num_epochs: int, optional
    :param hidden_sizes: The sizes of the hidden layers in the autoencoder. Default is [128, 64, 36, 18].
    :type hidden_sizes: list of int, optional
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict, optional
    :return: The encoded data if the analysis type is 'autoencoder'.
    :rtype: pandas.DataFrame
    :raises ValueError: If the analysis type is not supported.

    """
    # Instantiate the model, loss function, and optimizer
    if analysis_type.lower() == 'autoencoder':
        sample_names=data.index
        data = data.values
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 36, 18]
        autoencoder = DynamicAutoencoder(input_size = data.shape[1], hidden_sizes= hidden_sizes)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
        # Convert data to PyTorch tensor
        data = torch.FloatTensor(data)
        # Create DataLoader for batching
        batch_size = 4
        dataset = TensorDataset(data, data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Training the autoencoder
        for epoch in range(num_epochs):
            for batch_data, _ in dataloader:
                optimizer.zero_grad()
                output = autoencoder(batch_data)
                loss = criterion(output, batch_data)
                loss.backward()
                optimizer.step()
            # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        # Encode and decode the data
        encoded_data = autoencoder(data).detach().numpy()
        encoded_df = pd.DataFrame(encoded_data,index=sample_names, columns = [f'AE{i+1}' for i in range(encoded_data.shape[1])])
        return encoded_df
    else:
        raise ValueError(f"Supported analysis types is: {', '.join(analysis_type)}")



def apply_analysis_nonlinear4(data, analysis_type, n_components=None, tol=10e-6,**kwargs):
    """

    Apply various multivariate analysis techniques.

    :param data: Input data for analysis.
    :type data: pandas.DataFrame
    :param analysis_type: The type of analysis to be applied.
    :type analysis_type: str
    :param n_components: The number of components to extract (for isomap).
    :type n_components: int, optional
    :param kwargs: Additional parameters for specific analysis methods.
    :type kwargs: dict, optional
    :return: Transformed pandas DataFrame.
    :rtype: pandas.DataFrame

    """
    tensor_all = tl.tensor(data)
    # Perform the CP decomposition
    weights, factors = parafac(tensor_all, rank=n_components, init='random', tol=tol)
    # Reconstruct the image from the factors
    #cp_rec = tl.cp_to_tensor((weights, factors))
    X_combined=factors[1]
    return X_combined
    


def apply_analysis_linear1(data,  analysis_type,  n_components=None, **kwargs):
    """

    Apply various multivariate analysis techniques.

    :param data: Input data for analysis.
    :type data: pandas.DataFrame
    :param analysis_type: The type of analysis to be applied.
    :type analysis_type: str
    :param n_components: The number of components to extract (for PCA, ICA, IPCA).
    :type n_components: int, optional
    :param kwargs: Additional parameters for specific analysis methods.
    :type kwargs: dict, optional
    :return: Transformed pandas DataFrame.
    :rtype: pandas.DataFrame

    """

    analysis_types = ['pca', 'ica', 'ipca']
    if analysis_type.lower() not in analysis_types:
        raise ValueError(f"Supported analysis types are: {', '.join(analysis_types)}")
    if analysis_type.lower() == 'pca':
        pca = PCA(n_components=n_components, **kwargs)
        transformed_data = pd.DataFrame(pca.fit_transform(data),
                                         columns=[f'PC{i+1}' for i in range(pca.n_components_)],index = data.index)
    elif analysis_type.lower() == 'ica':
        ica = FastICA(n_components=n_components, **kwargs)
        transformed_data = pd.DataFrame(ica.fit_transform(data), columns=[f'IC{i+1}' for i in range(ica.n_components)],index = data.index)
    elif analysis_type.lower() == 'ipca':
        ipca = IncrementalPCA(n_components=n_components, **kwargs)
        transformed_data = pd.DataFrame(ipca.fit_transform(data), columns=[f'iPC{i+1}' for i in range(ipca.n_components_)],index = data.index)
    return transformed_data



def apply_analysis_linear2(traindata,  prediction_label,  analysis_type,  n_components=None, **kwargs):
    """

    Apply various multivariate analysis techniques.

    :param data: Input data for analysis.
    :type data: pandas.DataFrame
    :param analysis_type: The type of analysis to be applied.
    :type analysis_type: str
    :param n_components: The number of components to extract (for PCA, ICA, IPCA).
    :type n_components: int, optional
    :param kwargs: Additional parameters for specific analysis methods.
    :type kwargs: dict, optional
    :return: Transformed pandas DataFrame.
    :rtype: pandas.DataFrame
    
    """
    if analysis_type.lower() == 'plsda':
        pls_canonical = PLSRegression(n_components=n_components, **kwargs)
        #print(traindata.shape[0])
        #print(traindata.shape[1])
        #print(prediction_label.shape)
        pls_canonical.fit(traindata, prediction_label)
        transformed_data1 = pd.DataFrame(pls_canonical.transform(traindata),
                                          columns=[f'PLS{i+1}' for i in range(pls_canonical.n_components)],
                                          index = traindata.index) 
        # transformed_data2 = pd.DataFrame(plsa.fit_transform(testdata[0]), columns=[f'PLS{i+1}' for i in range(plsa.n_components)],index = testdata[0].index)        
    return transformed_data1, pls_canonical



def impute_class_specific(df, labels):
    # Assuming df is your DataFrame and 'class' is your class column
    df['class'] = labels.loc[df.index.to_list()]
    classes = df['class'].unique()
    # Create an imputer
    imputer = KNNImputer(n_neighbors=3)
    # Create a list to hold dataframes
    dfs = []
    for class_ in classes:
        # Create a copy of the DataFrame for each class
        df_class = df[df['class'] == class_].copy()
        # Perform KNN imputation for each class
        df_class.iloc[:, :-1] = imputer.fit_transform(df_class.iloc[:, :-1])
        # Append the imputed DataFrame to the list
        dfs.append(df_class)
    # Concatenate the imputed DataFrames
    df_imputed = pd.concat(dfs, ignore_index=True)
    df_imputed =df_imputed.drop("class", axis=1)
    return df_imputed