
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
from scipy import stats
import os
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.backends.backend_pdf import PdfPages

def plot_metrics(metrics_df, dataset_name):
    """
    This function creates a PDF file with box plots for each metric in the metrics dataframe.

    :param metrics_df: DataFrame containing the metrics
    :type metrics_df: pandas.DataFrame
    :param dataset_name: Name of the dataset
    :type dataset_name: str
    """
    if 'AUC' in metrics_df.columns:
        metrics_list = ['AUC', 'Accuracy', 'Balanced accuracy', 'Precision', 'Recall','f1 score', 'Kappa', 'MCC']
    else:
        metrics_list = ["R2 Score" , "MSE", "RMSE", "MAE"]

    model_type_palette = {'Non-linear': '#735DA5', 'linear': '#D3C5E5', 'No fusion': '#FAB4A5'}

    pdf = PdfPages(dataset_name+"metrics_plot.pdf")
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, dataset_name+' Dataset', ha='center', va='center', size=24)
    plt.axis('off')
    plt.savefig(pdf, format='pdf')

    for metric in metrics_list:
        # Create a new figure for each metric
        plt.figure(figsize=(10, 6))

        # Create the box plot
        sns.stripplot(x='Model', y=metric, data=metrics_df, hue='Fold', dodge=True, jitter=True, marker='o', alpha=0.8, palette='Set3', size =3)
        sns.boxplot(x='Model', y=metric, data=metrics_df, hue='Model type', palette=model_type_palette)

        # Enhance the plot
        plt.title(f'Box Plot with {metric} values 10 Fold CV')
        plt.legend(title='Fold', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        plt.savefig(pdf, format='pdf')  
    pdf.close()



def plot_models_boxplot(metrics_df, save_dir=None):
    """
    Plots the performance metrics of different models on a boxplot chart.

    :param metrics_df: The dataframe containing the performance metrics of different models.
    :type metrics_df: pd.DataFrame
    :param save_dir: The directory where the plot will be saved. If None, the plot will be displayed without being saved.
    :type save_dir: str, optional

    :output: Matplotlib figures with the boxplot charts.
    """

    if 'AUC' in metrics_df.columns:
        metrics_list = ['AUC', 'Accuracy', 'Precision', 'Recall','f1 score', 'Balanced accuracy', 'MCC', 'Kappa']
    else:
        metrics_list = ["R2 Score" , "MSE", "RMSE", "MAE"]

    #model_type_palette = {'Non-linear': '#F96167', 'linear': '#F9E795'}
    model_type_palette = {'Non-linear': '#735DA5', 'linear': '#D3C5E5'}
    
    for metric in metrics_list:
        # Create a new figure for each metric
        plt.figure(figsize=(10, 6))

        # Create the box plot
        sns.stripplot(x='Model', y=metric, data=metrics_df, hue='Fold', dodge=True, jitter=True, marker='o', alpha=0.8, palette='dark:gray')
        sns.boxplot(x='Model', y=metric, data=metrics_df, hue='Model type', palette=model_type_palette)

        # Add jitter for the Fold information with stripplot
        

        # Enhance the plot
        plt.title(f'Box Plot with {metric} values for different methods at different folds')
        plt.legend(title='Fold', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        plt.tight_layout()


        # Create the directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save the plot to the directory
        plt.savefig(os.path.join(save_dir, f"metrics_{metric}.png"))







from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

def plot_model_metrics(metrics_df, save_dir=None):
    """
    Plot the performance metrics of different models on a bar chart.

    :param metrics_df: A pandas DataFrame with the model names, types, and metrics as columns.
    :type metrics_df: pd.DataFrame
    :param save_dir: The directory to save the plots. If None, the plots are shown instead.
    :type save_dir: str or None

    .. note::  
        The metrics are : AUC, Accuracy, Precision, Recall, f1 score, Balanced accuracy, MCC and Kappa for classification. R2 Score, RMSE, MAE and MSE for regression.
        
        - **AUC**: The area under the receiver operating characteristic curve, which measures the trade-off between the true positive rate and the false positive rate.
        - **Accuracy**: The proportion of correctly classified instances among the total number of instances.
        - **Precision**: The proportion of correctly classified positive instances among the predicted positive instances.
        - **Recall**: The proportion of correctly classified positive instances among the actual positive instances.
        - **f1 score**: The harmonic mean of precision and recall, which balances both metrics.
        - **Balanced accuracy**: The average of recall obtained on each class, which is useful for imbalanced datasets.
        - **MCC**: The Matthews correlation coefficient, which measures the quality of a binary classification, taking into account true and false positives and negatives.
        - **Kappa**: The Cohen's kappa coefficient, which measures the agreement between two raters, adjusting for the chance agreement.
        - **R2 Score**: The statistical measure that indicates the proportion of variance in the dependent variable that's predictable from the independent variable(s). 
        - **RMSE (Root Mean Squared Error)**: The square root of the average of the squared differences between the actual and predicted values, used as a loss function in regression problems.
        - **MAE (Mean Absolute Error)**: The average of the absolute differences between the actual and predicted values, used as a loss function in regression problems.
        - **MSE (Mean Squared Error)**: The average of the squared differences between the actual and predicted values, used as a loss function in regression problems.
    """

    if 'AUC' in metrics_df.columns:
        metrics_list = ['AUC', 'Accuracy', 'Precision', 'Recall','f1 score', 'Balanced accuracy', 'MCC', 'Kappa']
    else:
        metrics_list = ["R2 Score" , "MSE"]
    for metric in metrics_list:
        fig, ax = plt.subplots()
        # Get unique values from the 'Model type' column
        unique_categories = metrics_df['Model type'].unique()
        # Create a dictionary mapping each unique category to a unique color
        color_mapping = {category: plt.cm.viridis(i / len(unique_categories)) for i, category in enumerate(unique_categories)}
        # Map the colors to the original 'Model type' column and convert to RGBA format
        metrics_df['Color'] = metrics_df['Model type'].map(color_mapping).apply(to_rgba)
        # Create a bar plot
        bars = ax.bar(metrics_df['Model'], metrics_df[metric], color=metrics_df['Color'], label=metrics_df['Model type'])
        # Add labels and title
        ax.set_xlabel('Models')
        ax.set_ylabel(f'{metric} Values')
        ax.set_title(f'Bar Plot with {metric} values for different methods')
        # Rotate x-axis labels by 45 degrees
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        # Add legend
        #ax.legend()
        # Adjust layout
        fig.tight_layout()
        # Save the figure
        if save_dir is not None:
            # create the directory if it does not exist
            if not os.path.exists(save_dir): 
                os.makedirs(save_dir) 
            # save the plot to the directory
            fig.savefig(f"{save_dir}/metrics{metric}.png")
        else:
            # show the plot
            plt.show()

    legend_handles = [Line2D([0], [0], color=color, lw=4, label=category) for category, color in color_mapping.items()]

    # Create legend without a plot
    legend_fig, legend_ax = plt.subplots(figsize=(6, 1))
    legend_ax.set_axis_off()
    legend_ax.legend(handles=legend_handles, loc='center', ncol=len(unique_categories))

    # Save the legend as a separate image
    if save_dir == None:
        legend_fig.show()
    else:
        legend_fig.savefig(f"{save_dir}/legend.png", bbox_inches='tight')

# importing package 
import matplotlib.pyplot as plt 
import numpy as np 



def plot_models_barplot(matrics,save_dir):
    (train_df, val_df, test_df) = matrics
    # train_df.set_index('Model',inplace=True)
    # test_df.set_index('Model',inplace=True)
    # val_df.set_index('Model',inplace=True)
    if 'AUC' in train_df.columns:
        metrics_list = ['AUC', 'Accuracy', 'Precision', 'Recall','f1 score', 'Balanced accuracy', 'MCC', 'Kappa']
    else:
        metrics_list = ["R2 Score" , "MSE", "RMSE", "MAE"]
    for met in metrics_list:
        plt.figure(figsize=(16, 8), dpi=80)
        # Assuming dfs_train, dfs_test, and dfs_val are your dataframes for train, test, and validation data
        # Concatenate AUC values from train, test, and validation dataframes
        df_auc = pd.concat([train_df[met], val_df[met], test_df[met]], axis=1)
        df_auc.columns = ['Train', 'Validation', 'Test', ]  # Rename columns
        df_auc['Model'] = train_df['Model']
        # Plot the grouped bar chart
        threshold = 10  # Example threshold
        # Filter out rows where any value exceeds the threshold
        df_auc = df_auc[(df_auc['Train'] < threshold) & (df_auc['Validation'] < threshold) & (df_auc['Test'] < threshold)]
        threshold = -10 
        df_auc = df_auc[(df_auc['Train'] > threshold) & (df_auc['Validation'] > threshold) & (df_auc['Test'] > threshold)]   
        # create data 
        x = np.arange(len(df_auc)) 
        y1 = df_auc['Train']
        y2 = df_auc['Validation']
        y3 = df_auc['Test']
        width = 0.2
        # plot data in grouped manner of bar type 
        plt.bar(x-0.2, y1, width, color='cyan') 
        plt.bar(x, y2, width, color='orange') 
        plt.bar(x+0.2, y3, width, color='green') 
        plt.xticks(x, df_auc['Model']) 
        plt.xlabel("Models") 
        plt.ylabel(met) 
        plt.legend(["Training", "Validation", "Testing"]) 
        #p#lt.show() 
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        #sns.barplot(data=df_auc, palette="Set3", linewidth=2.5)
        # Set labels and title
        plt.xlabel('Models')
        plt.ylabel(met)
        plt.title(met+' Comparison')
        # Show plot
        plt.xticks(rotation=45)
        plt.tight_layout()
        #plt.show()
        # Save or show the plot
        # Create the directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save the plot to the directory
        plt.savefig(os.path.join(save_dir, f"metrics_{met}.png"))
        plt.close()
