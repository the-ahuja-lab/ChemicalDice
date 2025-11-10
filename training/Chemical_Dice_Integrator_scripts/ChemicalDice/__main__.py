import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib


import pandas as pd

import os



def calculate_descriptors(input_filename):
    from ChemicalDice import smiles_preprocess, bioactivity, chemberta, Grover, ImageMol, chemical, quantum
    # Read the CSV file
    df = pd.read_csv(input_filename)
    # Select the required columns and remove duplicates
    # Rename the columns
    #df.columns = ['SMILES']
    print("Removing duplicte SMILES")

    if 'smiles' in df.columns:
        df.rename(columns={'smiles': 'SMILES'}, inplace=True)
        print("Column 'smiles' renamed to 'SMILES'")
    elif 'SMILES' in df.columns:
        print("Column 'SMILES' already present")
    else:
        raise("No 'smiles' or 'SMILES' column found in the file")
    # Print the DataFrame
    print("Your input")
    print(df)
    input_wrong = True
    while input_wrong:
        user_ans = input("Please varify if your input is correct. Start calculation (yes/no):")
        if user_ans=='yes':
            try:
                os.mkdir("Chemicaldice_data")
            except:
                print("Calculating descriptors ....")
                print("Generating mol2 files from SMILES")
                # Write the DataFrame to a new CSV file
                if os.path.exists("Chemicaldice_data/mopac.csv"):
                    pass
                else:
                    df.to_csv("Chemicaldice_data/input_data.csv", index=False)

                smiles_preprocess.create_mol2_files(input_file = "Chemicaldice_data/input_data.csv")
                
                if os.path.exists("Chemicaldice_data/mopac.csv"):
                    pass
                else:
                    print("Calculating Qunatum descriptors from MOPAC")
                    quantum.get_mopac_prerequisites()
                    quantum.descriptor_calculator(input_file = "Chemicaldice_data/input_data.csv", output_file="Chemicaldice_data/mopac.csv")

    
                if os.path.exists("Chemicaldice_data/mordred.csv"):
                    pass
                else:
                    print("Calculating PhysioChemical descriptors from Mordred")
                    smiles_preprocess.create_sdf_files(input_file = "Chemicaldice_data/input_data.csv")
                    chemical.descriptor_calculator(input_file = "Chemicaldice_data/input_data.csv", output_file="Chemicaldice_data/mordred.csv") 

                if os.path.exists("Chemicaldice_data/Chemberta.csv"):
                    pass
                else:
                    print("Generating Large language model from ChemBERTa")
                    smiles_preprocess.add_canonical_smiles(input_file = "Chemicaldice_data/input_data.csv")
                    chemberta.smiles_to_embeddings(input_file = "Chemicaldice_data/input_data.csv", output_file = "Chemicaldice_data/Chemberta.csv")

                if os.path.exists("Chemicaldice_data/ImageMol.csv"):
                    pass
                else:
                    print("Generating image embeddings from imageMol")
                    ImageMol.image_to_embeddings(input_file = "Chemicaldice_data/input_data.csv", output_file_name="Chemicaldice_data/ImageMol.csv")

                if os.path.exists("Chemicaldice_data/Grover.csv"):
                    pass
                else:
                    print("Generating graph embeddings from Grover")
                    Grover.get_embeddings(input_file = "Chemicaldice_data/input_data.csv",  output_file_name="Chemicaldice_data/Grover.csv")

                if os.path.exists("Chemicaldice_data/Signaturizer.csv"):
                    pass
                else:
                    try:
                        bioactivity.calculate_descriptors(input_file = "Chemicaldice_data/input_data.csv", output_file = "Chemicaldice_data/Signaturizer.csv")
                    except:
                        print("There is an error in Signaturizer")


                user_input=input("Chemicaldice_data already exist do you want to replace the files (yes/no)")
                if user_input == "yes":
                    pass
                else:
                    exit()
            print("Calculating descriptors ....")
            print("Generating mol2 files from SMILES")
            # Write the DataFrame to a new CSV file
            df.to_csv("Chemicaldice_data/input_data.csv", index=False)
            smiles_preprocess.create_mol2_files(input_file = "Chemicaldice_data/input_data.csv")

            print("Calculating Qunatum descriptors from MOPAC")
            quantum.get_mopac_prerequisites()
            quantum.descriptor_calculator(input_file = "Chemicaldice_data/input_data.csv", output_file="Chemicaldice_data/mopac.csv")


            print("Calculating PhysioChemical descriptors from Mordred")
            smiles_preprocess.create_sdf_files(input_file = "Chemicaldice_data/input_data.csv")
            chemical.descriptor_calculator(input_file = "Chemicaldice_data/input_data.csv", output_file="Chemicaldice_data/mordred.csv")

            print("Generating Large language model from ChemBERTa")
            smiles_preprocess.add_canonical_smiles(input_file = "Chemicaldice_data/input_data.csv")
            chemberta.smiles_to_embeddings(input_file = "Chemicaldice_data/input_data.csv", output_file = "Chemicaldice_data/Chemberta.csv")
            
            print("Generating image embeddings from imageMol")
            ImageMol.image_to_embeddings(input_file = "Chemicaldice_data/input_data.csv", output_file_name="Chemicaldice_data/ImageMol.csv")
            
            print("Generating graph embeddings from Grover")
            Grover.get_embeddings(input_file = "Chemicaldice_data/input_data.csv",  output_file_name="Chemicaldice_data/Grover.csv")

            try:
                bioactivity.calculate_descriptors(input_file = "Chemicaldice_data/input_data.csv", output_file = "Chemicaldice_data/Signaturizer.csv")
            except:
                print("There is an error in Signaturizer")
            
            print("Descriptors calculation done files saved in : Chemicaldice_data" )
            print("Generating Generating bioactivity signatures from Signaturizer")

            input_wrong = False 
        elif user_ans == 'no':
            input_wrong = False 
        else:
            pass

def load_data(directory):
    """
    Load data from CSV files in the given directory.
    """
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            name = os.path.splitext(filename)[0]
            data[name] = pd.read_csv(os.path.join(directory, filename))
    return data



def train_model(input_dir, input_file, n_components=10, methods = ['AER','plsda','pca','cca'], n_folds = 10):
    from ChemicalDice.fusionData import fusionData
    from ChemicalDice.plot_data import plot_model_boxplot
    
    cmb = os.path.join(input_dir, "Chemberta.csv")
    gro = os.path.join(input_dir, "Grover.csv")
    mop = os.path.join(input_dir, "mopac.csv")
    mod = os.path.join(input_dir, "mordred.csv")
    sig = os.path.join(input_dir, "Signaturizer.csv")
    iml = os.path.join(input_dir, "ImageMol.csv")
    label_data_path = os.path.join(input_dir, "input_data.csv")
    label_data = pd.read_csv(label_data_path)
    if "labels" in label_data.columns:
        label_data.drop("labels", axis=1, inplace=True)

    df = pd.read_csv(input_file)
    if 'label' in df.columns:
        df.rename(columns={'label': 'labels'}, inplace=True)
        print("Column 'label' renamed to 'labels'")
    elif 'Labels' in df.columns:
        df.rename(columns={'Labels': 'labels'}, inplace=True)
        print("Column 'Labels' renamed to 'labels'")
    elif 'labels' in df.columns:
        print("Column 'labels' already present")
    else:
        raise ValueError("No 'labels', 'Labels', or 'Label' column found in the file")
    
    label_data = pd.merge(df, label_data, left_on="SMILES", right_on="SMILES")
    label_data.to_csv(label_data_path, index=False)

    unique_labels = df['labels'].unique()

    if len(unique_labels) == 2 and all(isinstance(label, (int, float)) for label in unique_labels):
        regression = False
        print("Binary labels detected")
    elif all(isinstance(label, int) for label in unique_labels):
        regression = False
        print("Multiclass labels detected")
    elif all(isinstance(label, float) for label in unique_labels):
        regression = True
        print("Continuous labels detected")
    else:
        print("Unknown label type")

    data_paths = {
        "Chemberta": cmb,
        "Grover": gro,
        "mopac": mop,
        "mordred": mod,
        "Signaturizer": sig,
        "ImageMol": iml
    }

    fusiondata = fusionData(data_paths=data_paths, label_file_path=label_data_path, label_column="labels", id_column="id")

    fusiondata.keep_common_samples()

    print("\nMissing Values in the data")
    fusiondata.ShowMissingValues()
    print("Removing empty columns in the data")
    fusiondata.remove_empty_features(threshold=100)
    print("\nMissing Values in the data")
    fusiondata.ShowMissingValues()

    print("Imputing the data")
    fusiondata.ImputeData(method="knn")
    fusiondata.ShowMissingValues()

    print("Normalising the data")
    fusiondata.scale_data(scaling_type='standardize')

    print("Evaluating the model ...")
    fusiondata.evaluate_fusion_models_nfold(n_components=n_components,
                                            methods=methods,
                                            n_folds=n_folds,
                                            regression=regression)
    
    print("Accuracy metrics")
    print(fusiondata.Accuracy_metrics)
    top_models = fusiondata.top_combinations
    print("Plotting the accuracy metrics ...")
    plot_model_boxplot(top_models, save_dir='Chemicaldice_plots')
    print("Plots are saved in Chemicaldice_plots")
    return fusiondata


def predict_model(input_dir,input_file):
    from ChemicalDice.predData import fusionData
    from ChemicalDice.plot_data import plot_model_boxplot
    """
    Train a Random Forest classifier using the given data.
    """
    
    cmb = os.path.join(input_dir,"Chemberta.csv")
    gro = os.path.join(input_dir,"Grover.csv")
    mop = os.path.join(input_dir,"mopac.csv")
    mod = os.path.join(input_dir,"mordred.csv")
    sig = os.path.join(input_dir,"Signaturizer.csv")
    iml = os.path.join(input_dir,"ImageMol.csv")

    data_paths = {
        "Chemberta":cmb,
        "Grover":gro,
        "mopac":mop,
        "mordred":mod,
        "Signaturizer":sig,
        "ImageMol": iml
    }

    fusiondata = fusionData(data_paths = data_paths, smile_file_path=input_file, id_column="id")

    fusiondata.keep_common_samples()

    print("\nMissing Values in the data")
    fusiondata.ShowMissingValues()
    print("Removing empty columns in the data")
    fusiondata.remove_empty_features(threshold=100)
    print("\nMissing Values in the data")
    fusiondata.ShowMissingValues()

    # Imputing values with missing values
    print("Imputing the data")
    fusiondata.ImputeData(method="knn")
    fusiondata.ShowMissingValues()

    # Standardize data
    print("Normalising the data")
    fusiondata.scale_data(scaling_type = 'standardize')



    return fusiondata


def calculate_descriptors_pred(input_filename):
    from ChemicalDice import smiles_preprocess, bioactivity, chemberta, Grover, ImageMol, chemical, quantum
    # Read the CSV file
    df = pd.read_csv(input_filename)
    # Select the required columns and remove duplicates
    # Rename the columns
    #df.columns = ['SMILES']
    print("Removing duplicte SMILES")

    if 'smiles' in df.columns:
        df.rename(columns={'smiles': 'SMILES'}, inplace=True)
        print("Column 'smiles' renamed to 'SMILES'")
    elif 'SMILES' in df.columns:
        print("Column 'SMILES' already present")
    else:
        raise("No 'smiles' or 'SMILES' column found in the file")
    # Print the DataFrame
    print("Your input")
    print(df)
    input_wrong = True
    while input_wrong:
        user_ans = input("Please varify if your input is correct. Start calculation (yes/no):")
        if user_ans=='yes':
            try:
                os.mkdir("Chemicaldice_predictiondata")
            except:
                print("Calculating descriptors ....")
                print("Generating mol2 files from SMILES")
                # Write the DataFrame to a new CSV file
                if os.path.exists("Chemicaldice_predictiondata/mopac.csv"):
                    pass
                else:
                    df.to_csv("Chemicaldice_predictiondata/input_data.csv", index=False)

                smiles_preprocess.create_mol2_files(input_file = "Chemicaldice_predictiondata/input_data.csv")
                
                if os.path.exists("Chemicaldice_predictiondata/mopac.csv"):
                    pass
                else:
                    print("Calculating Qunatum descriptors from MOPAC")
                    quantum.get_mopac_prerequisites()
                    quantum.descriptor_calculator(input_file = "Chemicaldice_predictiondata/input_data.csv", output_file="Chemicaldice_predictiondata/mopac.csv")

    
                if os.path.exists("Chemicaldice_predictiondata/mordred.csv"):
                    pass
                else:
                    print("Calculating PhysioChemical descriptors from Mordred")
                    smiles_preprocess.create_sdf_files(input_file = "Chemicaldice_predictiondata/input_data.csv")
                    chemical.descriptor_calculator(input_file = "Chemicaldice_predictiondata/input_data.csv", output_file="Chemicaldice_predictiondata/mordred.csv") 

                if os.path.exists("Chemicaldice_predictiondata/Chemberta.csv"):
                    pass
                else:
                    print("Generating Large language model from ChemBERTa")
                    smiles_preprocess.add_canonical_smiles(input_file = "Chemicaldice_predictiondata/input_data.csv")
                    chemberta.smiles_to_embeddings(input_file = "Chemicaldice_predictiondata/input_data.csv", output_file = "Chemicaldice_predictiondata/Chemberta.csv")

                if os.path.exists("Chemicaldice_predictiondata/ImageMol.csv"):
                    pass
                else:
                    print("Generating image embeddings from imageMol")
                    ImageMol.image_to_embeddings(input_file = "Chemicaldice_predictiondata/input_data.csv", output_file_name="Chemicaldice_predictiondata/ImageMol.csv")

                if os.path.exists("Chemicaldice_predictiondata/Grover.csv"):
                    pass
                else:
                    print("Generating graph embeddings from Grover")
                    Grover.get_embeddings(input_file = "Chemicaldice_predictiondata/input_data.csv",  output_file_name="Chemicaldice_predictiondata/Grover.csv")

                if os.path.exists("Chemicaldice_predictiondata/Signaturizer.csv"):
                    pass
                else:
                    try:
                        bioactivity.calculate_descriptors(input_file = "Chemicaldice_predictiondata/input_data.csv", output_file = "Chemicaldice_predictiondata/Signaturizer.csv")
                    except:
                        print("There is an error in Signaturizer")


                user_input=input("Chemicaldice_predictiondata already exist do you want to replace the files (yes/no)")
                if user_input == "yes":
                    pass
                else:
                    exit()
            print("Calculating descriptors ....")
            print("Generating mol2 files from SMILES")
            # Write the DataFrame to a new CSV file
            df.to_csv("Chemicaldice_predictiondata/input_data.csv", index=False)
            smiles_preprocess.create_mol2_files(input_file = "Chemicaldice_predictiondata/input_data.csv")

            print("Calculating Qunatum descriptors from MOPAC")
            quantum.get_mopac_prerequisites()
            quantum.descriptor_calculator(input_file = "Chemicaldice_predictiondata/input_data.csv", output_file="Chemicaldice_predictiondata/mopac.csv")


            print("Calculating PhysioChemical descriptors from Mordred")
            smiles_preprocess.create_sdf_files(input_file = "Chemicaldice_predictiondata/input_data.csv")
            chemical.descriptor_calculator(input_file = "Chemicaldice_predictiondata/input_data.csv", output_file="Chemicaldice_predictiondata/mordred.csv")

            print("Generating Large language model from ChemBERTa")
            smiles_preprocess.add_canonical_smiles(input_file = "Chemicaldice_predictiondata/input_data.csv")
            chemberta.smiles_to_embeddings(input_file = "Chemicaldice_predictiondata/input_data.csv", output_file = "Chemicaldice_predictiondata/Chemberta.csv")
            
            print("Generating image embeddings from imageMol")
            ImageMol.image_to_embeddings(input_file = "Chemicaldice_predictiondata/input_data.csv", output_file_name="Chemicaldice_predictiondata/ImageMol.csv")
            
            print("Generating graph embeddings from Grover")
            Grover.get_embeddings(input_file = "Chemicaldice_predictiondata/input_data.csv",  output_file_name="Chemicaldice_predictiondata/Grover.csv")

            try:
                bioactivity.calculate_descriptors(input_file = "Chemicaldice_predictiondata/input_data.csv", output_file = "Chemicaldice_predictiondata/Signaturizer.csv")
            except:
                print("There is an error in Signaturizer")
            
            print("Descriptors calculation done files saved in : Chemicaldice_predictiondata" )
            print("Generating Generating bioactivity signatures from Signaturizer")

            input_wrong = False 
        elif user_ans == 'no':
            input_wrong = False 
        else:
            pass

def predict(model_file, input_file):
    """
    Predict property for SMILES in the input file using the trained model.
    """
    calculate_descriptors_pred(input_file)
    fd = joblib.load(model_file)
    fusiondata = predict_model(input_dir = "Chemicaldice_predictiondata",input_file= input_file)
    predictions = fusiondata.predict_fusion_model(fd, n_components=fd.n_components,regression = fd.regression, AER_dim=4096)
    input_data = pd.read_csv(input_file)
    input_data['Predicted_Label'] = predictions
    input_data.to_csv("Prediction_output.csv",index=False)
    print(input_data)
    print("Predictions are saved to Prediction_output.csv")



def main():
    parser = argparse.ArgumentParser(
        description="""Welcome to ChemicalDice CLI
        
        Calculate descriptors, train and predict using SMILES data""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('mode', choices=['calculate', 'train', 'predict'], help='Mode: calculate, train or predict')
    parser.add_argument('--input_file', help="""Input CSV file with SMILES for descriptor calculation or
                        Input CSV file with SMILES and labels for training
                        Input CSV file with SMILES for prediction""")
    parser.add_argument('--input_dir', help='Directory containing CSV files for training')
    parser.add_argument('--model', help='Path to trained model file (only required for predict mode)')
    parser.add_argument('--n_components', type=int, default=10, help='Number of components for fusion model evaluation')
    parser.add_argument('--methods', nargs='+', default=['pca', 'cca', 'plsda'], help='Methods for fusion model evaluation')
    parser.add_argument('--n_folds', type=int, default=10, help='Number of folds for cross-validation')

    args = parser.parse_args()

    print("ChemicalDice CLI")
    print("\n")

    if args.mode == 'calculate':
        if not args.input_file:
            parser.error("--input_file is required in calculate mode")
        calculate_descriptors(args.input_file)
    elif args.mode == 'train':
        if not args.input_dir:
            parser.error("--input_dir is required in train mode")
        if not args.input_file:
            parser.error("--input_file is required for input CSV file with SMILES and labels for training")
        model = train_model(args.input_dir, args.input_file, args.n_components, args.methods, args.n_folds)
        joblib.dump(model, 'trained_model.pkl')
        print("Model trained and saved as 'trained_model.pkl'")
    elif args.mode == 'predict':
        if not args.input_file:
            parser.error("--input_file is required in predict mode")
        if not args.model:
            parser.error("--model is required in predict mode")
        predict(args.model, args.input_file)

    print("\nSummary of Arguments")
    print(f"Mode: {args.mode}")
    print(f"Input File: {args.input_file if args.input_file else 'None'}")
    print(f"Input Directory: {args.input_dir if args.input_dir else 'None'}")
    print(f"Model: {args.model if args.model else 'None'}")
    print(f"Number of Components: {args.n_components}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Number of Folds: {args.n_folds}")

    print("\nProcessing complete.\n")


if __name__ == '__main__':
    main()


