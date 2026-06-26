import pandas as pd
from openbabel import pybel
from rdkit import Chem
import subprocess
from openbabel import openbabel
import os
import re
import fileinput
import urllib.request
import tempfile
from tqdm import tqdm

import tarfile

def extract_tar_gz(file_path):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall()



import requests

def download_file(url, filename):
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

# important function for mopac
def smile_to_mol2_uff(smile, steps, filename):
        mymol = pybel.readstring("smi", smile)
        #print(mymol.molwt)
        mymol.make3D(steps=steps,forcefield='uff')
        mymol.write(format="mol2",filename=filename,overwrite=True)

def smile_to_mol2_mmff(smile, steps, filename):
        mymol = pybel.readstring("smi", smile)
        #print(mymol.molwt)
        mymol.make3D(steps=steps,forcefield='mmff94')
        mymol.write(format="mol2",filename=filename,overwrite=True)

def ReadFile(filename):
    inputdict={}
    f=open(filename,'r')
    lines_list=f.read().split("\n")
    #print(lines_list)
    lines_list=[lines.strip() for lines in lines_list]
    #print(lines_list)
    descriptor_list=['HEAT OF FORMATION','GRADIENT NORM','DIPOLE','NO. OF FILLED LEVELS','IONIZATION POTENTIAL','HOMO LUMO ENERGIES (EV)','MOLECULAR WEIGHT','COSMO AREA','COSMO VOLUME','SCF CALCULATIONS']
    descriptors_values=[]
    for descriptors in descriptor_list:
        for lines in lines_list:
            if lines.startswith(descriptors):
                desc=re.findall(r"[-+]?(?:\d*\.*\d+)", lines)
                desc=[float(des) for des in desc]
                descriptors_values.append(desc)
    return(descriptors_values)



def ReadFile(filename, mol_name, smile):
    inputdict={}
    inputdict['mol_id']=mol_name
    inputdict['SMILES']=smile
    f=open(filename,'r')
    lines_list=f.read().split("\n")
    #print(lines_list)
    lines_list=[lines.strip() for lines in lines_list]
    #print(lines_list)
    descriptor_list=['HEAT OF FORMATION','GRADIENT NORM','DIPOLE','NO. OF FILLED LEVELS','IONIZATION POTENTIAL','HOMO LUMO ENERGIES (EV)','MOLECULAR WEIGHT','COSMO AREA','COSMO VOLUME','SCF CALCULATIONS']
    descriptors_values=[]
    for descriptors in descriptor_list:
        for lines in lines_list:
            if lines.startswith(descriptors):
                desc=re.findall(r"[-+]?(?:\d*\.*\d+)", lines)
                desc=[float(des) for des in desc]
                descriptors_values.append(desc)
    # HEAT OF FORMATION
    inputdict['Hf']=descriptors_values[0][1]
    # GRADIENT NORM
    inputdict['GN']=descriptors_values[1][0]
    # GRADIENT NORM PER ATOM
    inputdict['GNPA']=descriptors_values[1][1]
    # DIPOLE
    inputdict['mu']=descriptors_values[2][0]
    # NO. OF FILLED LEVELS
    inputdict['NFL']=descriptors_values[3][0]
    # IONIZATION POTENTIAL
    inputdict['IP']=descriptors_values[4][0]
    # HOMO LUMO ENERGIES (EV)
    inputdict['EHomo']=descriptors_values[5][0]
    inputdict['ELumo']=descriptors_values[5][1]
    # MOLECULAR WEIGHT
    inputdict['Mw']=descriptors_values[6][0]
    # COSMO AREA
    inputdict['CoArea']=descriptors_values[7][0]
    # COSMO VOLUME
    inputdict['CoVolume']=descriptors_values[8][0]
    f.close()
    return inputdict


def CalculateBasicQC(inputdict):
    EHomo=inputdict['EHomo']
    ELumo=inputdict['ELumo']
    dict={}
    dict.update(inputdict)
    dict['ChemicalPotential']=(ELumo+EHomo)/2.0
    dict['ChemicalHardness']=(ELumo-EHomo)/2.0
    dict['ChemicalSoftness']=1-dict['ChemicalHardness']
    dict['Electrophilicity']=(dict['ChemicalPotential']**2)/(2*dict['ChemicalHardness'])
    dict['fHL']=EHomo/ELumo
    dict['EA']=-ELumo
    dict['xmu']=(-ELumo-EHomo)/2.0
    dict['S']=2./(ELumo-EHomo)
    dict['GAP']=ELumo-EHomo
    return dict


def calculate_formalCharge_Multiplicity(file_path, file_format):
    # Create an Open Babel molecule object
    mol = openbabel.OBMol()
    # Read the input chemical structure from the file
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat(file_format)
    obConversion.ReadFile(mol, file_path)
    formal_charge = mol.GetTotalCharge()
    multiplicity = mol.GetTotalSpinMultiplicity()
    return (formal_charge,multiplicity)
