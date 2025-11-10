
# http://openmopac.net/mopac-22.0.6-win.exe



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



from ChemicalDice.quantum_need import *

import requests



MORSE_EXE = None

def get_mopac_prerequisites():
    morse_exe_file= "3dmorse.exe"
    # URL of the file to be downloaded
    # Name of the file to save as
    if os.name == "nt":
      result =subprocess.check_output(['mopac', '--version'])
      
      if result == b'MOPAC version 22.1.1 commit e7a5c3f6a4450286a1f770452b44e53be426fa8b\r\n':
         print("Mopac is already installed ") 
      else:
         print("For windows please install MOPAC. Use following link to download exe file to install MOPAC file https://github.com/openmopac/mopac/releases/download/v22.1.1/mopac-22.1.1-win.ex")
         raise ImportError("MOPAC is not installed.")

      # url = "https://github.com/openmopac/mopac/releases/download/v22.1.1/mopac-22.1.1-win.exe"
      # filename = "mopac-22.1.1-win.exe"
      # download_file(url, filename)
      
      pass
    else:
      url = "https://github.com/openmopac/mopac/releases/download/v22.1.1/mopac-22.1.1-linux.tar.gz"
      filename = "mopac-22.1.1-linux.tar.gz"
      download_file(url, filename)
      file_path = "mopac-22.1.1-linux.tar.gz"
      extract_tar_gz(file_path)
      print("Mopac is downloaded")

    if os.name == "nt":
        if os.path.exists(morse_exe_file):
            print("Provided 3dmorse.exe file is present")
        else:
            raise ImportError("For windows please download 3dmorse.exe . Use the following link to download and provide it by keep the file in current directory. https://github.com/devinyak/3dmorse")
    else:
        urllib.request.urlretrieve("https://raw.githubusercontent.com/devinyak/3dmorse/master/3dmorse.cpp", "3dmorse.cpp")


        with fileinput.FileInput("3dmorse.cpp", inplace=True, backup='.bak') as file:
            for line in file:
                print(line.replace('<tchar.h>', '"tchar.h"'), end='')
                #print(line.replace('//Some constants', '#include <math.h>'), end='')
        with fileinput.FileInput("3dmorse.cpp", inplace=True, backup='.bak') as file:
            for line in file:
                print(line.replace('//Some constants', '#include <math.h>'), end='')

        urllib.request.urlretrieve("https://home.cs.colorado.edu/~main/cs1300-old/include/tchar.h", "tchar.h")

        with fileinput.FileInput("tchar.h", inplace=True, backup='.bak') as file:
            for line in file:
                print(line.replace('<_mingw.h>', '"_mingw.h"'), end='')

        urllib.request.urlretrieve("https://home.cs.colorado.edu/~main/cs1300-old/include/_mingw.h", "_mingw.h")
        subprocess.run(['g++', '3dmorse.cpp', '-o', '3dmorse'])
        for file in ["3dmorse.cpp.bak","_mingw.h","tchar.h","3dmorse.cpp","tchar.h.bak"]:
            os.remove(file)
        print("Morse is compiled")


import psutil

def checkIfProcessRunning(processName):
    '''
    Check if there is any running process that contains the given name processName.
    '''
    #Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False;

import multiprocessing
cpu_to_use = multiprocessing.cpu_count() * 0.5
cpu_to_use = int(cpu_to_use)

def descriptor_calculator(input_file,output_file, output_dir = "temp_data/mopfiles",ncpu=cpu_to_use):
  """
  Calculate molecular descriptors using MOPAC and Morse for a list of molecules.

  This function performs the following steps:
  1. Reads the input file containing molecule information.
  2. Generates MOPAC input files for each molecule.
  3. Runs MOPAC calculations to obtain quantum chemical descriptors.
  4. Runs 3D Morse calculations to obtain Morse descriptors.
  5. Writes the calculated descriptors to the output file.

  Parameters
  ----------
  input_file : str
      Path to the input CSV file containing molecule information.
      The CSV file should have the following columns: 'mol2_files', 'id', 'SMILES'.
  output_file : str
      Path to the output CSV file where the descriptors will be written.
  output_dir : str, optional
      Directory where MOPAC input and output files will be stored.
      Default is "temp_data/mopfiles".
  ncpu : int, optional
      Number of CPU cores to use for MOPAC calculations.
      Default is the value of the `cpu_to_use` variable.

  Notes
  -----
  The input CSV file should have the following columns:
  - 'mol2_files': Path to the mol2 file for each molecule.
  - 'id': Unique identifier for each molecule.
  - 'SMILES': SMILES string representation of each molecule.

  The function creates MOPAC input files for each molecule and runs MOPAC calculations
  to obtain quantum chemical descriptors. It then runs 3D Morse calculations to obtain
  Morse descriptors. The calculated descriptors are written to the output CSV file.

  """
  start_from = 0
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  n_threads = ncpu
  n=0
  smiles_df = pd.read_csv(input_file)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  mol2file_name_list = smiles_df['mol2_files']
  id_list = smiles_df['id']
  smiles_list = smiles_df['SMILES']
  if start_from < 1:
    f = open(output_file,"w")
    f.write("id,SMILES,Hf,GN,GNPA,mu,NFL,IP,EHomo,ELumo,Mw,CoArea,CoVolume,ChemicalPotential,ChemicalHardness,ChemicalSoftness,Electrophilicity,fHL,EA,xmu,S,GAP,Mor01u,Mor02u,Mor03u,Mor04u,Mor05u,Mor06u,Mor07u,Mor08u,Mor09u,Mor10u,Mor11u,Mor12u,Mor13u,Mor14u,Mor15u,Mor16u,Mor17u,Mor18u,Mor19u,Mor20u,Mor21u,Mor22u,Mor23u,Mor24u,Mor25u,Mor26u,Mor27u,Mor28u,Mor29u,Mor30u,Mor31u,Mor32u,Mor01m,Mor02m,Mor03m,Mor04m,Mor05m,Mor06m,Mor07m,Mor08m,Mor09m,Mor10m,Mor11m,Mor12m,Mor13m,Mor14m,Mor15m,Mor16m,Mor17m,Mor18m,Mor19m,Mor20m,Mor21m,Mor22m,Mor23m,Mor24m,Mor25m,Mor26m,Mor27m,Mor28m,Mor29m,Mor30m,Mor31m,Mor32m,Mor01v,Mor02v,Mor03v,Mor04v,Mor05v,Mor06v,Mor07v,Mor08v,Mor09v,Mor10v,Mor11v,Mor12v,Mor13v,Mor14v,Mor15v,Mor16v,Mor17v,Mor18v,Mor19v,Mor20v,Mor21v,Mor22v,Mor23v,Mor24v,Mor25v,Mor26v,Mor27v,Mor28v,Mor29v,Mor30v,Mor31v,Mor32v,Mor01p,Mor02p,Mor03p,Mor04p,Mor05p,Mor06p,Mor07p,Mor08p,Mor09p,Mor10p,Mor11p,Mor12p,Mor13p,Mor14p,Mor15p,Mor16p,Mor17p,Mor18p,Mor19p,Mor20p,Mor21p,Mor22p,Mor23p,Mor24p,Mor25p,Mor26p,Mor27p,Mor28p,Mor29p,Mor30p,Mor31p,Mor32p,Mor01e,Mor02e,Mor03e,Mor04e,Mor05e,Mor06e,Mor07e,Mor08e,Mor09e,Mor10e,Mor11e,Mor12e,Mor13e,Mor14e,Mor15e,Mor16e,Mor17e,Mor18e,Mor19e,Mor20e,Mor21e,Mor22e,Mor23e,Mor24e,Mor25e,Mor26e,Mor27e,Mor28e,Mor29e,Mor30e,Mor31e,Mor32e,Mor01c,Mor02c,Mor03c,Mor04c,Mor05c,Mor06c,Mor07c,Mor08c,Mor09c,Mor10c,Mor11c,Mor12c,Mor13c,Mor14c,Mor15c,Mor16c,Mor17c,Mor18c,Mor19c,Mor20c,Mor21c,Mor22c,Mor23c,Mor24c,Mor25c,Mor26c,Mor27c,Mor28c,Mor29c,Mor30c,Mor31c,Mor32c")
    f.write("\n")
  else:
    f = open(output_file,"a+")
  for mol2file_name,id,smile in zip(mol2file_name_list, id_list, smiles_list):
    try:
      n+=1
      if n < start_from:
          print(n)
          continue
      total_charge,spin_multi= calculate_formalCharge_Multiplicity(mol2file_name,"mol2")
      spin_multi_dict={1:"SINGLET", 2:"DOUBLET", 3:"TRIPLET", 4:"QUARTET", 5:"QUINTET", 6:"SEXTET", 7:"SEPTET", 8:"OCTET"}
      spin_multi_name=spin_multi_dict[spin_multi]
      

      # Read mo2 file
      if os.path.exists(mol2file_name):
        for mol in pybel.readfile("mol2", mol2file_name):
            mymol = mol

        mopac_input = os.path.join(output_dir,id+".mop")

        #   if os.path.exists(mopac_input):
        #       continue
        # Making mopac input file for calculation type PM7
        calc_type=" PM7"

        key_parameter=" AUX LARGE CHARGE="+str(total_charge)+" "+spin_multi_name+calc_type+" THREADS="+str(n_threads)+" OPT"+"\n"+id
        mymol.write("mopcrt", mopac_input ,opt={"k":key_parameter},overwrite=True)

        mopac_output = os.path.join(output_dir,id+".arc")
        
        if os.name == "nt":
          mopac_executable = "mopac"
        else:
          mopac_executable = 'mopac-22.1.1-linux/bin/mopac'

        if os.path.exists(mopac_output):
          pass
        else:
          #cmd = ['mopac', mopac_input]
          if os.name == 'nt':
            rocess = subprocess.Popen([mopac_executable, mopac_input])
          else:
            process = subprocess.Popen([mopac_executable, mopac_input])
            cpu_percent = str(ncpu*100)
            cpulimit_process = subprocess.Popen(['cpulimit', '-p', str(process.pid), '-l', cpu_percent])
            process.wait()
            cpulimit_process.terminate()
      

    except Exception as e:
      print(" Error in running mopac ",end="\n")
      print(id)
      print(e)
  while checkIfProcessRunning("mopac.exe"):
     continue
    
  for mol2file_name,id,smile in zip(mol2file_name_list, id_list, smiles_list):
    # print("=",end="")
    try:
      n+=1
      if n < start_from:
          print(n)
          continue
      mopac_output = os.path.join(output_dir,id+".arc")
      if os.path.exists(mopac_output):
        # reading descriptor data from file
        desc_data=CalculateBasicQC(ReadFile(filename=mopac_output, mol_name = id,smile=smile))

        morse_file = os.path.join(output_dir,id+".csv")

        mopac_output = os.path.join(output_dir,id+".out")

        
        if os.name == "nt":
          if os.path.exists("3dmorse.exe"):
              pass
          else:
              raise ImportError("For windows please download 3dmorse.exe . Use the following link to download and provide it by keep the file in current directory. https://github.com/devinyak/3dmorse")
        # calculate morse descriptors
        if os.name == "nt":
          subprocess.run(["3dmorse.exe", mopac_output, morse_file])
        else:
          subprocess.run(['./3dmorse',mopac_output, morse_file])
        morse_desc = pd.read_csv(morse_file)
        morse_dict = morse_desc.to_dict('records')[0]
        desc_data.update(morse_dict)

        
        row_list = list(desc_data.values())
        row_list = [str(x) for x in row_list]
        row_str = ",".join(row_list)
        f.write(row_str)
        f.write("\n")
      else:
        pass
      #print("=",end="")
    except Exception as e:
      print(" Error in descriptor calculation ",end="\n")
      print(id)
      print(e)
  print()
  f.close()











