import os
import pandas as pd
import numpy as np
import argparse
from sklearn.impute import KNNImputer
import h5py

# Mordred selected columns that are non-empty (standard set)
SUBSET_COLUMNS = ["ABC","ABCGG","SlogP_VSA9","ATS7are","n8HRing","n5Ring","FNSA4","ECIndex","NsssssP","ATSC1se","SMR","AATS3p","ATS5m","SssGeH2","ATSC5i","AMID","Sm","nG12AHRing","SssssN","n9aHRing","nRing","SsNH2","nF","MATS3v","AETA_beta_ns","SIC3","ATS5i","n9ARing","nBondsS","AATS6Z","GATS6s","ATSC2Z","nBondsT","AATS7Z","PEOE_VSA10","SddssSe","GGI5","Xc-5d","SsssSiH","n8aHRing","ATS4m","MATS5se","MPC10","GATS4m","nN","NsssP","NsssAs","n6AHRing","ATSC2s","Xpc-4dv","ATSC5dv","AATSC4pe","nFaHRing","ATSC7i","MWC08","ATSC0i","SpMax_Dzi","VSA_EState4","LogEE_Dzi","AATSC0c","ATSC4pe","AXp-4dv","GATS3are","StsC","SaaO","GATS1c","Xc-4dv","ATS8are","AMID_O","JGI6","SsssCH","SZ","MATS7c","SaaSe","SpDiam_Dzp","MATS5c","n6aHRing","SpMAD_Dzse","AATSC6se","MATS1i","n9FHRing","NssSnH2","MATS5v","GGI8","MATS2i","MOMI-Z","ATS4Z","NssssSi","AATS6are","n5FAHRing","NssSe","Sare","AETA_dBeta","n11ARing","SsPbH3","Xch-6d","ATSC0m","ETA_epsilon_5","Mse","SpAD_D","AATSC1i","n8AHRing","MID_X","SpMAD_Dzp","n9FARing","ATSC3s","AATSC0s","AATSC2m","ATSC0c","AATSC2c","VE3_Dzi","AATSC6c","GATS1m","MATS3i","SdSe","SssSe","AATSC2dv","n9FaRing","MPC4","SpAbs_A","C2SP2","AATSC5v","ETA_psi_1","NddssS","AATSC0d","GATS1pe","DPSA5","AATSC2se","MWC04","AATSC5p","SssSnH2","RNCS","SpMax_Dzse","ATS6are","PEOE_VSA2","NssO","AATSC7p","BIC0","AXp-1d","SpAbs_Dzi","ATSC1d","Xch-7d","nG12Ring","SdCH2","SMR_VSA7","ATSC0Z","n7FRing","MWC07","AATS0p","ATSC7pe","n5FHRing","ATS0pe","GATS4s","Xch-4dv","ATS2s","ATS6s","GATS3s","AETA_eta_R","MATS5d","NsPbH3","EState_VSA7","AATSC7pe","AATSC0pe","Xp-7d","n6aRing","SpMAD_Dzpe","SlogP_VSA2","n11FAHRing","ATS3v","GATS3Z","GATS6m","n9FAHRing","BCUTse-1h","TopoPSA(NO)","Zagreb2","ATSC0p","EState_VSA4","AATSC6v","AATS4m","nHeavyAtom","ATS8dv","VR1_Dzi","VSA_EState7","ATS3dv","MPC9","VE1_Dzpe","n12AHRing","NsNH2","ATSC0s","ATS8d","PNSA3","SpDiam_Dzi","SpDiam_Dzare","n11FaHRing","piPC1","piPC10","AATSC5i","MATS4c","AATSC3Z","NdNH","VR1_Dzse","MATS7pe","SssNH","ATSC1p","ATS3Z","PPSA4","AATSC3s","ATS1dv","PEOE_VSA6","AATS0m","AXp-5dv","n12HRing","VE1_Dzm","VR3_Dzare","SpMAD_Dzv","HybRatio","nBase","GATS1are","AATS2pe","SsCl","AATSC7m","mZagreb1","SsNH3","SM1_DzZ","VE3_A","NtsC","SssAsH","ETA_epsilon_2","TMPC10","BIC4","AATS3pe","nBondsKD","MOMI-Y","GATS1i","n8ARing","SlogP_VSA8","mZagreb2","AATS1m","GATS5pe","ATSC3pe","MATS4are","VMcGowan","MATS2Z","TIC3","PNSA5","BCUTv-1l","AATS3se","GATS4dv","piPC9","naHRing","ATSC8se","n11AHRing","GATS6se","GATS2m","WPSA2","AXp-7d","RNCG","VE1_Dzi","VE1_Dzv","AATS6pe","AATS5se","AETA_eta","SM1_Dzv","GATS7d","MATS6se","SRW02","JGI4","SpMAD_A","MPC5","SpDiam_D","GATS5are","AATSC4dv","VSA_EState9","n5FRing","MATS5m","n6FaHRing","GATS5dv","NaaSe","ATS1Z","MATS7are","MIC2","MATS1pe","StCH","ATSC2p","nG12aRing","MATS1d","AMW","SsCH3","n10FARing","n11FaRing","LogEE_Dzpe","Xpc-6dv","JGI8","VR3_Dzp","AATSC1dv","MATS6pe","ATS2dv","ATS4dv","SaaCH","BCUTse-1l","ATSC1dv","SpAbs_Dzare","AETA_eta_BR","MATS6i","Mm","AATS7i","SpMAD_Dzi","MATS6dv","PEOE_VSA13","n6Ring","NsssSiH","AATS7are","GATS3pe","n5HRing","SsSH","BCUTc-1l","n12ARing","PEOE_VSA7","GATS5c","nBondsO","ATSC5are","AATSC3dv","ZMIC4","GRAV","MATS6s","SM1_Dzare","JGI5","nAtom","AATS4i","FPSA1","n5FaRing","ATSC5Z","nS","GATS2v","ATSC4s","Xp-7dv","FNSA5","GRAVp","NsssSnH","VR2_D","AATS3m","LogEE_Dzm","NsSnH3","nFaRing","NssssN","DPSA3","n4FRing","StN","nG12FHRing","AATS7m","MATS2d","n12FaRing","ATSC3Z","NdssC","AATS6i","ATSC5p","FPSA3","SsSeH","C2SP3","AXp-6d","ATSC1Z","AATSC6s","SIC1","AATS4Z","SRW09","WPSA4","SIC2","ATSC8v","n7aRing","ATS1m","VR1_Dzv","nFAHRing","n11aHRing","DPSA2","AATS5dv","ATSC8c","VE3_Dzse","SpMax_Dzpe","NddC","Radius","Xp-5dv","n8FAHRing","BCUTd-1h","ATS6d","Xc-6d","TpiPC10","TMWC10","NssssPb","ATS8m","TIC5","TopoPSA","SssPbH2","AATSC2Z","n6FARing","ATS5p","AATS7se","NdSe","AATS6v","GATS7pe","VE3_Dzpe","Xc-3dv","SsAsH2","NssBH","GATS1dv","ATS8Z","NssPH","n10FaRing","ATSC1c","SpDiam_Dzse","GATS5se","EState_VSA5","AATSC5pe","AATS5Z","nBr","AATSC0Z","AATS4dv","naRing","SssssPb","n8Ring","ATSC1s","MATS1s","AATSC6pe","PNSA4","ETA_beta_ns_d","BCUTZ-1h","Xp-0d","ATSC8s","VE2_Dzp","CIC0","AATS3Z","SpAD_DzZ","NdsCH","nAromAtom","ATS6p","ETA_dAlpha_B","SpDiam_DzZ","AATS0i","GATS4pe","ATS4p","ATSC0dv","SpMax_A","NsAsH2","ETA_shape_x","ATS8i","ETA_dPsi_B","ATSC3se","SlogP_VSA6","VR2_A","NssssGe","AATS0pe","AATS7d","n7Ring","NdO","MATS4se","VE3_Dzm","GATS4v","MATS1dv","TIC4","MPC6","n9Ring","ATS5d","WPath","SRW04","ATS2Z","VR2_Dzpe","SlogP_VSA7","MATS1m","n7FARing","ATS3are","n11Ring","Xch-3dv","AATS2dv","SpMAD_Dzare","GATS7v","AATS0se","SpAD_Dzse","ATSC2dv","NaaNH","SpMax_Dzv","AATS0d","ATS2se","AATSC4m","Xch-6dv","AXp-5d","AXp-0dv","ATSC6dv","ATS2v","GeomPetitjeanIndex","GATS3v","MID_h","NssS","VE2_Dzpe","ETA_eta","MWC06","SdO","GeomDiameter","SMR_VSA2","SM1_Dzp","EState_VSA2","GGI1","n8aRing","SdsssP","FNSA3","ATSC1m","AATS1v","MIC4","ATSC6pe","AXp-2d","PPSA5","Xch-7dv","AETA_eta_RL","GATS2Z","AATS2s","BCUTi-1h","AATSC5are","MATS5s","ATS0m","ATSC3m","Xp-4d","MZ","ATS3p","VR3_Dzpe","SddC","MATS1se","n10FRing","SpMAD_D","MATS4s","n12FRing","VE2_A","n4HRing","ATSC6m","Xc-4d","GATS2are","WNSA5","AATSC4are","NsI","ATS6i","TIC2","n6FaRing","SpDiam_Dzpe","BertzCT","ATS2i","ATSC2m","PPSA3","NssPbH2","LogEE_A","AATS5v","nBondsD","DPSA4","SsGeH3","SMR_VSA6","NaaaC","GATS1p","ATS0dv","NaaCH","JGT10","SddssS","LogEE_Dzv","ATSC3dv","SssS","SpMax_Dzm","SpAbs_DzZ","ATSC1v","ATS0d","VR2_Dzi","MATS3m","ATSC0d","ETA_shape_y","NaasN","ATS0i","ATSC4se","AETA_eta_B","ATS1i","SdNH","AATSC1se","GATS7m","AATSC5se","n7FAHRing","AATS4are","n6HRing","VSA_EState8","GATS3d","AATSC6m","ATSC7v","SdsN","VR2_DzZ","WNSA3","AATSC2s","AATSC1d","ETA_eta_R","VR3_A","ETA_epsilon_1","NsOH","Xch-4d","VAdjMat","ATSC6are","nH","nX","SpDiam_Dzm","VE2_Dzare","VR1_D","n8FaHRing","VR1_DzZ","Xch-5dv","AATS3i","SdssSe","VR3_Dzv","SddsN","n4FAHRing","NsNH3","C1SP2","SMR_VSA3","Kier2","MATS7v","SsssGeH","n7aHRing","FNSA1","NaaN","ZMIC0","BCUTare-1h","piPC3","AATSC7d","VR1_A","SRW03","MATS1v","MID_N","SM1_Dzm","NssssSn","NssAsH","GATS3dv","LogEE_Dzp","MATS5Z","GATS6are","AATS3d","ATSC0se","SpAD_Dzpe","VR1_Dzare","n11FARing","Xp-1d","MATS6v","GGI2","ATS3se","VR3_D","ATS7pe","AATS2i","MWC01","n8FARing","FNSA2","ATSC5s","GATS2i","SpAD_Dzm","AXp-0d","RotRatio","MATS7s","SssCH2","TSRW10","Mi","WPSA3","MATS3dv","C1SP3","ATSC2se","VR1_Dzpe","BIC1","GeomRadius","AATS2se","Sp","ATS6Z","MATS2are","VR3_Dzm","MATS5i","MATS1Z","n4FaHRing","WNSA1","TIC0","MATS5are","SsssdAs","AATS0are","AATSC5c","MIC1","AATSC1are","ATSC4are","BCUTv-1h","ATSC4v","TIC1","nHRing","TASA","MATS3p","SM1_Dzse","MATS1are","ATS1v","MATS6d","ATS0se","Zagreb1","BIC2","ATSC7s","AATSC1v","ATS2m","AXp-3dv","NsssN","SLogP","VE1_D","n9AHRing","ATSC2d","nSpiro","BCUTc-1h","CIC1","EState_VSA10","AATS2v","GATS7i","ATS6m","MATS4i","ETA_epsilon_3","Xp-4dv","AATS2Z","AATSC4v","SssBH","SdssS","n7HRing","AATSC6are","SRW10","nB","n5aHRing","GGI10","n10FAHRing","ATS2p","SsF","AATS4v","ATSC4i","SpMax_DzZ","AATSC3v","ATS8se","SsBr","AATS0s","MATS6Z","NssGeH2","ATS7i","ETA_eta_L","LogEE_Dzare","SpMAD_Dzm","n12aRing","ATS7Z","nBondsKS","AATSC3c","EState_VSA3","VE1_A","n10ARing","BIC3","n4FHRing","ATS8s","ATS7dv","AATS6d","n7ARing","ATS2are","NtCH","AATSC4se","VR3_Dzi","MWC02","AETA_eta_FL","AATS3v","MATS2dv","AMID_N","ETA_dEpsilon_D","AATS2are","AATSC0v","n3aRing","SsSiH3","GATS1d","ATSC4p","GATS1v","SssssB","MATS7se","ATS0Z","ETA_dBeta","FPSA2","MATS2se","Kier1","Xpc-5dv","SpAbs_Dzm","MATS4pe","SsssNH","ATS0v","ATSC5d","ATS6pe","AATS6m","ETA_eta_FL","GATS2se","SpMax_Dzp","NsssdAs","GATS6Z","n4AHRing","Xpc-4d","AATS7p","MIC5","C2SP1","AATSC4Z","AATSC0dv","ZMIC3","Xc-6dv","nBonds","SIC4","nG12HRing","NdCH2","SsssB","AATSC3i","AMID_C","ATSC3p","n8FRing","GATS2c","AMID_h","AATS2p","VR3_Dzse","GGI7","SMR_VSA9","AATSC3se","nFARing","VR2_Dzse","PEOE_VSA4","NssSiH2","n3ARing","AATS2m","NssNH","GATS3c","nG12ARing","Mpe","ATSC1are","Mare","MATS6p","VE3_D","n10FHRing","ETA_eta_F","SsssssP","GATS6v","SssssSi","LabuteASA","BIC5","MATS2pe","AATSC5dv","MWC03","ZMIC2","PetitjeanIndex","GATS5v","NdssS","MATS7Z","nBridgehead","BCUTm-1l","ATS4s","AATS5are","nG12FaHRing","ATSC3are","n9FaHRing","BCUTi-1l","nAromBond","AATSC2i","AATSC5Z","PNSA2","NsSiH3","n11HRing","n11aRing","JGI2","DPSA1","SssNH2","GATS5Z","ETA_beta_ns","n11FHRing","GATS1se","NdsN","ATS7v","SssPH","ATSC6se","SsssSnH","n6FRing","NsBr","LogEE_DzZ","MATS3s","SRW06","MATS3d","MATS5dv","n12Ring","AATS1i","ATSC6Z","nCl","ATSC6p","PEOE_VSA1","WNSA4","ATSC7m","ATSC8m","GATS6i","SaaNH","AATS1p","RPSA","n3HRing","AATSC7c","VE1_Dzse","LogEE_Dzse","IC5","ATSC2c","ATSC6i","SsssAs","AATSC2v","AATS4d","VE2_Dzv","MATS3c","VSA_EState3","n6FAHRing","NsssNH","AATS6p","EState_VSA8","ETA_eta_BR","piPC2","ATSC3c","ATS3d","ATSC5c","ATSC8i","n11FRing","GATS6c","AATSC0m","GATS1Z","ATS5dv","n9FRing","ATSC7p","GATS5i","ATS2pe","Xc-5dv","SlogP_VSA3","AATSC5s","n10AHRing","BCUTp-1l","MATS7dv","AATSC3are","GATS4Z","MIC0","MIC3","ATSC4Z","n12aHRing","GATS2p","nBondsA","GATS7p","n7AHRing","SaasC","Xp-3dv","SlogP_VSA5","AXp-6dv","SRW07","n12FaHRing","JGI1","Xc-3d","ATSC6d","AATSC1p","nI","Xch-5d","SsOH","NsLi","ATS3pe","AATSC1c","SdsCH","GATS4are","AATS5p","Xp-5d","GATS4d","GATS7Z","nARing","MATS4p","ETA_shape_p","AATS5s","Si","GATS2d","AATSC4p","AATS6s","AATSC3p","n4FaRing","ATS7d","SRW05","GATS7are","NtN","BCUTpe-1l","VE3_Dzare","ATSC2are","MATS2c","Xpc-5d","Xp-1dv","AATS0dv","n12FARing","NdsssP","WPSA5","Mv","MPC2","MDEC-22","n4ARing","ATS5Z","VE2_DzZ","AATSC3pe","MWC09","FPSA5","GATS5p","n9aRing","NdS","AATS1pe","NssssBe","AATS5i","ATSC2i","VSA_EState5","n5aRing","AATS4pe","AATS1Z","piPC7","SpMax_D","PEOE_VSA12","GATS6d","MW","ATS4pe","ATS0are","PPSA1","AXp-2dv","Xp-6dv","ATSC4m","MATS7m","GGI3","AATS3dv","ATS4se","ATS5pe","WPSA1","AATS3are","ATSC1i","GGI6","SpAD_Dzp","MPC3","EState_VSA6","AATS6dv","n3AHRing","SpMAD_DzZ","ATS5se","AATS0Z","GATS4p","nHBAcc","ATS1s","AATS7v","AATSC7are","GATS2dv","AATSC7se","piPC8","SaasN","VE2_Dzi","MATS6are","AXp-4d","ATSC6v","SlogP_VSA11","MATS6m","VE3_Dzv","AATSC4i","ATS1are","BalabanJ","VE1_DzZ","AATS4s","n8FHRing","NssNH2","Mp","AATS7s","ETA_eta_RL","MOMI-X","VSA_EState6","MATS6c","Xp-6d","NaaO","RPCG","BCUTpe-1h","GATS2pe","IC0","ATSC2pe","AATSC7i","AXp-1dv","MPC7","n4FARing","PEOE_VSA11","SpAbs_Dzpe","VR2_Dzv","BCUTare-1l","ETA_beta_s","AXp-7dv","AATS3s","piPC6","CIC5","Xch-3d","BCUTd-1l","AATSC4d","ATSC7c","VE2_Dzse","ATSC7Z","AATS5pe","MATS7p","ATS6se","ATSC0pe","ATSC1pe","AATS5d","nBondsM","ATS4v","GATS6dv","VE2_Dzm","n12FHRing","NsCl","SssSiH2","MATS4d","AATSC6Z","NsssGeH","MATS2s","apol","NsssssAs","AATSC0se","n5ARing","GATS2s","ATS0s","SdS","NsSH","n12FAHRing","NssssB","SpAbs_Dzp","MWC10","n10FaHRing","SM1_Dzi","n7FaHRing","AETA_eta_F","GATS4i","AETA_alpha","SlogP_VSA4","nAHRing","PNSA1","ATSC8pe","SssssC","FPSA4","IC1","CIC3","NaasC","TPSA","SpMax_Dzare","nG12FAHRing","AATSC5d","WPol","AATSC0p","LogEE_D","MPC8","SpAD_Dzv","AETA_beta_s","n5FaHRing","ETA_beta","ATSC0are","MATS3Z","SaaN","C3SP3","AATS0v","n10aHRing","ATS3m","NsSeH","MID_O","JGI3","GATS5s","GATS3m","AATSC3m","CIC4","SdssC","MATS7i","ATS3i","TopoShapeIndex","GATS7dv","GGI4","ATSC8are","SpAD_Dzi","bpol","VE3_Dzp","MATS4m","MDEC-33","n3Ring","MATS3se","AATSC0i","VSA_EState1","ETA_dAlpha_A","SpDiam_A","MATS4Z","PEOE_VSA8","VR2_Dzare","NdssSe","AATS2d","SIC0","piPC5","Sse","AETA_eta_L","ATSC8p","ATS7p","ATS1pe","ATSC7dv","ATSC0v","SsssN","MATS4dv","GATS3se","SsI","n10Ring","MID","SpAD_A","n5AHRing","GATS3p","ATS5s","nG12aHRing","piPC4","SMR_VSA1","ATSC3i","n7FHRing","SpAbs_D","n4aRing","MWC05","ATS6dv","SpAbs_Dzv","BCUTdv-1l","nO","AATS4p","ATSC8Z","ATSC3d","AETA_beta_ns_d","MID_C","SM1_Dzpe","ATSC6c","GATS7s","AATS4se","GATS6p","ATSC8d","AATSC6p","MATS3are","BCUTdv-1h","AATSC7v","PEOE_VSA5","AATSC2pe","NsssCH","JGI10","NsssB","C4SP3","NsssPbH","NssBe","SsLi","ETA_dEpsilon_A","Xp-3d","VSA_EState2","NsCH3","AATSC1Z","SsPH2","GATS3i","ATS6v","VR1_Dzp","SRW08","PEOE_VSA3","ETA_dPsi_A","Xp-2dv","nG12FRing","AATSC5m","Xp-0dv","ATS8v","AATS1d","VR1_Dzm","SssssGe","BCUTm-1h","AATS7dv","SpAD_Dzare","ATS8pe","SssO","AATSC6i","SIC5","SlogP_VSA1","PEOE_VSA9","n4Ring","ATSC4d","AATS1dv","MATS5pe","ETA_dEpsilon_C","VR2_Dzm","nP","NddssSe","n10HRing","AATS1se","SssssBe","BCUTZ-1l","GATS5d","Spe","AATSC1pe","ATS4i","nHBDon","PPSA2","AXp-3d","ATSC6s","ATSC8dv","AATS6se","SpAbs_Dzse","nFRing","n5FARing","nG12FARing","n8FaRing","SMR_VSA5","VE1_Dzp","MATS1c","NsF","MATS7d","fMF","GATS1s","ATSC4dv","Kier3","MATS4v","ATS1d","AATSC3d","ATSC7se","ATS7se","GATS4se","SssBe","AATSC2d","AATS1are","MATS2m","BCUTp-1h","MATS3pe","AETA_beta","nC","MATS2v","AATSC7dv","IC3","IC4","GATS5m","WNSA2","GATS7se","MATS2p","ATSC5pe","fragCpx","AATSC7s","AATSC6d","EState_VSA9","VE2_D","ZMIC5","BCUTs-1h","ATSC5m","SsssssAs","C1SP1","GATS7c","AATSC4c","ATSC7are","n4aHRing","n6FHRing","AATSC1m","ATS4d","BCUTs-1l","IC2","ETA_dEpsilon_B","NsPH2","AATSC0are","ATS2d","C3SP2","GGI9","Xp-2d","nAcid","ATS5v","AATSC2p","SsssP","RASA","SpDiam_Dzv","ATS7s","ATSC5se","Xpc-6d","ATSC5v","MATS5p","AATSC7Z","AATS1s","EState_VSA1","ATS1p","nG12FaRing","ATS4are","n6ARing","n3aHRing","SlogP_VSA10","GATS4c","VE3_DzZ","ATS7m","SsssPbH","ZMIC1","ATSC2v","AATS7pe","AATS5m","ATS8p","SaaaC","nRot","Diameter","GeomShapeIndex","MDEC-23","n10aRing","NssCH2","ATSC7d","VR3_DzZ","ATS0p","ATS3s","ETA_alpha","SMR_VSA4","GATS6pe","SsSnH3","Sv","nFHRing","AATSC2are","RPCS","AATSC4s","ETA_epsilon_4","JGI7","SssssSn","NsGeH3","AATSC6dv","ATSC4c","SMR_VSA8","NssssC","MATS1p","NddsN","VR2_Dzp","ATS1se","AATSC1s","CIC2","n7FaRing","SaaS","NaaS","VE1_Dzare","n9HRing","JGI9","AMID_X","ATSC3v","ETA_eta_B"]

def get_common_ids(csv_files, chunksize=10000):
    common_ids = None
    for csv_path in csv_files:
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping from intersection.")
            continue
        ids_in_file = set()
        sep = '\t' if 'imagemol' in os.path.basename(csv_path).lower() else ','
        for chunk in pd.read_csv(csv_path, usecols=["id"], chunksize=chunksize, sep=sep):
            ids_in_file.update(chunk["id"].dropna().values)
        if common_ids is None:
            common_ids = ids_in_file
        else:
            common_ids &= ids_in_file
        print(f"{os.path.basename(csv_path)} -> {len(common_ids)} common ids so far")
    return common_ids

def csv_to_hdf(csv_path, h5_path, common_ids, chunksize=10000):
    if not os.path.exists(csv_path):
        return
    with h5py.File(h5_path, 'w') as h5_file:
        first = True
        sep = '\t' if 'imagemol' in os.path.basename(csv_path).lower() else ','
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, sep=sep):
            chunk = chunk[chunk["id"].isin(common_ids)]
            if chunk.empty: continue
            chunk = chunk.drop(columns=[c for c in ["id", "SMILES"] if c in chunk.columns])
            if first:
                h5_file.create_dataset("data", data=chunk.values, maxshape=(None, chunk.shape[1]), chunks=True)
                first = False
            else:
                h5_file["data"].resize(h5_file["data"].shape[0] + chunk.shape[0], axis=0)
                h5_file["data"][-chunk.shape[0]:] = chunk.values

def process_mordred(csv_path, h5_path, common_ids, knn_neighbors=5):
    if not os.path.exists(csv_path):
        return
    imputer = KNNImputer(n_neighbors=knn_neighbors)
    with h5py.File(h5_path, 'w') as h5_file:
        first = True
        for chunk in pd.read_csv(csv_path, chunksize=10000):
            chunk = chunk[chunk["id"].isin(common_ids)]
            if chunk.empty: continue
            chunk = chunk[SUBSET_COLUMNS].apply(pd.to_numeric, errors="coerce")
            imputed = imputer.fit_transform(chunk)
            imputed_df = pd.DataFrame(imputed, columns=SUBSET_COLUMNS)
            if first:
                h5_file.create_dataset("data", data=imputed_df.values, maxshape=(None, imputed_df.shape[1]), chunks=True)
                first = False
            else:
                h5_file["data"].resize(h5_file["data"].shape[0] + imputed_df.shape[0], axis=0)
                h5_file["data"][-imputed_df.shape[0]:] = imputed_df.values

def convert_to_hdf5(input_dir="Chemicaldice_data", output_dir="Chemicaldice_data", chunk_size=10000, knn_neighbors=5):
    """
    Convert descriptor CSVs to HDF5 structures.
    
    Args:
        input_dir (str): Directory containing CSV files.
        output_dir (str): Directory to save H5 files.
        chunk_size (int): Size of chunks for processing.
        knn_neighbors (int): Neighbors for Mordred KNN imputation.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [os.path.join(input_dir, f) for f in 
                 ["mopac.csv", "Grover.csv", "ImageMol.csv", "Chemberta.csv", "Signaturizer.csv", "mordred.csv"]]
    
    print("Computing common ids across all modalities...")
    common_ids = get_common_ids(csv_files, chunksize=chunk_size)
    if not common_ids:
        print("Error: No common IDs found across datasets.")
        return
    print(f"Final common ids: {len(common_ids)}")

    mappings = {
        "mopac.csv": "mopac.h5",
        "Grover.csv": "Grover.h5",
        "ImageMol.csv": "ImageMol.h5",
        "Chemberta.csv": "Chemberta.h5",
        "Signaturizer.csv": "Signaturizer.h5",
    }

    for csv_name, h5_name in mappings.items():
        csv_path = os.path.join(input_dir, csv_name)
        h5_path = os.path.join(output_dir, h5_name)
        if os.path.exists(csv_path):
            print(f"Converting {csv_name} -> {h5_name}")
            csv_to_hdf(csv_path, h5_path, common_ids, chunksize=chunk_size)

    mordred_csv = os.path.join(input_dir, "mordred.csv")
    if os.path.exists(mordred_csv):
        print("Processing mordred.csv with KNN imputation...")
        process_mordred(mordred_csv, os.path.join(output_dir, "mordred.h5"), common_ids, knn_neighbors=knn_neighbors)

def main():
    parser = argparse.ArgumentParser(description="Convert descriptor CSVs to HDF5 structures.")
    parser.add_argument("--input_dir", default="Chemicaldice_data", help="Directory containing CSV files.")
    parser.add_argument("--output_dir", default="Chemicaldice_data", help="Directory to save H5 files.")
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--knn", type=int, default=5, help="KNN neighbors for Mordred imputation.")
    
    args = parser.parse_args()
    convert_to_hdf5(args.input_dir, args.output_dir, args.chunk_size, args.knn)

if __name__ == "__main__":
    main()
