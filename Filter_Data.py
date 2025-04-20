import uproot
import ROOT
import numpy as np
import pandas as pd
import pickle
import torch
from torch import nn
from array import array
from sklearn.preprocessing import StandardScaler

# Load the real data
root_file_exp = '/storage/epp2/phsnab/BuKPiPiMuMu/Bu2KpipiMM-Data-2018-MagUp-Combined-2324.root'
events_exp = uproot.open( root_file_exp + ':DecayTree' )

root_file1 = '/storage/epp2/phsnab/BuKPiPiMuMu/Bu2KpipiMM-Data-2016-MagUp-Combined-2324.root'
events_1 = uproot.open( root_file1 + ':DecayTree' )
 
root_file2 = '/storage/epp2/phsnab/BuKPiPiMuMu/Bu2KpipiMM-Data-2016-MagDn-Combined-2324.root'
events_2 = uproot.open( root_file2 + ':DecayTree' )
 
root_file3 = '/storage/epp2/phsnab/BuKPiPiMuMu/Bu2KpipiMM-Data-2017-MagUp-Combined-2324.root'
events_3 = uproot.open( root_file3 + ':DecayTree' )
 
root_file4 = '/storage/epp2/phsnab/BuKPiPiMuMu/Bu2KpipiMM-Data-2017-MagDn-Combined-2324.root'
events_4 = uproot.open( root_file4 + ':DecayTree' )

root_file5 = '/storage/epp2/phsnab/BuKPiPiMuMu/Bu2KpipiMM-Data-2018-MagDn-Combined-2324.root'
events_5 = uproot.open( root_file5 + ':DecayTree' )

# Define the Neural Network
class EventPrediction(nn.Module):
        # Construct a three layer neural network 
        def __init__(self, in_features, out_features):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=in_features, out_features=out_features).to(torch.float64)
            self.layer_2 = nn.Linear(in_features=out_features, out_features=out_features).to(torch.float64)
            self.layer_3 = nn.Linear(in_features=out_features, out_features=out_features).to(torch.float64)
            # Activation functions 
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            self.act3 = nn.ReLU()
            # Output function
            self.output = nn.Linear(in_features=out_features, out_features=1).to(torch.float64)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.act1(self.layer_1(x))
            x = self.act2(self.layer_2(x))
            x = self.act3(self.layer_3(x))
            return self.sigmoid(self.output(x))

# Load the Neural Network weights
with open('Deep_NN_5.pkl', 'rb') as f:
    deep_nn =  pickle.load(f)

# Define the variables used by the Neural Network
variables = ['B_PT', 'B_FDCHI2_OWNPV', 'B_DiraAngle',
             'K_PT', 'K_IPCHI2_OWNPV', 'K_ProbNNk',
             'JPsi_FDCHI2_OWNPV', 'JPsi_ENDVERTEX_CHI2',
             'piplus_ProbNNpi', 'piminus_ProbNNpi']

# Define the variables used after the Neural Network
secondary_variables = ['B_M', 'JPsi_M', 'piplus_isMuon', 'piminus_isMuon']
particles = ['piplus', 'piminus', 'muplus', 'muminus', 'K', 'JPsi']
P = ['_PE', '_PX', '_PY', '_PZ']
secondary_variables += [ particle + p for particle in particles for p in P]

# Construct the data array
parray_0 = events_exp.arrays(variables + secondary_variables, library='pd')
parray_1 = events_1.arrays(variables + secondary_variables, library='pd')
parray_2 = events_2.arrays(variables + secondary_variables, library='pd')
parray_3 = events_3.arrays(variables + secondary_variables, library='pd')
parray_4 = events_4.arrays(variables + secondary_variables, library='pd')
parray_5 = events_5.arrays(variables + secondary_variables, library='pd')
parray_exp = pd.concat([parray_0, parray_1, parray_2, parray_3, parray_4, parray_5])
# Construct feature tensor and scale it
with open('Scaler_5.pkl', 'rb') as f:
    scaler =  pickle.load(f)
parray_exp_tensor = torch.tensor(scaler.transform(parray_exp.drop(secondary_variables, axis=1)))

# Construct new columns related to the Neural Network
parray_exp['event_ID_pred_proba'] = deep_nn(parray_exp_tensor).detach().numpy()

# Filter based on threshold
threshold = 0.8338938938938939

filtered_parray = parray_exp.query(f'{threshold} < event_ID_pred_proba')
print(f'The length of the original experimental data is {len(parray_exp)}')
print(f'The length of the filtered experimental data is {len(filtered_parray)}')

# Create new ROOT file
f = ROOT.TFile('MVA_Combined.root', 'RECREATE')
tree = ROOT.TTree('DecayTree', 'MVA_Filtered_Tree')

# Define the branches
var_dictionary = {var: array('f', [0.]) for var in (variables + secondary_variables)}
var_dictionary['S_Probability'] = array('f', [0.])
for var, item in var_dictionary.items():
        tree.Branch(var, item, f'{var}/F')

# Fill the Tree
for _, row in filtered_parray.iterrows():
    for var in (variables + secondary_variables):
        var_dictionary[var][0] = row[var]
    var_dictionary['S_Probability'][0] = row['event_ID_pred_proba']
    tree.Fill()

# Finalise
tree.Write()
f.Close()
