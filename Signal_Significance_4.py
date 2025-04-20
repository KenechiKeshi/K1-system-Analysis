import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
from torch import nn
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler

# Load the Monte Carlo
root_file_MC = '/storage/epp2/phsnab/BuKPiPiMuMu/Bu2KpipiMM-MC-12115020-2018-MagUp-Sim09h-StrippingBu2LLK.root'
events_MC  = uproot.open( root_file_MC + ':Bu2LLK_mmLine_K1+/DecayTree')

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

# Construct the arrays from the data
parray_MC = events_MC.arrays(variables + ['B_BKGCAT', 'B_ENDVERTEX_CHI2', 'B_M', 'K1_1270_M', 'JPsi_M', 'muplus_isMuon', 'muminus_isMuon'], library = 'pd').query('B_PT > 2000 & B_ENDVERTEX_CHI2 < 25 & K1_1270_M < 2400 & muplus_isMuon == True & muminus_isMuon == True & K_ProbNNk > 0.1 & piplus_ProbNNpi > 0.1 & piminus_ProbNNpi > 0.1')
print(f'The length of the Monte Carlo array is {len(parray_MC)}')
parray_MC = parray_MC.loc[((parray_MC['JPsi_M']**2 < 8000000) | ((11000000 <= parray_MC['JPsi_M']**2) & (parray_MC['JPsi_M']**2 <= 12500000)) | (parray_MC['JPsi_M']**2 > 15000000))]
print(f'The length of the Monte Carlo array after dimuon filtering is {len(parray_MC)}')
parray_0 = events_exp.arrays(variables + ['JPsi_M', 'B_M'], library='pd')
parray_1 = events_1.arrays(variables + ['JPsi_M', 'B_M'], library='pd')
parray_2 = events_2.arrays(variables + ['JPsi_M', 'B_M'], library='pd')
parray_3 = events_3.arrays(variables + ['JPsi_M', 'B_M'], library='pd')
parray_4 = events_4.arrays(variables + ['JPsi_M', 'B_M'], library='pd')
parray_5 = events_5.arrays(variables + ['JPsi_M', 'B_M'], library='pd')
parray_exp = pd.concat([parray_0, parray_1, parray_2, parray_3, parray_4, parray_5])
print(f'The length of the experimental array is {len(parray_exp)}')
parray_exp = parray_exp.loc[((parray_exp['JPsi_M']**2 < 8000000) | ((11000000 <= parray_exp['JPsi_M']**2) & (parray_exp['JPsi_M']**2 <= 12500000)) | (parray_exp['JPsi_M']**2 > 15000000))]
print(f'The length of the experimental array after dimuon filtering is {len(parray_exp)}')

# Construct feature tensor and scale it
with open('Scaler_5.pkl', 'rb') as f:
    scaler =  pickle.load(f)
X_tensor = torch.tensor(scaler.transform(parray_MC.drop(['B_BKGCAT', 'B_ENDVERTEX_CHI2', 'B_M', 'K1_1270_M', 'muplus_isMuon', 'muminus_isMuon', 'JPsi_M'], axis=1)))
parray_exp_tensor = torch.tensor(scaler.transform(parray_exp.drop(['JPsi_M', 'B_M'], axis=1)))

# Construct new columns related to the Neural Network
parray_MC['event_ID_pred_proba'] = deep_nn(X_tensor).detach().numpy()
parray_exp['event_ID_pred_proba'] = deep_nn(parray_exp_tensor).detach().numpy()

# Signal Significance
thresholds = np.linspace(0.0, 1.0, 1000)
scores = []
signals = []
sigmas = []

# Set a window around the known B mass
queried_parray_MC = parray_MC.query('5340 > B_M > 5220')
queried_parray_exp = parray_exp.query('5340 > B_M > 5220')
print((min(queried_parray_exp['event_ID_pred_proba']), max(queried_parray_exp['event_ID_pred_proba'])))

# Equalise the lengths of the experimental and Monte Carlo arrays
k = min(len(queried_parray_MC), len(queried_parray_exp))
queried_parray_MC = queried_parray_MC.sample(n=k)
queried_parray_exp = queried_parray_exp.sample(n=k)

for threshold in thresholds:
    filtered_parray_MC = queried_parray_MC.loc[(queried_parray_MC['event_ID_pred_proba'] > threshold)]
    filtered_parray_exp = queried_parray_exp.loc[queried_parray_exp['event_ID_pred_proba'] > threshold]
    signal_count = ((filtered_parray_MC['B_BKGCAT'] < 11) | (filtered_parray_MC['B_BKGCAT'] == 50)).astype(int).sum()
    sigma = np.sqrt(len(filtered_parray_exp))
    score = signal_count / sigma if sigma != 0 else 0
    scores.append(score)
    signals.append(signal_count)
    sigmas.append(sigma)

optimal_threshold = thresholds[scores.index(max(scores))]

# Plot the results
plt.plot(thresholds, scores)
plt.title('Signal Significance at Various Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Significance')
plt.text(min(thresholds), max(scores), f'Maximum at {optimal_threshold: .3f}')
plt.show()

print(f'The optimal threshold is {optimal_threshold}')
'''
plt.hist(parray_exp['event_ID_pred_proba'], alpha=0.9, bins=100, density=True, label='Real Data', color='lime')
plt.hist(parray_MC.loc[parray_MC['B_BKGCAT'] < 60, ['event_ID_pred_proba']], alpha=0.9, bins=100, label='Monte Carlo - S', density=True, color='black')
plt.hist(parray_MC.loc[parray_MC['B_BKGCAT'] >= 60, ['event_ID_pred_proba']], alpha=0.6, bins=100, label='Monte Carlo - B', density=True, color='fuchsia')
plt.yscale('log')
plt.legend()
plt.show()
plt.plot(thresholds, sigmas)
plt.show()
'''
    





