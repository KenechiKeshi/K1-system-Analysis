import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import copy
import shap
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

root_file = '/storage/epp2/phsnab/BuKPiPiMuMu/Bu2KpipiMM-Data-2018-MagUp-Combined-2324.root'
events = uproot.open( root_file + ':DecayTree' )
root_file2 = '/storage/epp2/phsnab/BuKPiPiMuMu/Bu2KpipiMM-MC-12115020-2018-MagUp-Sim09h-StrippingBu2LLK.root'
events_sim = uproot.open( root_file2 + ':Bu2LLK_mmLine_K1+/DecayTree')

# Define variables
variables = ['B_IPCHI2_OWNPV', 'B_PT', 'B_ENDVERTEX_CHI2', 'B_DiraAngle', 'JPsi_ENDVERTEX_CHI2', 'K1_1270_ENDVERTEX_CHI2', 'K1_1270_PT', 'piplus_PT', 'piminus_PT', 'muplus_PT', 'muminus_PT']

# Construct the arrays from the data
parray_sim = events_sim.arrays(variables + ['B_BKGCAT', 'piminus_ProbNNpi', 'piplus_ProbNNpi', 'K_ProbNNk', 'K1_1270_M', 'B_M', 'muplus_isMuon', 'muminus_isMuon'], library = 'pd').query('B_PT >= 2000 & B_DiraAngle < 0.02 &  K_ProbNNk > 0.1 & piplus_ProbNNpi > 0.1 & piminus_ProbNNpi > 0.1 & B_ENDVERTEX_CHI2 <= 25 & K1_1270_M <= 2400 & muplus_isMuon == True & muminus_isMuon == True')
parray_real = events.arrays(variables + ['K1_1270_M', 'B_M'], library = 'pd').query('B_PT >= 2000 & B_ENDVERTEX_CHI2 <= 25 & K1_1270_M <= 2400 & B_M > 5800')

# Define signal and background within the simulation
def assign_map(row):
    if row['B_BKGCAT'] < 11 or row['B_BKGCAT'] == 50:
        return int(1)
    else:
        return int(0)
parray_sim['event_ID'] = parray_sim.apply(assign_map, axis=1)

# Display the ratio for the different signal categories (0, 10, 50)
print(r'The ratio for the different signal categories:')
print(parray_sim.loc[parray_sim['event_ID'] == 1, 'B_BKGCAT'].value_counts(normalize=True))

# Drop selection criteria - No longer needed
parray_sim.drop(['B_BKGCAT', 'muminus_isMuon', 'muplus_isMuon', 'piminus_ProbNNpi', 'piplus_ProbNNpi', 'K_ProbNNk'], axis=1, inplace=True)

# Define background map
def background_map(row):
    return int(0)
parray_real['event_ID'] = parray_real.apply(background_map, axis=1)

# Create a combined dataframe of simulated and experimental data
parray = pd.concat([parray_sim.loc[parray_sim['event_ID'] == 1, :], parray_real])
parray.drop(['B_M', 'B_ENDVERTEX_CHI2', 'K1_1270_M'], axis=1, inplace=True)

# Construct correlation matrix
correlation = parray.drop(['event_ID'], axis=1).corr()
sns.set(font_scale=0.8)
sns.heatmap(correlation, annot=True, fmt='.1f')
plt.show()

# Define the Neural Netowrk
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

def optimiser(n_estimators, max_depth, learning_rate, n_iter, test_size, colsample_bytree=0.5, multicollinearity=True):
    # Remove mass terms
    mass_terms = [var for var in parray.columns if '_M' in var]
    parray.drop(mass_terms, axis=1, inplace=True)
    
    if multicollinearity == True:
        # Address multicollinearity
        threshold = 0.75 # If the correlation between two non-target features is above this threshold, they need to be addressed
        removed_features = []
        for feature in parray.drop(['event_ID'], axis=1):
            if feature not in removed_features:
                for feature2 in parray.drop(columns=['event_ID']+removed_features):
                    if feature != feature2 and abs(correlation[feature][feature2]) > threshold:
                        removed_features.append(feature2)
        parray.drop(removed_features, axis=1, inplace=True)
        print(f'\nThe features removed to address multicollinearity are: {removed_features}')
      
    # Train-test splitting
    X = parray.drop(['event_ID'], axis=1)
    Y = parray['event_ID']
    weights = Y.value_counts(normalize=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=test_size)
    '''
    print('\nTraining XGBoost...')

    # Model training
    alg = XGBClassifier(n_jobs=-1, n_estimators=n_estimators,  max_depth=max_depth, colsample_bytree=colsample_bytree, learning_rate=learning_rate, random_state=42)
    alg.fit(X_train, Y_train)

    # Model evaluation
    # Compare test split with training split
    test_prediction = alg.predict(X_test)
    probabilities_test = alg.predict_proba(X_test)[:, 1]
    probabilities_test_0 = np.array([k for k in probabilities_test if k < 0.5])
    probabilities_test_1 = np.array([k for k in probabilities_test if k >= 0.5])
    
    scoring = { #Performance metrics
        'Precision': precision_score,
        'Recall': recall_score,
        'F1': f1_score,
        'Accuracy': accuracy_score,
        'ROC AUC': roc_auc_score,
    }
    
    # Evaluation on the test split
    evaluation_test = {}
    for score in scoring:
        evaluation_test[score] = scoring[score](Y_test, test_prediction)
    print(f'The evaluation scores for the prediction on the test split are: {evaluation_test}')

    train_prediction = alg.predict(X_train)
    probabilities_train = alg.predict_proba(X_train)[:, 1]
    probabilities_train_0 = np.array([k for k in probabilities_train if k < 0.5])
    probabilities_train_1 = np.array([k for k in probabilities_train if k >= 0.5])

    # Evaluation on the training split
    evaluation_train = {}
    for score in scoring:
        evaluation_train[score] = scoring[score](Y_train, train_prediction)
    print(f'\nWhereas the evaluation scores for the prediction on the training split are: {evaluation_train}' + '\n')

    # Prediction probability histograms
    plt.hist(probabilities_train_0, 100, alpha=0.3, label='Training Data', color='r', density=True)
    plt.hist(probabilities_test_0, 100, alpha=0.3, label='Test Data', color='b', density=True)
    plt.title('Distribution for probabilities of events with a negative class prediction')
    plt.legend()
    plt.show()

    plt.hist(probabilities_train_1, 100, alpha=0.3, label='Training Data', color='r', density=True)
    plt.hist(probabilities_test_1, 100, alpha=0.3, label='Test Data', color='b', density=True)
    plt.title('Distribution for probabilities of events with a positive class prediction')
    plt.legend()
    plt.show()

    # Which features provided the most predictive power
    importance = alg.feature_importances_
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)

    # Shapley Information Gain
    explainer = shap.TreeExplainer(alg)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

    plt.bar(x=importance_df['Feature'], height=importance_df['Importance'])
    plt.title('Feature Importance for XGBoost')
    plt.xticks(rotation=75, ha='right', fontsize=6)
    plt.show()
    
    # Using Neural Networks
    '''
    # Feature Scaling for Neural Networks
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    '''
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensors
    X_train = torch.tensor(X_train_scaled)
    X_test = torch.tensor(X_test_scaled)
    Y_train = torch.tensor(Y_train.values, dtype=torch.float64).reshape(-1,1)
    Y_test = torch.tensor(Y_test.values, dtype=torch.float64).reshape(-1,1)

    # Constructing the model
    in_features = len(parray.columns) - 1 # The number of predictive variables
    out_features = in_features + 5 # The number of neurons
                 
    nn_model = EventPrediction(in_features, out_features)

    # Define the optimisation algorithm
    nn_optimiser = torch.optim.Adam(nn_model.parameters(), lr=learning_rate) if learning_rate is not None else torch.optim.Adam(nn_model.parameters(), lr=0.1)

    # Define a suitable loss function 
    loss_function = nn.BCELoss()

    # Model Training
    def evaluation_function(preds, true):
        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0
        for idx, pred in enumerate(preds):
            if pred == 1:
                if true[idx] == 1:
                    true_pos += 1
                else:
                    false_pos += 1
            if pred == 0:
                if true[idx] == 0:
                    true_neg += 1
                else:
                    false_neg += 1
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) != 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) != 0 else 0
        f1 =  2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        accuracy = (preds == true).float().mean().item()
        return precision, recall, f1, accuracy

    def auc_function(probs, true):
        probs_np = probs.detach().numpy()
        true_np = true.detach().numpy()
        roc_auc = roc_auc_score(true_np, probs_np)
        return roc_auc

    print('Training Neural Network...')

    # Keep the weights that return the best precision
    best_auc = 0
    best_weights = None

    for iter in range(n_iter):
        # Forward pass
        nn_model.train()
        Y_pred_proba = nn_model(X_train)
        Y_pred = torch.round(Y_pred_proba)

        # Calculate loss and other evaluation metrics
        loss = loss_function(Y_pred_proba, Y_train)
        precision, recall, f1, accuracy, roc_auc = *evaluation_function(Y_pred, Y_train), auc_function(Y_pred_proba, Y_train)
        
        nn_optimiser.zero_grad()
        loss.backward()
        nn_optimiser.step()

        # Provide some update on the progress of the training loop
        if iter % 10 == 0:
            print(f'Iteration = {int(iter)} | Loss = {loss:.5f} | Precision = {precision:.3f} | Recall = {recall:.3f} | F1 = {f1:.3f} | Accuracy = {accuracy:.3f} | ROC AUC = {roc_auc:.3f}')

        # Evaluation step
        nn_model.eval()
        Y_pred_test_proba = nn_model(X_test)
        roc_auc_test = auc_function(Y_pred_test_proba, Y_test)
        # Take the model that outputs the best Area Under Curve
        if roc_auc_test > best_auc:
            best_auc = roc_auc_test
            best_weights = copy.deepcopy(nn_model.state_dict())

    # Return the model with the best performance
    nn_model.load_state_dict(best_weights)

    # Compare results with training and test data results
    Y_test_pred_proba = nn_model(X_test)
    probabilities_test_0 = np.array([k for k in Y_test_pred_proba.detach().numpy() if k < 0.5])
    probabilities_test_1 = np.array([k for k in Y_test_pred_proba.detach().numpy() if k >= 0.5])

    # Test probability scatter graph
    plt.scatter(np.arange(len(Y_test_pred_proba)), Y_test_pred_proba.detach().numpy())
    plt.title(r'Deep Neural Network: Test Probabilities')
    plt.show()

    # Evaluation on the test split
    evaluation_test = {}
    Y_test_pred = torch.round(Y_test_pred_proba)
    evals_test = list(evaluation_function(Y_test_pred, Y_test)) + [auc_function(Y_test_pred_proba, Y_test)]
    for idx, score in enumerate(scoring):
        evaluation_test[score] = evals_test[idx]
    print(f'\nThe evaluation scores for the prediction on the test split are: {evaluation_test}')

    Y_train_pred_proba = nn_model(X_train)
    probabilities_train_0 = np.array([k for k in Y_train_pred_proba.detach().numpy() if k < 0.5])
    probabilities_train_1 = np.array([k for k in Y_train_pred_proba.detach().numpy() if k >= 0.5])

    # Train probability scatter graph
    plt.scatter(np.arange(len(Y_train_pred_proba)), Y_train_pred_proba.detach().numpy())
    plt.title(r'Deep Neural Network: Training Probabilities')
    plt.show()
   
    # Evaluation on the train split
    evaluation_train = {}
    Y_train_pred = torch.round(Y_train_pred_proba)
    evals_train = list(evaluation_function(Y_train_pred, Y_train)) + [auc_function(Y_train_pred_proba, Y_train)]
    for idx, score in enumerate(scoring):
        evaluation_train[score] = evals_train[idx]
    print(f'\nWhereas the evaluation scores for the prediction on the training split are: {evaluation_train}')

    # Prediction probability histograms
    plt.hist(probabilities_train_0, 100, alpha=0.3, label='Training Data', color='r', density=True)
    plt.hist(probabilities_test_0, 100, alpha=0.3, label='Test Data', color='b', density=True)
    plt.title('Distribution for probabilities of events with a negative class prediction')
    plt.legend()
    plt.show()

    plt.hist(probabilities_train_1, 100, alpha=0.3, label='Training Data', color='r', density=True)
    plt.hist(probabilities_test_1, 100, alpha=0.3, label='Test Data', color='b', density=True)
    plt.title('Distribution for probabilities of events with a positive class prediction')
    plt.legend()
    plt.show()

    # Shapley Information Gain
    background = X_train[np.random.choice(X_train.detach().numpy().shape[0], 100, replace=False)]
    explainer = shap.DeepExplainer(nn_model, background)
    shap_values = explainer.shap_values(X_test[0:1000])
    shap.summary_plot(shap_values, X_test[0:1000], feature_names=X.columns)
    '''
    return scaler

local_scaler = optimiser(multicollinearity=True, n_estimators=10000,  max_depth=3, colsample_bytree=0.75, learning_rate=0.01, n_iter=10001, test_size=0.3)
'''
xgb, deep = Local_XGB, Local_Deep

# Save the models

with open('XGBoost_3.pkl', 'wb') as f:
    pickle.dump(xgb, f)

with open('Deep_NN_3.pkl', 'wb') as f:
    pickle.dump(deep, f)
'''
scaler = local_scaler
with open('Scaler_2.pkl', 'wb') as f:
    pickle.dump(scaler, f)





