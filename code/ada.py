import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from torch.nn import Sequential, LSTM
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
from xgboost import XGBRegressor
# from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def extract_features(deep_smiles):
    try:
        mol = Chem.MolFromSmiles(deep_smiles)
        if mol is None:
            return np.nan
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 1, nBits=2048)
        morgan_fp_array = np.array(morgan_fp)
        return np.array([*morgan_fp_array])
    except:
        return np.nan


def handle_invalid_values(X):
    if np.isnan(X).any() or np.isinf(X).any():
        print("NaN or infinite values detected.")
        X = np.nan_to_num(X)
    return X


def fill_missing_values(row):
    if pd.isnull(row['solvent2 SMILES']) and pd.isnull(row['solvent3 SMILES']):
        return pd.Series([row['solvent1 SMILES'], row['solvent1 SMILES'], row['solvent1 SMILES']],
                         index=['solvent1 SMILES', 'solvent2 SMILES', 'solvent3 SMILES'])
    elif pd.isnull(row['solvent3 SMILES']):
        return pd.Series([row['solvent1 SMILES'], row['solvent2 SMILES'], row['solvent2 SMILES']],
                         index=['solvent1 SMILES', 'solvent2 SMILES', 'solvent3 SMILES'])
    else:
        return pd.Series([row['solvent1 SMILES'], row['solvent2 SMILES'], row['solvent3 SMILES']],
                         index=['solvent1 SMILES', 'solvent2 SMILES', 'solvent3 SMILES'])


# Load your dataset
file_path = 'E:\\zhai\\new_chem\\IA-1114 copy.xlsx'
file_path_new = 'E:\\zhai\\new_chem\\IA-addition-0229.xlsx'
data = pd.read_excel(file_path)
data_new = pd.read_excel(file_path_new)
# data = pd.concat([data, data_new],axis=0)
# Feature extraction
data[['solvent1 SMILES', 'solvent2 SMILES', 'solvent3 SMILES']] = data.apply(fill_missing_values, axis=1)
data['features'] = data['Host SMILES'].apply(extract_features)
data['solvent_features1'] = data['solvent1 SMILES'].apply(extract_features)
data['solvent_features2'] = data['solvent2 SMILES'].apply(extract_features)
data['solvent_features3'] = data['solvent3 SMILES'].apply(extract_features)

# Clean the 'log K' column
data['log K'] = pd.to_numeric(data['log K'], errors='coerce')

# Remove rows with NaN in 'features', 'solvent_features' or 'log K'
data = data.dropna(subset=['features', 'solvent_features1', 'solvent_features2', 'solvent_features3', 'log K'])

# Combine features with guest radius and temperature
data['combined_features'] = data.apply(lambda row: np.concatenate((
    [row['guest radius'], row['T']],
    row['features'],
    row['solvent_features1'],
    row['solvent_features2'],
    row['solvent_features3']
)), axis=1)

# Splitting the data
X = np.array(data['combined_features'].tolist())
y = data['log K'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = handle_invalid_values(X_train)
X_test = handle_invalid_values(X_test)

# Training the model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model = XGBRegressor()
# model = BaggingRegressor()
# model = SVR()
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model = DecisionTreeRegressor(random_state=42)
model = ExtraTreesRegressor()

model.fit(X_train, y_train)

# Predictions
test_pred = model.predict(X_test)

# Calculate the accuracy for each prediction
accuracy = 100 * (1 - np.abs(y_test - test_pred) / y_test)

# Create a DataFrame with test data, predictions, and accuracy
results_df = pd.DataFrame({
    'Guest Radius': X_test[:, 0],  # Assuming the first feature is the guest radius
    'Temperature': X_test[:, 1],  # Assuming the second feature is the temperature
    'Actual Log K': y_test,
    'Predicted Log K': test_pred,
    'Accuracy (%)': accuracy
})

# Mean Squared Error
mse = mean_squared_error(y_test, test_pred)
rscore = r2_score(y_test, test_pred)
# Visualization
# plt.figure(figsize=(12, 6))
# plt.scatter(range(len(y_test)), y_test, color='red', label='Testing data')
# plt.scatter(range(len(y_test)), test_pred, color='orange', marker='x', label='Predictions on testing data')
# plt.title('Testing Data with Predictions')
# plt.xlabel('Data Points')
# plt.ylabel('Log K')
# plt.legend()
# plt.show()

print("Mean Squared Error: {:.2f}".format(mse))
print("R^2 Squared Error: {:.2f}".format(rscore))
