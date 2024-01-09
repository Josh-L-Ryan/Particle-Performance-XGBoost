import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Params which define the appearance of graphs

params = {
    'axes.labelsize': 14,  # Set size for both x and y axis labels
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.minor.size': 4,   # minor tick length
    'ytick.minor.size': 4,  
    'xtick.major.size': 6,   # major tick length
    'ytick.major.size': 6,
    'xtick.major.width': 1.4,   # major tick width
    'ytick.major.width': 1.4,
    'xtick.minor.width': 1.0,   # minor tick width
    'ytick.minor.width': 1.0
}

plt.rcParams.update(params)

# Create empty lists to store errors
all_train_errors_y1 = []
all_val_errors_y1 = []
all_train_errors_y2 = []
all_val_errors_y2 = []

# Read in experimental data
df = pd.read_csv(r'E:\Wax Project Y4\Hinokitiol release data\Neural Network\Wax Experiment Conditions temp5.csv')
X = df.drop(['Plateau Abs 25c', 'Plateau Abs 80c', 'Abs Frac', 'Expt #', 'Median Size'], axis=1).values

# Normalize the raw data
y1_original = df['Abs Frac'].values
y2_original = df['Median Size'].values
y1_scaler = MinMaxScaler()
y2_scaler = MinMaxScaler()
y1 = y1_scaler.fit_transform(y1_original.reshape(-1, 1))
y2 = y2_scaler.fit_transform(y2_original.reshape(-1, 1))
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# k-fold cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds)

all_predictions_y1 = []
all_predictions_y2 = []

#Train the model
for train_idx, val_idx in kf.split(X_normalized):
    
    train_data, val_data = X_normalized[train_idx], X_normalized[val_idx]
    train_label_1, val_label_1 = y1[train_idx], y1[val_idx]
    train_label_2, val_label_2 = y2[train_idx], y2[val_idx]
    
    feature_names = ['Loading', 'H/L', 'Surf.', 'Stirring', '$\it t$', 'BA']
    
    dtrain_1 = xgb.DMatrix(train_data, label=train_label_1, feature_names=feature_names)
    dval_1 = xgb.DMatrix(val_data, label=val_label_1, feature_names=feature_names)
    dtrain_2 = xgb.DMatrix(train_data, label=train_label_2, feature_names=feature_names)
    dval_2 = xgb.DMatrix(val_data, label=val_label_2, feature_names=feature_names)

    
    # Hyperparameters for model tuning
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'booster': 'gbtree',
        'colsample_bytree': 0.7, 'eta': 0.80, 'gamma': 0.05, 'lambda': 0.84, 'max_depth': 2, 'min_child_weight': 1.0
        }

    bst1 = xgb.train(param, dtrain_1, num_boost_round=1000, evals=[(dval_1, 'validation_y1')], verbose_eval=False)
    val_predictions_y1 = bst1.predict(dval_1)
    all_predictions_y1.extend(val_predictions_y1)

    
    bst2 = xgb.train(param, dtrain_2, num_boost_round=1000, evals=[(dval_2, 'validation_y2')], verbose_eval=False)
    val_predictions_y2 = bst2.predict(dval_2)
    all_predictions_y2.extend(val_predictions_y2)

    # Calculate errors for y1
    train_predictions_y1 = bst1.predict(dtrain_1)
    train_mae_y1 = mean_absolute_error(train_label_1, train_predictions_y1)
    val_mae_y1 = mean_absolute_error(val_label_1, val_predictions_y1)
    all_train_errors_y1.append(train_mae_y1)
    all_val_errors_y1.append(val_mae_y1)

    # Calculate errors for y2
    train_predictions_y2 = bst2.predict(dtrain_2)
    train_mae_y2 = mean_absolute_error(train_label_2, train_predictions_y2)
    val_mae_y2 = mean_absolute_error(val_label_2, val_predictions_y2)
    all_train_errors_y2.append(train_mae_y2)
    all_val_errors_y2.append(val_mae_y2)


# Get a list of all unique features
all_features = list(set(bst1.get_score(importance_type='gain').keys()) | set(bst2.get_score(importance_type='gain').keys()))

# Create a color mapping for each feature importance
colors = plt.cm.viridis(np.linspace(0, 1, len(all_features)))  # Generating a range of colors
feature_color_map = dict(zip(all_features, colors))

# Plotting function with feature-specific colors for feature importances
def plot_with_colors(bst, title):
    ax = plot_importance(bst, title=title, xlabel='F-score', ylabel='Features', importance_type='gain', show_values=False)
    ax.grid(False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    
    for idx, p in enumerate(ax.patches):
        feature = ax.get_yticklabels()[idx].get_text()
        if feature in feature_color_map:
            p.set_color(feature_color_map[feature])


# Plotting feature importance for both models with feature-specific colors
plt.figure(figsize=(12, 8))
plot_with_colors(bst1, '')
plt.subplots_adjust(left=0.126, bottom=0.11, right=0.971, top=1.0, wspace=0.2, hspace=0.2)
#plt.savefig(r"E:\Thesis Chapters\Wax Chapter\Figures\XGBoost feature importances\ER importance fig2.png", dpi=600)
#plt.show()


plt.figure(figsize=(12, 8))
plot_with_colors(bst2, '')
plt.subplots_adjust(left=0.126, bottom=0.112, right=0.971, top=0.993, wspace=0.2, hspace=0.2)
#plt.savefig(r"E:\Thesis Chapters\Wax Chapter\Figures\XGBoost feature importances\Median size importance fig2.png", dpi=600)
plt.show()



# Convert predictions back to original scale
all_predictions_y1 = y1_scaler.inverse_transform(np.array(all_predictions_y1).reshape(-1, 1)).flatten()
all_predictions_y2 = y2_scaler.inverse_transform(np.array(all_predictions_y2).reshape(-1, 1)).flatten()



print(f"Average Train MAE for y1: {np.mean(all_train_errors_y1)}")
print(f"Average Validation MAE for y1: {np.mean(all_val_errors_y1)}")
print(f"Average Train MAE for y2: {np.mean(all_train_errors_y2)}")
print(f"Average Validation MAE for y2: {np.mean(all_val_errors_y2)}")


# Gridsearch for optimal experimental conditions which minimize y1, y2
from itertools import product

# Define a grid for each parameter
param_grid = {
    'wax_loading': [0.04, 0.16, 0.28, 0.40],
    'wax_ab' : [0.5, 1, 2],
    'surf_conc' : [0, 1, 2, 5],
    'stir_speed' : [350, 500, 750, 1000],
    'emul_time' : [2, 5, 15, 30],
    'theo_ai_loading' : [0.5, 1.25, 5, 10],
}


# Gridsearch for optimal experimental conditions

# Generate all combinations of parameters
#all_combinations = list(product(*param_grid.values()))

#top_combinations = [{'combination': None, 'score': float('inf')} for _ in range(10)]

#for combination in all_combinations:
#    feature_values = list(combination) 
#    input_features = np.array(feature_values).reshape(1, -1)
#    input_features_normalized = scaler.transform(input_features)
    
    # Predicted values in normalized scale
#    predicted_y1_norm = bst1.predict(xgb.DMatrix(input_features_normalized))
#    predicted_y2_norm = bst2.predict(xgb.DMatrix(input_features_normalized))
    
    # Convert the predicted values back to their original scales
#    predicted_y1_original = y1_scaler.inverse_transform(predicted_y1_norm.reshape(-1, 1)).flatten()[0]
#    predicted_y2_original = y2_scaler.inverse_transform(predicted_y2_norm.reshape(-1, 1)).flatten()[0]
    
    # If the score is better than the worst score in top_combinations
#    if predicted_y1_original < top_combinations[-1]['score']:
#        top_combinations[-1] = {'combination': combination, 'score': predicted_y1_original, 'predicted_y2': predicted_y2_original}
        # Sort the top_combinations by score
#        top_combinations = sorted(top_combinations, key=lambda x: x['score'])

#Print out best experimental conditions
#for idx, item in enumerate(top_combinations):
#    print(f"Rank {idx+1}: Combination {item['combination']} with Predicted y1 (Abs Frac): {item['score']}, Predicted y2 (Median Size): {item['predicted_y2']}")



# Visualization
plt.scatter(y1_original, y2_original, color='green', label='True Values')
plt.scatter(all_predictions_y1, all_predictions_y2, color='blue', alpha=0.5, label='Predictions')
plt.xlabel('Abs Frac')
plt.ylabel('Median Size')
plt.legend()
plt.show()


