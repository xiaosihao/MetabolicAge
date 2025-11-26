import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, r2_score,mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
from sqlalchemy import create_engine
import pandas as pd 
import pickle
from sklearn.base import clone
from shaphypetune import BoostBoruta
from machine_learning import *
import pickle
import gzip
import plotly.io as pio
from sklearn.model_selection import train_test_split

def plot_regression_results(ax, y_true, y_pred, title, color=None):
    """Scatter plot of the predicted vs true targets."""

    m, b = np.polyfit(y_true,y_pred, 1)

    # ax.plot(
    #     [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--", linewidth=2,
    #     color='#b2182b'
    # )

    #calculate r
    r = np.corrcoef(y_true,y_pred)[0,1]
    #calculate mse
    mse = mean_squared_error(y_true,y_pred)

    ax.plot(y_true, m*y_true + b,color='#14213d', label='r=%.2f \nR2 = %.2f'%(r,r2_score(y_true,y_pred)), linewidth=2)
    ax.text(0.80, 0.0, 'r = %.2f \nR2 = %.2f \nMSE = %.2f'%(r,r2_score(y_true,y_pred),mse), horizontalalignment='left', verticalalignment='bottom', transform=plt.gca().transAxes, fontsize=12)

    ax.scatter(y_true, y_pred, alpha=0.9,s=3, color=color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_pred.min(), y_pred.max()])
    ax.set_xlabel("Chronological age", fontsize=12)
    ax.set_ylabel("Predicted age", fontsize=12)
    # ax.legend(loc="lower right", labelspacing=1.3)

    # title = title + "\n Evaluation in {:.2f} seconds".format(elapsed_time)
    ax.set_title(title, fontsize=14, fontweight='bold',y=1.05)

def score_cv_early_stopping(X,y,model,fit_params,splits=10,random_state = None):
    from sklearn import metrics
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=splits,shuffle=True,random_state=random_state)
    scores = []
    for i, (train, test) in enumerate(cv.split(X, y)):
        clf = clone(model)
        clf.fit(X[train], y[train],**fit_params)
        
        scores.append(clf.score(X[test],y[test]))
    mean_score = np.mean(scores, axis=0)
    return mean_score

def shapley_feature_ranking(shap_values, X):
    feature_order = np.argsort(np.mean(np.abs(shap_values), axis=0))
    return pd.DataFrame(
        {
            "features": [X.columns[i] for i in feature_order][::-1],
            "importance": [
                np.mean(np.abs(shap_values), axis=0)[i] for i in feature_order
            ][::-1],
        }
    )

ukb = pd.read_feather('path')
ukb = ukb.set_index('eid')

#exclude drugs/supplements
drug_df = pd.read_csv('path',index_col=0)
drug_df = drug_df[(drug_df['drug_lipid_lowering']==0)]
ukb = ukb.loc[drug_df.index]

met = pd.read_feather('path')
met = met.set_index('eid')
#left join ukb and met based on index
df = met.join(ukb[['sex','age_at_recruitment']],how='inner')

ukb = pd.read_feather('path')
ukb = ukb.set_index('eid')

early_stopping_rounds = 20
random_state = 1996
sex = 'Female'

with open('path', 'rb') as f:
    param_dict_f = pickle.load(f)


df_train = df[df['sex']==sex]
#drop sex column
df_train = df_train.drop(['sex'],axis=1)


train_df, test_df = train_test_split(df_train, test_size=0.3,random_state=1996,shuffle=True)
train_df, val_df = train_test_split(train_df, test_size=0.2,random_state=1996,shuffle=True)

X_train = train_df.drop(['age_at_recruitment'],axis=1)
y_train = train_df['age_at_recruitment']

X_val = val_df.drop(['age_at_recruitment'],axis=1)
y_val = val_df['age_at_recruitment']

X_test = test_df.drop(['age_at_recruitment'],axis=1)
y_test = test_df['age_at_recruitment']

lm = clone(param_dict_f['init'])

lm.fit(X_train, y_train, eval_set=[(X_val, y_val)],eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)
y_pred = lm.predict(X_test)

fig, axs = plt.subplots(figsize=(6,6))
plot_regression_results(axs, y_test,y_pred, f'Model performance in test dataset in {sex}', color='darkred')
#set xlim and ylim
axs.set_xlim(39, 72)
axs.set_ylim(39, 72)
#save figure
plt.savefig(f'path', bbox_inches='tight', dpi=300)

from joblib import Parallel, delayed

def train_and_evaluate_fold(train_idx, val_idx, X, y, param_dict_f):
    # Splitting the data
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]
    
    # Cloning and fitting the model
    model = clone(param_dict_f['init'])
    model.fit(X_train_fold, y_train_fold,eval_set=[(X_val_fold, y_val_fold)],eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)
    
    # Predicting and calculating R2 score
    y_pred = model.predict(X_val_fold)
    r2 = r2_score(y_val_fold, y_pred)
    
    return r2

# Configuring KFold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
X = pd.concat([X_train, X_val])
y = pd.concat([y_train, y_val])

# Parallel execution using joblib
r2_scores = Parallel(n_jobs=-1)(
    delayed(train_and_evaluate_fold)(train_idx, val_idx, X, y, param_dict_f)
    for train_idx, val_idx in cv.split(X, y)
)

mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
print(f'R2 score in 5 fold CV: {mean_r2} +/- {std_r2}')

early_stopping_rounds = 20
random_state = 1996
sex = 'Male'

with open(f'path', 'rb') as f:
    param_dict = pickle.load(f)


df_train = df[df['sex']==sex]
#drop sex column
df_train = df_train.drop(['sex'],axis=1)

train_df, test_df = train_test_split(df_train, test_size=0.3,random_state=1996,shuffle=True)
train_df, val_df = train_test_split(train_df, test_size=0.2,random_state=1996,shuffle=True)

X_train = train_df.drop(['age_at_recruitment'],axis=1)
y_train = train_df['age_at_recruitment']

X_val = val_df.drop(['age_at_recruitment'],axis=1)
y_val = val_df['age_at_recruitment']

X_test = test_df.drop(['age_at_recruitment'],axis=1)
y_test = test_df['age_at_recruitment']

lm = clone(param_dict['init'])

lm.fit(X_train, y_train, eval_set=[(X_val, y_val)],eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)
y_pred = lm.predict(X_test)

fig, axs = plt.subplots(figsize=(6,6))
plot_regression_results(axs, y_test,y_pred, f'Model performance in test dataset in {sex}', color='#3c5488ff')
#set xlim and ylim
axs.set_xlim(39, 72)
axs.set_ylim(39, 72)
plt.savefig(f'path', bbox_inches='tight', dpi=300)

from joblib import Parallel, delayed

def train_and_evaluate_fold(train_idx, val_idx, X, y, param_dict):
    # Splitting the data
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]
    
    # Cloning and fitting the model
    model = clone(param_dict['init'])
    model.fit(X_train_fold, y_train_fold,eval_set=[(X_val_fold, y_val_fold)],eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)
    
    # Predicting and calculating R2 score
    y_pred = model.predict(X_val_fold)
    r2 = r2_score(y_val_fold, y_pred)
    
    return r2

# Configuring KFold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
X = pd.concat([X_train, X_val])
y = pd.concat([y_train, y_val])

# Parallel execution using joblib
r2_scores = Parallel(n_jobs=-1)(
    delayed(train_and_evaluate_fold)(train_idx, val_idx, X, y, param_dict)
    for train_idx, val_idx in cv.split(X, y)
)

mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
print(f'R2 score in 5 fold CV: {mean_r2} +/- {std_r2}')

early_stopping_rounds = 20
random_state = 1996
sex = 'Female'

with open('path', 'rb') as f:
    param_dict = pickle.load(f)


with open('path', 'rb') as f:
    model = pickle.load(f)

df_train = df[df['sex']==sex]
#drop sex column
df_train = df_train.drop(['sex'],axis=1)

train_df, test_df = train_test_split(df_train, test_size=0.3,random_state=1996,shuffle=True)
train_df, val_df = train_test_split(train_df, test_size=0.2,random_state=1996,shuffle=True)

X_train = train_df.drop(['age_at_recruitment'],axis=1).iloc[:,model.support_]
y_train = train_df['age_at_recruitment']

X_val = val_df.drop(['age_at_recruitment'],axis=1).iloc[:,model.support_]
y_val = val_df['age_at_recruitment']

X_test = test_df.drop(['age_at_recruitment'],axis=1).iloc[:,model.support_]
y_test = test_df['age_at_recruitment']

lm = clone(param_dict['boruta'])

lm.fit(X_train, y_train, eval_set=[(X_val, y_val)],eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)
y_pred = lm.predict(X_test)

fig, axs = plt.subplots(figsize=(6,6))
plot_regression_results(axs, y_test,y_pred, f'Model performance in test dataset in {sex}', color='darkred')
#set xlim and ylim
axs.set_xlim(39, 72)
axs.set_ylim(39, 72)
plt.savefig(f'path', bbox_inches='tight', dpi=300)

from joblib import Parallel, delayed

def train_and_evaluate_fold(train_idx, val_idx, X, y, param_dict):
    # Splitting the data
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]
    
    # Cloning and fitting the model
    model = clone(param_dict['boruta'])
    model.fit(X_train_fold, y_train_fold,eval_set=[(X_val_fold, y_val_fold)],eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)
    
    # Predicting and calculating R2 score
    y_pred = model.predict(X_val_fold)
    r2 = r2_score(y_val_fold, y_pred)
    
    return r2

# Configuring KFold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
X = pd.concat([X_train, X_val])
y = pd.concat([y_train, y_val])

# Parallel execution using joblib
r2_scores = Parallel(n_jobs=-1)(
    delayed(train_and_evaluate_fold)(train_idx, val_idx, X, y, param_dict)
    for train_idx, val_idx in cv.split(X, y)
)

mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
print(f'R2 score in 5 fold CV: {mean_r2} +/- {std_r2}')

early_stopping_rounds = 20
random_state = 1996
sex = 'Male'

with open(f'path', 'rb') as f:
    param_dict = pickle.load(f)


with open(f'path', 'rb') as f:
    model = pickle.load(f)

df_train = df[df['sex']==sex]
#drop sex column
df_train = df_train.drop(['sex'],axis=1)

train_df, test_df = train_test_split(df_train, test_size=0.3,random_state=1996,shuffle=True)
train_df, val_df = train_test_split(train_df, test_size=0.2,random_state=1996,shuffle=True)

X_train = train_df.drop(['age_at_recruitment'],axis=1).iloc[:,model.support_]
y_train = train_df['age_at_recruitment']

X_val = val_df.drop(['age_at_recruitment'],axis=1).iloc[:,model.support_]
y_val = val_df['age_at_recruitment']

X_test = test_df.drop(['age_at_recruitment'],axis=1).iloc[:,model.support_]
y_test = test_df['age_at_recruitment']

lm = clone(param_dict['boruta'])

lm.fit(X_train, y_train, eval_set=[(X_val, y_val)],eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)
y_pred = lm.predict(X_test)

fig, axs = plt.subplots(figsize=(6,6))
plot_regression_results(axs, y_test,y_pred, f'Model performance in test dataset in {sex}', color='#3c5488ff')
#set xlim and ylim
axs.set_xlim(39, 72)
axs.set_ylim(39, 72)
plt.savefig(f'path', bbox_inches='tight', dpi=300)

from joblib import Parallel, delayed

def train_and_evaluate_fold(train_idx, val_idx, X, y, param_dict):
    # Splitting the data
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]
    
    # Cloning and fitting the model
    model = clone(param_dict['init'])
    model.fit(X_train_fold, y_train_fold,eval_set=[(X_val_fold, y_val_fold)],eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)
    
    # Predicting and calculating R2 score
    y_pred = model.predict(X_val_fold)
    r2 = r2_score(y_val_fold, y_pred)
    
    return r2

# Configuring KFold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
X = pd.concat([X_train, X_val])
y = pd.concat([y_train, y_val])

# Parallel execution using joblib
r2_scores = Parallel(n_jobs=-1)(
    delayed(train_and_evaluate_fold)(train_idx, val_idx, X, y, param_dict)
    for train_idx, val_idx in cv.split(X, y)
)

mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
print(f'R2 score in 5 fold CV: {mean_r2} +/- {std_r2}')

sex = 'Male'

with open(f'path', 'rb') as f:
    model = pickle.load(f)

met_list_m = df.drop(['sex','age_at_recruitment'],axis=1).iloc[:,model.support_].columns.to_list()

sex = 'Female'

with open(f'path', 'rb') as f:
    model = pickle.load(f)

met_list_f = df.drop(['sex','age_at_recruitment'],axis=1).iloc[:,model.support_].columns.to_list()


#Calculate mAge for all samples
from joblib import Parallel, delayed

sex = 'Male'
with open(f'path', 'rb') as f:
    param_dict = pickle.load(f)

with open(f'path', 'rb') as f:
    model = pickle.load(f)

df_train = df[df['sex']==sex]
#drop sex column
df_train = df_train.drop(['sex'],axis=1)

cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

def train_predict_fold(train_idx, test_idx, df_train, model, param_dict, random_state, early_stopping_rounds):
    df_train_fold = df_train.iloc[train_idx]
    df_train_fold, df_val_fold = train_test_split(df_train_fold, test_size=0.2, random_state=1996, shuffle=True)
    df_test_fold = df_train.iloc[test_idx]

    X_train_fold = df_train_fold.drop(['age_at_recruitment'], axis=1).iloc[:, model.support_]
    y_train_fold = df_train_fold['age_at_recruitment']

    X_val_fold = df_val_fold.drop(['age_at_recruitment'], axis=1).iloc[:, model.support_]
    y_val_fold = df_val_fold['age_at_recruitment']

    X_test_fold = df_test_fold.drop(['age_at_recruitment'], axis=1).iloc[:, model.support_]
    y_test_fold = df_test_fold['age_at_recruitment']

    # Train model
    lm = clone(param_dict['boruta'])
    lm.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)

    # Predict
    y_pred_test = lm.predict(X_test_fold)
    return pd.DataFrame(y_pred_test, index=X_test_fold.index)

all_preds = Parallel(n_jobs=-1, verbose=10)(
    delayed(train_predict_fold)(train_idx, test_idx, df_train, model, param_dict, random_state, early_stopping_rounds)
    for i, (train_idx, test_idx) in enumerate(cv.split(df_train))
)

# Concatenate all predictions
all_preds_m = pd.concat(all_preds)


random_state = 1996
early_stopping_rounds = 20
#Calculate mAge for all samples
from joblib import Parallel, delayed

sex = 'Female'
with open(f'path', 'rb') as f:
    param_dict = pickle.load(f)

with open(f'path', 'rb') as f:
    model = pickle.load(f)

df_train = df[df['sex']==sex]
#drop sex column
df_train = df_train.drop(['sex'],axis=1)

cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

def train_predict_fold(train_idx, test_idx, df_train, model, param_dict, random_state, early_stopping_rounds):
    df_train_fold = df_train.iloc[train_idx]
    df_train_fold, df_val_fold = train_test_split(df_train_fold, test_size=0.2, random_state=1996, shuffle=True)
    df_test_fold = df_train.iloc[test_idx]

    X_train_fold = df_train_fold.drop(['age_at_recruitment'], axis=1).iloc[:, model.support_]
    y_train_fold = df_train_fold['age_at_recruitment']

    X_val_fold = df_val_fold.drop(['age_at_recruitment'], axis=1).iloc[:, model.support_]
    y_val_fold = df_val_fold['age_at_recruitment']

    X_test_fold = df_test_fold.drop(['age_at_recruitment'], axis=1).iloc[:, model.support_]
    y_test_fold = df_test_fold['age_at_recruitment']

    # Train model
    lm = clone(param_dict['boruta'])
    lm.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)

    # Predict
    y_pred_test = lm.predict(X_test_fold)
    return pd.DataFrame(y_pred_test, index=X_test_fold.index)

all_preds = Parallel(n_jobs=-1, verbose=10)(
    delayed(train_predict_fold)(train_idx, test_idx, df_train, model, param_dict, random_state, early_stopping_rounds)
    for i, (train_idx, test_idx) in enumerate(cv.split(df_train))
)

# Concatenate all predictions
all_preds_f = pd.concat(all_preds)

#change the name of column to PreAge
all_preds_m = all_preds_m.rename(columns={0: 'PreAge'})
all_preds_f = all_preds_f.rename(columns={0: 'PreAge'})

all_preds_m = all_preds_m.join(df['age_at_recruitment'],how='left')
all_preds_f = all_preds_f.join(df['age_at_recruitment'],how='left')

from scipy.stats import linregress

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(all_preds_m['age_at_recruitment'], all_preds_m['PreAge'])

# Calculate predicted values using the regression line equation (y = mx + b)
predicted_col2 = slope * all_preds_m['age_at_recruitment'] + intercept

# Calculate residuals by subtracting the predicted values from the actual values
all_preds_m['residual'] = all_preds_m['PreAge'] - predicted_col2

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(all_preds_f['age_at_recruitment'], all_preds_f['PreAge'])

# Calculate predicted values using the regression line equation (y = mx + b)
predicted_col2 = slope * all_preds_f['age_at_recruitment'] + intercept

# Calculate residuals by subtracting the predicted values from the actual values
all_preds_f['residual'] = all_preds_f['PreAge'] - predicted_col2

#save all_preds_f to feather file
all_preds_f.reset_index().to_feather('path')
all_preds_m.reset_index().to_feather('path')