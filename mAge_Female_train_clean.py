import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, r2_score
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


n_trials = 150
early_stopping_rounds = 20
random_state = 1996
sex = 'Female'

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

df_train = df[df['sex']==sex]

#drop sex column
df_train = df_train.drop(['sex'],1)

train_df, test_df = train_test_split(df_train, test_size=0.3,random_state=1996,shuffle=True)
train_df, val_df = train_test_split(train_df, test_size=0.2,random_state=1996,shuffle=True)


X_train = pd.concat([train_df.drop(['age_at_recruitment'],1),val_df.drop(['age_at_recruitment'],1)])
y_train = pd.concat([train_df['age_at_recruitment'],val_df['age_at_recruitment']])

def optuna_lgbm(X, y,storage,study_name,n_trials,early_stopping_rounds):
    # make sqlite database engine to run with optuna
    engine = create_engine(storage, echo=False)

    def objective(trial):
        params = {
            'objective': 'regression',
            'verbose': -1,
            'boosting_type': 'gbdt',
            'n_estimators': 5000,
            'num_leaves': trial.suggest_int('num_leaves', 2, 100),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 2, 100),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 100),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1),
            'random_state': random_state,
            'metric': 'None',
            'n_jobs': -1
        }
        
        #Stratified KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        r2_scores = []
        for train_idx, val_idx in cv.split(X,y):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
        
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)
        
            y_pred = model.predict(X_val_fold)
            r2 = r2_score(y_val_fold, y_pred)

            r2_scores.append(r2)
        
        return np.mean(r2_scores)

    # Run the optimization using optuna
    study = optuna.create_study(direction='maximize',storage=storage,study_name=study_name)
    study.optimize(objective, n_trials=n_trials)

    return study

# Run optuna
storage = f'sqlite:////path'

#load param_dict
with open(f'path', 'rb') as f:
    param_dict = pickle.load(f)


study1 = optuna_lgbm(X_train, y_train,storage,'init',n_trials,early_stopping_rounds)

# Get the best hyperparameters and train the final model
best_params = study1.best_params

base_params = {'boosting_type': 'gbdt','metric':"None",'n_estimators':5000,'n_jobs':-1,'random_state':random_state}
base_params.update(best_params)
best_model =lgb.LGBMRegressor(**base_params)

#save best model
param_dict['init'] = best_model


with open(f'path', 'wb') as f:
    pickle.dump(param_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print('finish')