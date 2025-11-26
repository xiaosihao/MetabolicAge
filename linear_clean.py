import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

#Read metabolic age
all_preds_f = pd.read_feather('path')
#eid as str
all_preds_f['eid'] = all_preds_f['eid'].astype(str)
all_preds_f.set_index('eid', inplace=True)
#drop age_at_recruitment
all_preds_f.drop(columns=['age_at_recruitment'], inplace=True)

all_preds_m = pd.read_feather('path')
all_preds_m['eid'] = all_preds_m['eid'].astype(str)
all_preds_m.set_index('eid', inplace=True)
#drop age_at_recruitment
all_preds_m.drop(columns=['age_at_recruitment'], inplace=True)

mAge_df = {'Female':all_preds_f, 'Male':all_preds_m}

#Read UKB data
ukb = pd.read_feather('path')
ukb = ukb.set_index('eid')

all_ncd = pd.read_feather('path')
#eid as index
all_ncd.set_index('eid', inplace=True)

ukb = ukb.join(all_ncd[['prevalent_hypertension','prevalent_obesity','prevalent_diabetes']], how='left',on='eid')

wh_df = pd.read_csv('path')
#change column names to eid, hip, waist
wh_df.columns = ['eid','hip','waist']
#eid as str type
wh_df['eid'] = wh_df['eid'].astype(str)

wh_df.set_index('eid', inplace=True)
#calculate waist hip ratio
wh_df['waist_hip_ratio'] = wh_df['waist']/wh_df['hip']
#join to ukb
ukb = ukb.join(wh_df[['waist_hip_ratio']], how='left',on='eid')
#drug data
drug_df = pd.read_csv('path',index_col=0)
drug_df.index = drug_df.index.astype(str)
#join to ukb
ukb = ukb.join(drug_df, how='left',on='eid')


ukb['smoking_ever'] = np.where(ukb['smoking_status'].isna(), np.nan,
                                  np.where(ukb['smoking_status'].isin(['Current','Previous']), 1, 0))

#replace value in alcohol_freq column One to three times a month to 1, Once or twice a week to 2, Three or four times a week to 3, Daily or almost daily to 4 and keep the NA
ukb['alcohol_freq'] = np.where(ukb['alcohol_freq'].isna(), np.nan,
                               np.where(ukb['alcohol_freq'] == 'One to three times a month', 1,
                                        np.where(ukb['alcohol_freq'] == 'Once or twice a week', 2,
                                                 np.where(ukb['alcohol_freq'] == 'Three or four times a week', 3,
                                                          np.where(ukb['alcohol_freq'] == 'Daily or almost daily', 4, np.nan)))))


#change symbol
bio_type_new =[
    # 'Albumin',
    'Alkaline_phosphatase',
    # 'Apolipoprotein_A',
    # 'Apolipoprotein_B',
    'Calcium',
    'Glucose',
    'HbA1c',
    # 'HDL_cholesterol',
    # 'Lipoprotein_A',
    # 'Phosphate',
    'SHBG',
    'Testosterone',
    'Total_protein',
    'Urate',
    'Urea',
    'Vitamin_D',
    'LTL_zadj',
    'IGF-1',
    'C-reactive_protein',
    # 'Creatinine',
    'Cystatin_C',
    'Alanine_aminotransferase',
    'Aspartate_aminotransferase',
    # 'AST/ALT',
    'Gamma_glutamyltransferase',
    'Total_bilirubin',
    # 'Cholesterol',
    # 'Triglycerides'
]

bio_type_old =[
    # 'Albumin',
    'Alkaline phosphatase',
    # 'Apolipoprotein A',
    # 'Apolipoprotein B',
    'Calcium',
    'Glucose',
    'Glycated haemoglobin (HbA1c)',
    # 'HDL cholesterol',
    # 'Lipoprotein A',
    # 'Phosphate',
    'SHBG',
    'Testosterone',
    'Total protein',
    'Urate',
    'Urea',
    'Vitamin D',
    'LTL_zadj',
    'IGF-1',
    'C-reactive protein',
    # 'Creatinine',
    'Cystatin C',
    'Alanine aminotransferase',
    'Aspartate aminotransferase',
    # 'AST/ALT',
    'Gamma glutamyltransferase',
    'Total bilirubin',
    # 'Cholesterol',
    # 'Triglycerides'
]
bio_type =[
    # 'Albumin',
    'Alkaline_phosphatase',
    # 'Apolipoprotein_A',
    # 'Apolipoprotein_B',
    'Calcium',
    'Glucose',
    'HbA1c',
    # 'HDL_cholesterol',
    # 'Lipoprotein_A',
    # 'Phosphate',
    'SHBG',
    'Testosterone',
    'Total_protein',
    'Urate',
    'Urea',
    'Vitamin_D',
    'LTL_zadj',
    'IGF_1',
    'C_reactive_protein',
    # 'Creatinine',
    'Cystatin_C',
    'Alanine_aminotransferase',
    'Aspartate_aminotransferase',
    # 'AST/ALT',
    'Gamma_glutamyltransferase',
    'Total_bilirubin',
    # 'Cholesterol',
    # 'Triglycerides'
]
bio_name = [
    # 'Albumin',
    'Alkaline phosphatase',
    # 'Apolipoprotein A',
    # 'Apolipoprotein B',
    'Calcium',
    'Glucose',
    'Glycated haemoglobin (HbA1c)',
    # 'HDL cholesterol',
    # 'Lipoprotein A',
    # 'Phosphate',
    'SHBG',
    'Testosterone',
    'Total protein',
    'Urate',
    'Urea',
    'Vitamin D',
    'Telomere length',
    'IGF-1',
    'C-reactive protein',
    # 'Creatinine',
    'Cystatin C',
    'Alanine aminotransferase',
    'Aspartate aminotransferase',
    # 'AST/ALT',
    'Gamma glutamyltransferase',
    'Total bilirubin',
    # 'Cholesterol',
    # 'Triglycerides'
]


phys_type = [
    'hand_grip_strength_right_normalized',
    'hand_grip_strength_left_normalized',
    'BMI',
    'waist_hip_ratio',
    # 'FEV1_standardized',
    # 'FVC_standardized',
    # 'FEV1_FVC_ratio_z_score',
    # 'haemoglobin_concentration',
    'heel_bone_mineral_density',
    'pulse_wave_arterial_stiffness_index',
    'systolic_bp',
    'diastolic_bp',
    'identify_matches_mean_time',
    'fluid_intelligence',
    # '',
    'overall_health_poor',
    'usual_walking_pace_slow',
    'facial_ageing_older',
    'tiredness_freq_everyday',
    'sleep_difficulty_usually',
    'sleep_hours_10',
    'prevalent_hypertension',
    'prevalent_obesity',
    'prevalent_diabetes',
    'smoking_ever',
    'pack_years',
    'alcohol_freq'
    ]
phys_name = [
    'Hand grip strength (right)',
    'Hand grip strength (left)',
    'BMI',
    'Waist hip ratio',
    # 'Lung function (FEV1)',
    # 'Lung function (FVC)',
    # 'Lung function (FEV1/FVC)',
    # 'Haemoglobin concentration',
    'Heel bone mineral density',
    'Arterial stiffness index',
    'Systolic blood pressure',
    'Diastolic blood pressure',
    'Reaction time',
    'Fluid intelligence',
    # 'Frailty index (continuous)',
    'Poor self-rated health',
    'Slow walking pace',
    'Self-rated facial aging',
    'Tired/lethargic every day',
    'Frequent insomnia',
    'Sleep 10+ hours / day',
    'Hypertension',
    'Obesity',
    'Type II Diabetes',
    'Ever smoker',
    'Smoking Packyears',
    'Alcohol frequency'
]

binary_type = ['overall_health_poor',
    'usual_walking_pace_slow',
    'facial_ageing_older',
    'tiredness_freq_everyday',
    'sleep_difficulty_usually',
    'sleep_hours_10',
    'prevalent_hypertension',
    'prevalent_obesity',
    'prevalent_diabetes',
    'smoking_ever'
    ]

#change columns in bio_type into bio_type_new in ukb
ukb.rename(columns=dict(zip(bio_type_old,bio_type)), inplace=True)

lm_models = {}

exposure = 'residual'
sex = 'Male'
lm_models[sex] = {}
lm_models[sex][exposure] = {'model1':{}}

co_var_list = ['recruitment_centre','ethnicity','education_years','townsend_deprivation_index']
# 'age_at_recruitment','sex',,'townsend_deprivation_index','education_years','IPAQ_activity_group','BMI'
all_data = ukb.join(mAge_df[sex], how='inner')
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for tag in (bio_type + phys_type):
    temp_df = all_data.copy()

    if tag == 'BMI':
        co_var_list_temp = [x for x in co_var_list if x != 'BMI']
    else:
        co_var_list_temp = co_var_list[:]

    # if tag in ['systolic_bp','diastolic_bp','Calcium','prevalent_hypertension']:
    #     print(tag)
    #     print(temp_df.shape)
    #     temp_df = temp_df[~(temp_df['drug_hypertensive'] == 1)]
    #     print(temp_df.shape)
    #     # co_var_list_temp = co_var_list_temp + ['blood_pressure_meds']
    # else:
    #     temp_df = temp_df
    

    temp_df = temp_df[co_var_list_temp + [tag] + [exposure]]
    temp_df = temp_df.dropna()

    formula = f'{exposure} ~ {tag} + ' + ' + '.join(co_var_list_temp)

    if tag in binary_type:
        model = sm.formula.ols(formula=formula, data=temp_df).fit()
    else:
        #make tag standardized
        if tag != 'LTL_zadj':
            temp_df[tag] = (temp_df[tag] - temp_df[tag].mean()) / temp_df[tag].std()
        model = sm.formula.ols(formula=formula, data=temp_df).fit()
    
    lm_models[sex][exposure]['model1'][tag] = model
        

exposure = 'residual'
sex = 'Female'
lm_models[sex] = {}
lm_models[sex][exposure] = {'model1':{}}

co_var_list = ['recruitment_centre','ethnicity','education_years','townsend_deprivation_index']
# 'age_at_recruitment','sex',,'townsend_deprivation_index','education_years','IPAQ_activity_group','BMI'
all_data = ukb.join(mAge_df[sex], how='inner')
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for tag in (bio_type + phys_type):
    temp_df = all_data.copy()

    if tag == 'BMI':
        co_var_list_temp = [x for x in co_var_list if x != 'BMI']
    else:
        co_var_list_temp = co_var_list[:]

    # if tag in ['systolic_bp','diastolic_bp','Calcium','prevalent_hypertension']:
    #     temp_df = temp_df[~(temp_df['drug_hypertensive'] == 'Yes')]
    #     # co_var_list_temp = co_var_list_temp + ['blood_pressure_meds']
    # else:
    #     temp_df = temp_df
    

    temp_df = temp_df[co_var_list_temp + [tag] + [exposure]]
    temp_df = temp_df.dropna()

    formula = f'{exposure} ~ {tag} + ' + ' + '.join(co_var_list_temp)

    if tag in binary_type:
        model = sm.formula.ols(formula=formula, data=temp_df).fit()
    else:
        #make tag standardized
        if tag != 'LTL_zadj':
            temp_df[tag] = (temp_df[tag] - temp_df[tag].mean()) / temp_df[tag].std()
        model = sm.formula.ols(formula=formula, data=temp_df).fit()
    
    lm_models[sex][exposure]['model1'][tag] = model
        

from statsmodels.stats.multitest import multipletests

exposure = 'residual'
sex = 'Female'

# List to store hazard ratios and p-values
effect_size = []
p_values = []
ci_low_values = []
ci_high_values = []
event_counts = []

for tag in bio_type:
    # Get hazard ratio and p-value
    model = lm_models[sex][exposure]['model1'][tag]

    es = model.params[tag]
    p = model.pvalues[tag]
    clow = model.conf_int().loc[tag, 0]
    chigh = model.conf_int().loc[tag, 1]
    event_count = model.nobs

    effect_size.append(es)
    p_values.append(p)
    ci_low_values.append(clow)
    ci_high_values.append(chigh)
    event_counts.append(event_count)

sorted_indices = np.argsort(effect_size)
sorted_hr_values = np.array(effect_size)[sorted_indices]
sorted_ci_low = np.array(ci_low_values)[sorted_indices]
sorted_ci_high = np.array(ci_high_values)[sorted_indices]
sorted_disease_list = np.array(bio_name)[sorted_indices]
sorted_pvals = np.array(p_values)[sorted_indices]
sorted_events = np.array(event_counts)[sorted_indices]


# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(sorted_pvals, method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, 'darkred', '#bdbdbd')

bio_df_f = pd.DataFrame({'name':sorted_disease_list, 'pval_f':sorted_pvals, 'fdr_pval_f':fdr_corrected_pvals,'hr_f':sorted_hr_values, 'ci_low_f':sorted_ci_low, 'ci_high_f':sorted_ci_high, 'colors_f':colors})

effect_size = []
p_values = []
ci_low_values = []
ci_high_values = []
event_counts = []

for tag in phys_type:
    # Get hazard ratio and p-value
    model = lm_models[sex][exposure]['model1'][tag]

    es = model.params[tag]
    p = model.pvalues[tag]
    clow = model.conf_int().loc[tag, 0]
    chigh = model.conf_int().loc[tag, 1]
    event_count = model.nobs

    effect_size.append(es)
    p_values.append(p)
    ci_low_values.append(clow)
    ci_high_values.append(chigh)
    event_counts.append(event_count)

sorted_indices = np.argsort(effect_size)
sorted_hr_values = np.array(effect_size)[sorted_indices]
sorted_ci_low = np.array(ci_low_values)[sorted_indices]
sorted_ci_high = np.array(ci_high_values)[sorted_indices]
sorted_disease_list = np.array(phys_name)[sorted_indices]
sorted_pvals = np.array(p_values)[sorted_indices]
sorted_events = np.array(event_counts)[sorted_indices]


# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(sorted_pvals, method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, 'darkred', '#bdbdbd')

# colors = np.where(sorted_pvals < 0.05, 'darkred', 'black')
phy_df_f = pd.DataFrame({'name':sorted_disease_list, 'pval_f':sorted_pvals, 'fdr_pval_f':fdr_corrected_pvals,'hr_f':sorted_hr_values, 'ci_low_f':sorted_ci_low, 'ci_high_f':sorted_ci_high,'colors_f':colors})

#Male
sex = 'Male'

effect_size = []
p_values = []
ci_low_values = []
ci_high_values = []
event_counts = []

for tag in bio_type:
    # Get hazard ratio and p-value
    model = lm_models[sex][exposure]['model1'][tag]

    es = model.params[tag]
    p = model.pvalues[tag]
    clow = model.conf_int().loc[tag, 0]
    chigh = model.conf_int().loc[tag, 1]
    event_count = model.nobs

    effect_size.append(es)
    p_values.append(p)
    ci_low_values.append(clow)
    ci_high_values.append(chigh)
    event_counts.append(event_count)

sorted_indices = np.argsort(effect_size)
sorted_hr_values = np.array(effect_size)[sorted_indices]
sorted_ci_low = np.array(ci_low_values)[sorted_indices]
sorted_ci_high = np.array(ci_high_values)[sorted_indices]
sorted_disease_list = np.array(bio_name)[sorted_indices]
sorted_pvals = np.array(p_values)[sorted_indices]
sorted_events = np.array(event_counts)[sorted_indices]


# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(sorted_pvals, method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, '#3c5488ff', '#bdbdbd')

bio_df_m = pd.DataFrame({'name':sorted_disease_list, 'pval_m':sorted_pvals, 'fdr_pval_m':fdr_corrected_pvals,'hr_m':sorted_hr_values, 'ci_low_m':sorted_ci_low, 'ci_high_m':sorted_ci_high, 'colors_m':colors})

# List to store hazard ratios and p-values
effect_size = []
p_values = []
ci_low_values = []
ci_high_values = []
event_counts = []

for tag in phys_type:
    # Get hazard ratio and p-value
    model = lm_models[sex][exposure]['model1'][tag]

    es = model.params[tag]
    p = model.pvalues[tag]
    clow = model.conf_int().loc[tag, 0]
    chigh = model.conf_int().loc[tag, 1]
    event_count = model.nobs

    effect_size.append(es)
    p_values.append(p)
    ci_low_values.append(clow)
    ci_high_values.append(chigh)
    event_counts.append(event_count)

sorted_indices = np.argsort(effect_size)
sorted_hr_values = np.array(effect_size)[sorted_indices]
sorted_ci_low = np.array(ci_low_values)[sorted_indices]
sorted_ci_high = np.array(ci_high_values)[sorted_indices]
sorted_disease_list = np.array(phys_name)[sorted_indices]
sorted_pvals = np.array(p_values)[sorted_indices]
sorted_events = np.array(event_counts)[sorted_indices]


# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(sorted_pvals, method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, '#3c5488ff', '#bdbdbd')

# colors = np.where(sorted_pvals < 0.05, 'darkred', 'black')
phy_df_m = pd.DataFrame({'name':sorted_disease_list, 'pval_m':sorted_pvals, 'fdr_pval_m':fdr_corrected_pvals,'hr_m':sorted_hr_values, 'ci_low_m':sorted_ci_low, 'ci_high_m':sorted_ci_high,'colors_m':colors})

#combine male and female
bio_df = bio_df_f.merge(bio_df_m, on='name')
phy_df = phy_df_f.merge(phy_df_m, on='name')

phy_dict = dict(zip(phys_type,phys_name))
binary_name = [phy_dict[x] for x in binary_type]

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(6, 6))
# Create a horizontal line at y=0
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
temp_df = phy_df.copy()
#select only if name in binary_name
temp_df = temp_df[temp_df['name'].isin(binary_name)]
# Sort the dataframes by hazard ratio
temp_df = temp_df.sort_values(by='hr_f', ascending=True)
#reset index
temp_df = temp_df.reset_index(drop=True)

interval = 0.2
# Plot the hazard ratios and confidence intervals with colored dots
for i in range(len(temp_df['hr_f'])):
    plt.errorbar(
        x = temp_df['hr_f'][i], 
        y = i+interval, 
        xerr=[[temp_df['hr_f'][i] - temp_df['ci_low_f'][i]], [temp_df['ci_high_f'][i] - temp_df['hr_f'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=temp_df['colors_f'][i]
    )
    plt.errorbar(
        x = temp_df['hr_m'][i], 
        y = i-interval, 
        xerr=[[temp_df['hr_m'][i] - temp_df['ci_low_m'][i]], [temp_df['ci_high_m'][i] - temp_df['hr_m'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=temp_df['colors_m'][i]
    )
# Annotate the number of events to the right of the plot
index =1.6
    
# Annotate the p-values to the right of the plot
plt.text(index+0, len(temp_df['fdr_pval_f']) + -0.1, 'P-value', ha='left', va='center', fontweight='bold')
for i, (count1, count2) in enumerate(zip(temp_df['fdr_pval_f'],temp_df['fdr_pval_m'])):
    plt.text(index+0, i+interval, f'{count1:.2e}', ha='left', va='center', fontsize=9)
    plt.text(index+0, i-interval, f'{count2:.2e}', ha='left', va='center', fontsize=9)

plt.xlabel('Beta')
plt.yticks(range(len(temp_df['hr_f'])), temp_df['name'],fontsize=12)
plt.title(f'Binary variables', fontweight='bold')
plt.xlim(-0.3,1.55)

#add legend
patch1 = mpatches.Patch(color='darkred', label='Female',ls='--')
patch2 = mpatches.Patch(color='#3c5488ff', label='Male',ls='--')


plt.legend(handles=[patch1, patch2], loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2)


plt.tight_layout()
plt.savefig(f'path', format='pdf', dpi=800, bbox_inches='tight')


fig, ax = plt.subplots(figsize=(6, 6))
# Create a horizontal line at y=0
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
temp_df = phy_df.copy()
#select only if name not in binary_name
temp_df = temp_df[~temp_df['name'].isin(binary_name)]
# Sort the dataframes by hazard ratio
temp_df = temp_df.sort_values(by='hr_f', ascending=True)
#reset index
temp_df = temp_df.reset_index(drop=True)

interval = 0.2
# Plot the hazard ratios and confidence intervals with colored dots
for i in range(len(temp_df['hr_f'])):
    plt.errorbar(
        x = temp_df['hr_f'][i], 
        y = i+interval, 
        xerr=[[temp_df['hr_f'][i] - temp_df['ci_low_f'][i]], [temp_df['ci_high_f'][i] - temp_df['hr_f'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=temp_df['colors_f'][i]
    )
    plt.errorbar(
        x = temp_df['hr_m'][i], 
        y = i-interval, 
        xerr=[[temp_df['hr_m'][i] - temp_df['ci_low_m'][i]], [temp_df['ci_high_m'][i] - temp_df['hr_m'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=temp_df['colors_m'][i]
    )
# Annotate the number of events to the right of the plot
index =1.6
    
# Annotate the p-values to the right of the plot
plt.text(index+0, len(temp_df['fdr_pval_f']) + -0.1, 'P-value', ha='left', va='center', fontweight='bold')
for i, (count1, count2) in enumerate(zip(temp_df['fdr_pval_f'],temp_df['fdr_pval_m'])):
    plt.text(index+0, i+interval, f'{count1:.2e}', ha='left', va='center', fontsize=9)
    plt.text(index+0, i-interval, f'{count2:.2e}', ha='left', va='center', fontsize=9)

plt.xlabel('Beta')
plt.yticks(range(len(temp_df['hr_f'])), temp_df['name'],fontsize=12)
plt.title(f'Continuous variables', fontweight='bold')
plt.xlim(-0.3,1.55)

#add legend
patch1 = mpatches.Patch(color='darkred', label='Female',ls='--')
patch2 = mpatches.Patch(color='#3c5488ff', label='Male',ls='--')


plt.legend(handles=[patch1, patch2], loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2)

plt.tight_layout()
plt.savefig(f'path', format='pdf', dpi=800, bbox_inches='tight')


