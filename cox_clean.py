import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
from lifelines.plotting import add_at_risk_counts
from lifelines import CoxPHFitter


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

all_ncd = pd.read_feather('path')
#eid as index
all_ncd.set_index('eid', inplace=True)

#drop ['recruitment_date', 'recruitment_centre']
all_ncd.drop(['recruitment_date', 'recruitment_centre'], axis=1, inplace=True)

ukb = pd.read_feather('path')
ukb = ukb.set_index('eid')

all_ncd = all_ncd.join(ukb, how='inner')

ncd_type = ['ACM',
# 'diabetes',
 'IHD',
 'ischemic_stroke',
 'all_stroke',
 'COPD',
 'liver',
 'kidney',
 'all_dementia',
 'alzheimers',
 'parkinsons',
 'rheumatoid',
 'macular',
 'osteoporosis',
 'osteoarthritis',
 'vasc_dementia']


ncd_name = ['All-cause mortality','Ischemic heart disease','Ischemic stroke','All stroke','COPD','Chronic liver disease','Chronic kidney disease','All-cause dementia',"Alzheimer's disease","Parkinson's disease",'Rheumatoid arthritis','Macular degeneration','Osteoporosis','Osteoarthritis','Vascular dementia']

co_var_list1 = ['age_at_recruitment']
co_var_list2 = co_var_list1 + ['recruitment_centre','townsend_deprivation_index','ethnicity','education_years']
co_var_list3 = co_var_list2 + ['IPAQ_activity_group','smoking_status','alcohol_freq','BMI']

co_var_list_all = co_var_list3

cox_models = {}

sex = 'Female'
exposure = 'residual'
cox_models['age'] = {}
sd = mAge_df[sex]['residual'].std()

all_data = all_ncd.join(mAge_df[sex], how='inner')
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

#>55
# all_data = all_data[all_data['age_at_recruitment']>=55]

for ncd_tag,name in zip(ncd_type,ncd_name):

    if ncd_tag == 'ACM':
        temp_df = all_data.copy()
    else:
        condition = (all_data[f'incident_{ncd_tag}'] == 0) & (all_data[f'{ncd_tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df = all_data[~condition]

    # if ncd_tag in ['alzheimers','all_dementia']:
    #     temp_df = temp_df[temp_df['age_at_recruitment']>=60]

    co_var_list_temp = co_var_list1


    temp_df = temp_df[co_var_list_temp+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()
    
    cph = CoxPHFitter()
    formula = 'residual + '+ ' + '.join(co_var_list1)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula
    )
    cox_models['age'][ncd_tag] = cph
    # # extract c index
    # c_ind = cph.concordance_index_

cox_models['model2'] = {}
all_data = all_ncd.join(mAge_df[sex], how='inner')
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for ncd_tag,name in zip(ncd_type,ncd_name):
    print(name)
    if ncd_tag == 'ACM':
        temp_df = all_data.copy()
    else:
        condition = (all_data[f'incident_{ncd_tag}'] == 0) & (all_data[f'{ncd_tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df = all_data[~condition]
    # if ncd_tag in ['alzheimers','all_dementia']:
    #     temp_df = temp_df[temp_df['age_at_recruitment']>=60]
        
    co_var_list_temp = co_var_list2

    temp_df = temp_df[co_var_list_temp+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()

    cph = CoxPHFitter()
    scipy_minimize_options = {'step_size': 0.1}
    formula = 'residual + '+ ' + '.join(co_var_list2)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula,
        fit_options=scipy_minimize_options
    )
    cox_models['model2'][ncd_tag] = cph

cox_models['model3'] = {}
all_data = all_ncd.join(mAge_df[sex], how='inner')
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)

#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for ncd_tag,name in zip(ncd_type,ncd_name):
    print(name)
    if ncd_tag == 'ACM':
        temp_df = all_data.copy()
    else:
        condition = (all_data[f'incident_{ncd_tag}'] == 0) & (all_data[f'{ncd_tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df = all_data[~condition]
    # if ncd_tag in ['alzheimers','all_dementia']:
    #     temp_df = temp_df[temp_df['age_at_recruitment']>=60]
        
    co_var_list_temp = co_var_list3


    temp_df = temp_df[co_var_list_temp+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()

    cph = CoxPHFitter()
    scipy_minimize_options = {'step_size': 0.1}
    formula = 'residual + '+ ' + '.join(co_var_list3)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula,
        fit_options=scipy_minimize_options
    )
    cox_models['model3'][ncd_tag] = cph

# Create a new figure and specify the layout using gridspec
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
# import multipletests
from statsmodels.stats.multitest import multipletests
import math

fig = plt.figure(figsize=(6, 12))
gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig, width_ratios=[1], height_ratios=[14, 1, 1])

# plot a: age and sex
ax = plt.subplot(gs[0, 0], aspect='auto')  # Span the entire left column
# ax.set_title('a', fontweight='bold', loc='left')
# ax.set_title('Model 1', fontsize=9, loc='center')

odds_ratio = False
exposure = 'residual'
#Model1
# Extract hazard ratios and p-values from each model
df1 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio1':[],
                    'p_value1':[],
                    'ci_low1':[],
                    'ci_high1':[],
                    'event_counts1':[]})

for ncd_tag,name in zip(ncd_type,ncd_name):
    model = cox_models['age'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df1
    new_row = {'ncd_name': name,
            'hazard_ratio1': hr,
            'p_value1': pval,
            'ci_low1': clow,
            'ci_high1': chigh,
            'event_counts1': events}
    df1 = pd.concat([df1, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df1['p_value1'], method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, 'darkred', '#bdbdbd')
#add to the df1
df1['colors1'] = colors

#model2
# Extract hazard ratios and p-values from each model
df2 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio2':[],
                    'p_value2':[],
                    'ci_low2':[],
                    'ci_high2':[],
                    'event_counts2':[]})

for ncd_tag,name in zip(ncd_type,ncd_name):
    model = cox_models['model2'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df1
    new_row = {'ncd_name': name,
            'hazard_ratio2': hr,
            'p_value2': pval,
            'ci_low2': clow,
            'ci_high2': chigh,
            'event_counts2': events}
    df2 = pd.concat([df2, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df2['p_value2'], method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, '#14213d', '#bdbdbd')
#add to the df1
df2['colors2'] = colors

#Model3
# Extract hazard ratios and p-values from each model
df3 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio3':[],
                    'p_value3':[],
                    'ci_low3':[],
                    'ci_high3':[],
                    'event_counts3':[]})

for ncd_tag,name in zip(ncd_type,ncd_name):
    model = cox_models['model3'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df3
    new_row = {'ncd_name': name,
            'hazard_ratio3': hr,
            'p_value3': pval,
            'ci_low3': clow,
            'ci_high3': chigh,
            'event_counts3': events}
    df3 = pd.concat([df3, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df3['p_value3'], method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, '#00a087ff', '#bdbdbd')
#add to the df1
df3['colors3'] = colors

#merge df1 and df2
df = df1.merge(df2, on='ncd_name')
df = df.merge(df3, on='ncd_name')

# Sort the dataframes by hazard ratio
df = df.sort_values(by='hazard_ratio1', ascending=True)
#reset index
df = df.reset_index(drop=True)

# Create a horizontal line at y=0
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)

interval = 0.25
# Plot the hazard ratios and confidence intervals with colored dots
for i in range(len(df['hazard_ratio1'])):
    plt.errorbar(
        x = df['hazard_ratio1'][i], 
        y = i+interval, 
        xerr=[[df['hazard_ratio1'][i] - df['ci_low1'][i]], [df['ci_high1'][i] - df['hazard_ratio1'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors1'][i]
    )
    plt.errorbar(
        x = df['hazard_ratio2'][i], 
        y = i, 
        xerr=[[df['hazard_ratio2'][i] - df['ci_low2'][i]], [df['ci_high2'][i] - df['hazard_ratio2'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors2'][i]
    )

    plt.errorbar(
        x = df['hazard_ratio3'][i], 
        y = i-interval, 
        xerr=[[df['hazard_ratio3'][i] - df['ci_low3'][i]], [df['ci_high3'][i] - df['hazard_ratio3'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors3'][i]
    )

# Annotate the number of events to the right of the plot
index = 1.5
plt.text(index, len(df['event_counts1']) + 0.2, 'Events', ha='left', va='center', fontweight='bold')
for i, (count1, count2,count3) in enumerate(zip(df['event_counts1'],df['event_counts2'],df['event_counts3'])):

    plt.text(index, i+interval, f'{int(count1)}', ha='left', va='center', fontsize=9)
    plt.text(index, i, f'{int(count2)}', ha='left', va='center', fontsize=9)
    plt.text(index, i-interval, f'{int(count3)}', ha='left', va='center', fontsize=9)

    
# Annotate the p-values to the right of the plot
plt.text(index+0.08, len(df['p_value1']) + 0.2, 'P-value', ha='left', va='center', fontweight='bold')
for i, (count1, count2,count3) in enumerate(zip(df['p_value1'],df['p_value2'],df['p_value3'])):
    plt.text(index+0.08, i+interval, f'{count1:.2e}', ha='left', va='center', fontsize=9)
    plt.text(index+0.08, i, f'{count2:.2e}', ha='left', va='center', fontsize=9)
    plt.text(index+0.08, i-interval, f'{count3:.2e}', ha='left', va='center', fontsize=9)

#add legend
patch1 = mpatches.Patch(color='darkred', label='Model1',ls='--')
patch2 = mpatches.Patch(color='#14213d', label='Model2',ls='--')
patch3 = mpatches.Patch(color='#00a087ff', label='Model3',ls='--')
patch4 = mpatches.Patch(color='#bdbdbd', label='Not significant',ls='--')

plt.legend(handles=[patch1, patch2, patch3, patch4], loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=4)

plt.xlabel('Hazard ratio (95% CI)')
plt.yticks(range(len(df['hazard_ratio1'])), df['ncd_name'],fontsize=12)
plt.title(f'Multi-variate Cox model for metAgeGap in female', fontweight='bold')

#save as pdf
plt.savefig(f'path', format='pdf', dpi=800, bbox_inches='tight')


ordered_ncd_name = df.sort_values(by='hazard_ratio1', ascending=False)['ncd_name'].to_list()


# Create a dictionary to map names to tags
name_to_tag = dict(zip(ncd_name, ncd_type))


# Get corresponding tags based on the new order of names
ordered_ncd_type = [name_to_tag[name] for name in ordered_ncd_name]

print(ordered_ncd_type)
print(ordered_ncd_name)


df.sort_values(by='hazard_ratio1', ascending=False).to_csv('path')


df.sort_values(by='hazard_ratio1', ascending=False)

# ## Male

cox_models = {}


sex = 'Male'
exposure = 'residual'
cox_models['age'] = {}
sd = mAge_df[sex]['residual'].std()

all_data = all_ncd.join(mAge_df[sex], how='inner')
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for ncd_tag,name in zip(ncd_type,ncd_name):

    if ncd_tag == 'ACM':
        temp_df = all_data
    else:
        condition = (all_data[f'incident_{ncd_tag}'] == 0) & (all_data[f'{ncd_tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df = all_data[~condition]
    # if ncd_tag in ['alzheimers','all_dementia']:
    #     temp_df = temp_df[temp_df['age_at_recruitment']>=60]

    co_var_list_temp = co_var_list1


    temp_df = temp_df[co_var_list_temp+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()
    
    cph = CoxPHFitter()
    formula = 'residual + '+ ' + '.join(co_var_list1)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula
    )
    cox_models['age'][ncd_tag] = cph
    # # extract c index
    # c_ind = cph.concordance_index_

cox_models['model2'] = {}
all_data = all_ncd.join(mAge_df[sex], how='inner')
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)

#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for ncd_tag,name in zip(ncd_type,ncd_name):
    print(name)
    if ncd_tag == 'ACM':
        temp_df = all_data
    else:
        condition = (all_data[f'incident_{ncd_tag}'] == 0) & (all_data[f'{ncd_tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df = all_data[~condition]
        
    # if ncd_tag in ['alzheimers','all_dementia']:
    #     temp_df = temp_df[temp_df['age_at_recruitment']>=60]
    co_var_list_temp = co_var_list2


    temp_df = temp_df[co_var_list_temp+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()

    cph = CoxPHFitter()
    scipy_minimize_options = {'step_size': 0.1}
    formula = 'residual + '+ ' + '.join(co_var_list2)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula,
        fit_options=scipy_minimize_options
    )
    cox_models['model2'][ncd_tag] = cph

cox_models['model3'] = {}
all_data = all_ncd.join(mAge_df[sex], how='inner')
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)

#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for ncd_tag,name in zip(ncd_type,ncd_name):
    print(name)
    if ncd_tag == 'ACM':
        temp_df = all_data.copy()
    else:
        condition = (all_data[f'incident_{ncd_tag}'] == 0) & (all_data[f'{ncd_tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df = all_data[~condition]
    if ncd_tag in ['alzheimers','all_dementia']:
        temp_df = temp_df[temp_df['age_at_recruitment']>=60]
        
    co_var_list_temp = co_var_list3


    temp_df = temp_df[co_var_list_temp+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()

    cph = CoxPHFitter()
    scipy_minimize_options = {'step_size': 0.1}
    formula = 'residual + '+ ' + '.join(co_var_list3)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula,
        fit_options=scipy_minimize_options
    )
    cox_models['model3'][ncd_tag] = cph

# Create a new figure and specify the layout using gridspec
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
# import multipletests
from statsmodels.stats.multitest import multipletests

fig = plt.figure(figsize=(6, 12))
gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig, width_ratios=[1], height_ratios=[14, 1, 1])

# plot a: age and sex
ax = plt.subplot(gs[0, 0], aspect='auto')  # Span the entire left column
# ax.set_title('a', fontweight='bold', loc='left')
# ax.set_title('Model 1', fontsize=9, loc='center')

odds_ratio = False
exposure = 'residual'
#Model1
# Extract hazard ratios and p-values from each model
df1 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio1':[],
                    'p_value1':[],
                    'ci_low1':[],
                    'ci_high1':[],
                    'event_counts1':[]})

for ncd_tag,name in zip(ncd_type,ncd_name):
    model = cox_models['age'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df1
    new_row = {'ncd_name': name,
            'hazard_ratio1': hr,
            'p_value1': pval,
            'ci_low1': clow,
            'ci_high1': chigh,
            'event_counts1': events}
    df1 = pd.concat([df1, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df1['p_value1'], method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, 'darkred', '#bdbdbd')
#add to the df1
df1['colors1'] = colors

#model2
# Extract hazard ratios and p-values from each model
df2 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio2':[],
                    'p_value2':[],
                    'ci_low2':[],
                    'ci_high2':[],
                    'event_counts2':[]})

for ncd_tag,name in zip(ncd_type,ncd_name):
    model = cox_models['model2'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df1
    new_row = {'ncd_name': name,
            'hazard_ratio2': hr,
            'p_value2': pval,
            'ci_low2': clow,
            'ci_high2': chigh,
            'event_counts2': events}
    df2 = pd.concat([df2, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df2['p_value2'], method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, '#14213d', '#bdbdbd')
#add to the df1
df2['colors2'] = colors

#Model3
# Extract hazard ratios and p-values from each model
df3 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio3':[],
                    'p_value3':[],
                    'ci_low3':[],
                    'ci_high3':[],
                    'event_counts3':[]})

for ncd_tag,name in zip(ncd_type,ncd_name):
    model = cox_models['model3'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df3
    new_row = {'ncd_name': name,
            'hazard_ratio3': hr,
            'p_value3': pval,
            'ci_low3': clow,
            'ci_high3': chigh,
            'event_counts3': events}
    df3 = pd.concat([df3, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df3['p_value3'], method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, '#00a087ff', '#bdbdbd')
#add to the df1
df3['colors3'] = colors

#merge df1 and df2
df = df1.merge(df2, on='ncd_name')
df = df.merge(df3, on='ncd_name')

# Sort the dataframes by hazard ratio
df = df.sort_values(by='hazard_ratio1', ascending=True)
#reset index
df = df.reset_index(drop=True)

# Create a horizontal line at y=0
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)

interval = 0.25
# Plot the hazard ratios and confidence intervals with colored dots
for i in range(len(df['hazard_ratio1'])):
    plt.errorbar(
        x = df['hazard_ratio1'][i], 
        y = i+interval, 
        xerr=[[df['hazard_ratio1'][i] - df['ci_low1'][i]], [df['ci_high1'][i] - df['hazard_ratio1'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors1'][i]
    )
    plt.errorbar(
        x = df['hazard_ratio2'][i], 
        y = i, 
        xerr=[[df['hazard_ratio2'][i] - df['ci_low2'][i]], [df['ci_high2'][i] - df['hazard_ratio2'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors2'][i]
    )

    plt.errorbar(
        x = df['hazard_ratio3'][i], 
        y = i-interval, 
        xerr=[[df['hazard_ratio3'][i] - df['ci_low3'][i]], [df['ci_high3'][i] - df['hazard_ratio3'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors3'][i]
    )

# Annotate the number of events to the right of the plot
index = 1.32
plt.text(index, len(df['event_counts1']) + 0.2, 'Events', ha='left', va='center', fontweight='bold')
for i, (count1, count2,count3) in enumerate(zip(df['event_counts1'],df['event_counts2'],df['event_counts3'])):

    plt.text(index, i+interval, f'{int(count1)}', ha='left', va='center', fontsize=9)
    plt.text(index, i, f'{int(count2)}', ha='left', va='center', fontsize=9)
    plt.text(index, i-interval, f'{int(count3)}', ha='left', va='center', fontsize=9)

    
# Annotate the p-values to the right of the plot
plt.text(index+0.08, len(df['p_value1']) + 0.2, 'P-value', ha='left', va='center', fontweight='bold')
for i, (count1, count2,count3) in enumerate(zip(df['p_value1'],df['p_value2'],df['p_value3'])):
    plt.text(index+0.08, i+interval, f'{count1:.2e}', ha='left', va='center', fontsize=9)
    plt.text(index+0.08, i, f'{count2:.2e}', ha='left', va='center', fontsize=9)
    plt.text(index+0.08, i-interval, f'{count3:.2e}', ha='left', va='center', fontsize=9)

#add legend
patch1 = mpatches.Patch(color='darkred', label='Model1',ls='--')
patch2 = mpatches.Patch(color='#14213d', label='Model2',ls='--')
patch3 = mpatches.Patch(color='#00a087ff', label='Model3',ls='--')
patch4 = mpatches.Patch(color='#bdbdbd', label='Not significant',ls='--')

plt.legend(handles=[patch1, patch2, patch3, patch4], loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=4)

plt.xlabel('Hazard ratio (95% CI)')
plt.yticks(range(len(df['hazard_ratio1'])), df['ncd_name'],fontsize=12)
plt.title(f'Multi-variate Cox model for metAgeGap in male', fontweight='bold')

#save as pdf
plt.savefig(f'path', format='pdf', dpi=800, bbox_inches='tight')


ordered_ncd_name = df.sort_values(by='hazard_ratio1', ascending=False)['ncd_name'].to_list()


# Create a dictionary to map names to tags
name_to_tag = dict(zip(ncd_name, ncd_type))


# Get corresponding tags based on the new order of names
ordered_ncd_type = [name_to_tag[name] for name in ordered_ncd_name]

print(ordered_ncd_type)
print(ordered_ncd_name)

df.sort_values(by='hazard_ratio1', ascending=False).to_csv('path')

df.sort_values(by='hazard_ratio1', ascending=False)

# # Cancer

cancer_type = ['Lung', 'Colorectal', 'Pancreas', 'Kidney',  'Oesophagus', 'Liver','Prostate','Breast','Ovary','Brain','Non-Hodgkin lymphoma','Leukaemia']
cancer_name = ['Lung cancer', 'Colorectal cancer', 'Pancreatic cancer', 'Kidney cancer', 'Oesophageal cancer', 'Liver cancer','Prostate cancer','Breast cancer','Ovarian cancer','Brain cancer','Non-Hodgkin lymphoma','Leukaemia']


# ## Female

cox_models = {}
co_var_list_all = co_var_list3

#model1 
sex = 'Female'
cox_models['age'] = {}
sd = mAge_df[sex]['residual'].std()
exposure = 'residual'

all_data = all_ncd.join(mAge_df[sex], how='inner')

all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]


for ncd_tag,name in zip(cancer_type,cancer_name):
    temp_df = all_data[all_data[f'incident_{ncd_tag}'] != 'Prevalent diagnosis']

    if ncd_tag in ['Breast','Ovarian']:
        if sex == 'Male':
            continue
    if ncd_tag in ['Prostate']:
        if sex == 'Female':
            continue


    co_var_list_temp = co_var_list1


    temp_df = temp_df[co_var_list_temp+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()
    
    cph = CoxPHFitter()
    formula = f'{exposure} + '+ ' + '.join(co_var_list_temp)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula
    )
    cox_models['age'][ncd_tag] = cph


#model2
cox_models['model2'] = {}
all_data = all_ncd.join(mAge_df[sex], how='inner')

all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for ncd_tag,name in zip(cancer_type,cancer_name):
    temp_df = all_data[all_data[f'incident_{ncd_tag}'] != 'Prevalent diagnosis']
    if ncd_tag in ['Breast','Ovarian']:
        if sex == 'Male':
            continue
    if ncd_tag in ['Prostate']:
        if sex == 'Female':
            continue
    print(ncd_tag)
    co_var_list_temp = co_var_list2


    temp_df = temp_df[co_var_list_temp+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()
    
    cph = CoxPHFitter()
    formula = f'{exposure} + '+ ' + '.join(co_var_list_temp)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula
    )
    cox_models['model2'][ncd_tag] = cph

#model3
cox_models['model3'] = {}
all_data = all_ncd.join(mAge_df[sex], how='inner')

all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for ncd_tag,name in zip(cancer_type,cancer_name):
    temp_df = all_data[all_data[f'incident_{ncd_tag}'] != 'Prevalent diagnosis']
    if ncd_tag in ['Breast','Ovarian']:
        if sex == 'Male':
            continue
    if ncd_tag in ['Prostate']:
        if sex == 'Female':
            continue
    print(ncd_tag)

    co_var_list_temp = co_var_list3

    temp_df = temp_df[co_var_list_temp+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()

    
    cph = CoxPHFitter()
    scipy_minimize_options = {'step_size': 0.1}

    formula = f'{exposure} + '+ ' + '.join(co_var_list_temp)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula,
        fit_options=scipy_minimize_options

    )
    cox_models['model3'][ncd_tag] = cph

# Create a new figure and specify the layout using gridspec
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
# import multipletests
from statsmodels.stats.multitest import multipletests

exposrue = 'residual'

fig = plt.figure(figsize=(6, 10))
gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig, width_ratios=[1], height_ratios=[12, 1, 1])

# plot a: age and sex
ax = plt.subplot(gs[0, 0], aspect='auto')  # Span the entire left column
# ax.set_title('a', fontweight='bold', loc='left')
# ax.set_title('Model 1', fontsize=9, loc='center')

odds_ratio = False

#Model1
# Extract hazard ratios and p-values from each model
df1 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio1':[],
                    'p_value1':[],
                    'ci_low1':[],
                    'ci_high1':[],
                    'event_counts1':[]})

for ncd_tag,name in zip(cancer_type,cancer_name):
    if ncd_tag in ['Breast','Ovarian']:
        if sex == 'Male':
            continue
    if ncd_tag in ['Prostate']:
        if sex == 'Female':
            continue
    model = cox_models['age'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df1
    new_row = {'ncd_name': name,
            'hazard_ratio1': hr,
            'p_value1': pval,
            'ci_low1': clow,
            'ci_high1': chigh,
            'event_counts1': events}
    df1 = pd.concat([df1, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df1['p_value1'], method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, 'darkred', '#bdbdbd')
#add to the df1
df1['colors1'] = colors

#model2
# Extract hazard ratios and p-values from each model
df2 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio2':[],
                    'p_value2':[],
                    'ci_low2':[],
                    'ci_high2':[],
                    'event_counts2':[]})

for ncd_tag,name in zip(cancer_type,cancer_name):
    if ncd_tag in ['Breast','Ovarian']:
        if sex == 'Male':
            continue
    if ncd_tag in ['Prostate']:
        if sex == 'Female':
            continue
    model = cox_models['model2'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df1
    new_row = {'ncd_name': name,
            'hazard_ratio2': hr,
            'p_value2': pval,
            'ci_low2': clow,
            'ci_high2': chigh,
            'event_counts2': events}
    df2 = pd.concat([df2, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df2['p_value2'], method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, '#14213d', '#bdbdbd')
#add to the df1
df2['colors2'] = colors

#Model3
# Extract hazard ratios and p-values from each model
df3 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio3':[],
                    'p_value3':[],
                    'ci_low3':[],
                    'ci_high3':[],
                    'event_counts3':[]})

for ncd_tag,name in zip(cancer_type,cancer_name):
    if ncd_tag in ['Breast','Ovary']:
        if sex == 'Male':
            continue
    if ncd_tag in ['Prostate']:
        if sex == 'Female':
            continue
    model = cox_models['model3'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df3
    new_row = {'ncd_name': name,
            'hazard_ratio3': hr,
            'p_value3': pval,
            'ci_low3': clow,
            'ci_high3': chigh,
            'event_counts3': events}
    df3 = pd.concat([df3, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df3['p_value3'], method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, '#00a087ff', '#bdbdbd')
#add to the df1
df3['colors3'] = colors

#merge df1 and df2
df = df1.merge(df2, on='ncd_name')
df = df.merge(df3, on='ncd_name')

#remove in df where event_counts1 < 80
df = df[df['event_counts3'] >= 80]

# Sort the dataframes by hazard ratio
df = df.sort_values(by='hazard_ratio1', ascending=True)
#reset index
df = df.reset_index(drop=True)

# Create a horizontal line at y=0
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)

interval = 0.3
# Plot the hazard ratios and confidence intervals with colored dots
for i in range(len(df['hazard_ratio1'])):
    plt.errorbar(
        x = df['hazard_ratio1'][i], 
        y = i+interval, 
        xerr=[[df['hazard_ratio1'][i] - df['ci_low1'][i]], [df['ci_high1'][i] - df['hazard_ratio1'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors1'][i]
    )
    plt.errorbar(
        x = df['hazard_ratio2'][i], 
        y = i, 
        xerr=[[df['hazard_ratio2'][i] - df['ci_low2'][i]], [df['ci_high2'][i] - df['hazard_ratio2'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors2'][i]
    )
    plt.errorbar(
        x = df['hazard_ratio3'][i], 
        y = i-interval, 
        xerr=[[df['hazard_ratio3'][i] - df['ci_low3'][i]], [df['ci_high3'][i] - df['hazard_ratio3'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors3'][i]
    )
# Annotate the number of events to the right of the plot
index = 1.45
plt.text(index, len(df['event_counts1']) + 0.2, 'Events', ha='left', va='center', fontweight='bold')
for i, (count1, count2,count3) in enumerate(zip(df['event_counts1'],df['event_counts2'],df['event_counts3'])):

    plt.text(index, i+interval, f'{int(count1)}', ha='left', va='center', fontsize=9)
    plt.text(index, i, f'{int(count2)}', ha='left', va='center', fontsize=9)
    plt.text(index, i-interval, f'{int(count3)}', ha='left', va='center', fontsize=9)

    
# Annotate the p-values to the right of the plot
plt.text(index+0.1, len(df['p_value1']) + 0.2, 'P-value', ha='left', va='center', fontweight='bold')
for i, (count1, count2,count3) in enumerate(zip(df['p_value1'],df['p_value2'],df['p_value3'])):
    plt.text(index+0.1, i+interval, f'{count1:.2e}', ha='left', va='center', fontsize=9)
    plt.text(index+0.1, i, f'{count2:.2e}', ha='left', va='center', fontsize=9)
    plt.text(index+0.1, i-interval, f'{count3:.2e}', ha='left', va='center', fontsize=9)

#add legend
patch1 = mpatches.Patch(color='darkred', label='Model1',ls='--')
patch2 = mpatches.Patch(color='#14213d', label='Model2',ls='--')
patch3 = mpatches.Patch(color='#00a087ff', label='Model3',ls='--')
patch4 = mpatches.Patch(color='#bdbdbd', label='Not significant',ls='--')

plt.legend(handles=[patch1, patch2, patch3, patch4], loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=4)

plt.xlabel('Hazard ratio (95% CI)')
plt.yticks(range(len(df['hazard_ratio1'])), df['ncd_name'],fontsize=12)
plt.title(f'Multi-variate Cox model for metAgeGap in female', fontweight='bold')

#save as pdf
plt.savefig(f'path', format='pdf', dpi=800, bbox_inches='tight')


ordered_ncd_name = df.sort_values(by='hazard_ratio1', ascending=False)['ncd_name'].to_list()


ordered_ncd_name = df.sort_values(by='hazard_ratio1', ascending=False)['ncd_name'].to_list()


# Create a dictionary to map names to tags
name_to_tag = dict(zip(cancer_name, cancer_type))


# Get corresponding tags based on the new order of names
ordered_ncd_type = [name_to_tag[name] for name in ordered_ncd_name]

print(ordered_ncd_type)
print(ordered_ncd_name)

df.sort_values(by='hazard_ratio1', ascending=False).to_csv('path')

df.sort_values(by='hazard_ratio1', ascending=False)

# ## Male

cox_models = {}
#model1 
co_var_list_all = co_var_list3
sex = 'Male'
co_var_list = []
# 'age_at_recruitment'
cox_models['age'] = {}
sd = mAge_df[sex]['residual'].std()
exposure = 'residual'

all_data = all_ncd.join(mAge_df[sex], how='inner')
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]


for ncd_tag,name in zip(cancer_type,cancer_name):
    temp_df = all_data[all_data[f'incident_{ncd_tag}'] != 'Prevalent diagnosis']

    if ncd_tag in ['Breast','Ovary']:
        if sex == 'Male':
            continue
    if ncd_tag in ['Prostate']:
        if sex == 'Female':
            continue
    print(ncd_tag)
    co_var_list_temp = co_var_list1

    temp_df = temp_df[co_var_list_temp+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()
    
    cph = CoxPHFitter()
    formula = f'{exposure} + '+ ' + '.join(co_var_list_temp)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula
    )
    cox_models['age'][ncd_tag] = cph


#model2
# ,'smoking_status'
cox_models['model2'] = {}

all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for ncd_tag,name in zip(cancer_type,cancer_name):
    print(ncd_tag)
    temp_df = all_data[all_data[f'incident_{ncd_tag}'] != 'Prevalent diagnosis']
    if ncd_tag in ['Breast','Ovary']:
        if sex == 'Male':
            continue
    if ncd_tag in ['Prostate']:
        if sex == 'Female':
            continue
    co_var_list_temp = co_var_list2

    temp_df = temp_df[co_var_list_temp+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()
    
    cph = CoxPHFitter()
    formula = f'{exposure} + '+ ' + '.join(co_var_list_temp)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula
    )
    cox_models['model2'][ncd_tag] = cph

#model3
# ,'smoking_status'
cox_models['model3'] = {}
all_data = all_ncd.join(mAge_df[sex], how='inner')

all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for ncd_tag,name in zip(cancer_type,cancer_name):
    print(ncd_tag)
    temp_df = all_data[all_data[f'incident_{ncd_tag}'] != 'Prevalent diagnosis']
    if ncd_tag in ['Breast','Ovary']:
        if sex == 'Male':
            continue
    if ncd_tag in ['Prostate']:
        if sex == 'Female':
            continue
    co_var_list_temp = co_var_list3

    temp_df = temp_df[co_var_list_temp+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()
    
    cph = CoxPHFitter()
    scipy_minimize_options = {'step_size': 0.1}
    formula = f'{exposure} + '+ ' + '.join(co_var_list_temp)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula,
        fit_options=scipy_minimize_options
    )
    cox_models['model3'][ncd_tag] = cph

# Create a new figure and specify the layout using gridspec
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
# import multipletests
from statsmodels.stats.multitest import multipletests

exposrue = 'residual'

fig = plt.figure(figsize=(6, 10))
gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig, width_ratios=[1], height_ratios=[12, 1, 1])

# plot a: age and sex
ax = plt.subplot(gs[0, 0], aspect='auto')  # Span the entire left column
# ax.set_title('a', fontweight='bold', loc='left')
# ax.set_title('Model 1', fontsize=9, loc='center')

odds_ratio = False

#Model1
# Extract hazard ratios and p-values from each model
df1 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio1':[],
                    'p_value1':[],
                    'ci_low1':[],
                    'ci_high1':[],
                    'event_counts1':[]})

for ncd_tag,name in zip(cancer_type,cancer_name):
    if ncd_tag in ['Breast','Ovary']:
        if sex == 'Male':
            continue
    if ncd_tag in ['Prostate']:
        if sex == 'Female':
            continue
    model = cox_models['age'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df1
    new_row = {'ncd_name': name,
            'hazard_ratio1': hr,
            'p_value1': pval,
            'ci_low1': clow,
            'ci_high1': chigh,
            'event_counts1': events}
    df1 = pd.concat([df1, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df1['p_value1'], method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, 'darkred', '#bdbdbd')
#add to the df1
df1['colors1'] = colors

#model2
# Extract hazard ratios and p-values from each model
df2 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio2':[],
                    'p_value2':[],
                    'ci_low2':[],
                    'ci_high2':[],
                    'event_counts2':[]})

for ncd_tag,name in zip(cancer_type,cancer_name):
    if ncd_tag in ['Breast','Ovary']:
        if sex == 'Male':
            continue
    if ncd_tag in ['Prostate']:
        if sex == 'Female':
            continue
    model = cox_models['model2'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df1
    new_row = {'ncd_name': name,
            'hazard_ratio2': hr,
            'p_value2': pval,
            'ci_low2': clow,
            'ci_high2': chigh,
            'event_counts2': events}
    df2 = pd.concat([df2, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df2['p_value2'], method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, '#14213d', '#bdbdbd')
#add to the df1
df2['colors2'] = colors

#Model3
# Extract hazard ratios and p-values from each model
df3 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio3':[],
                    'p_value3':[],
                    'ci_low3':[],
                    'ci_high3':[],
                    'event_counts3':[]})

for ncd_tag,name in zip(cancer_type,cancer_name):
    if ncd_tag in ['Breast','Ovary']:
        if sex == 'Male':
            continue
    if ncd_tag in ['Prostate']:
        if sex == 'Female':
            continue
    model = cox_models['model3'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df3
    new_row = {'ncd_name': name,
            'hazard_ratio3': hr,
            'p_value3': pval,
            'ci_low3': clow,
            'ci_high3': chigh,
            'event_counts3': events}
    df3 = pd.concat([df3, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df3['p_value3'], method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, '#00a087ff', '#bdbdbd')
#add to the df1
df3['colors3'] = colors

#merge df1 and df2
df = df1.merge(df2, on='ncd_name')
df = df.merge(df3, on='ncd_name')

# Sort the dataframes by hazard ratio
df = df.sort_values(by='hazard_ratio1', ascending=True)
#reset index
df = df.reset_index(drop=True)
#remove in df where event_counts1 < 80
df = df[df['event_counts3'] >= 80]

# Create a horizontal line at y=0
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)

interval = 0.3
# Plot the hazard ratios and confidence intervals with colored dots
for i in range(len(df['hazard_ratio1'])):
    plt.errorbar(
        x = df['hazard_ratio1'][i], 
        y = i+interval, 
        xerr=[[df['hazard_ratio1'][i] - df['ci_low1'][i]], [df['ci_high1'][i] - df['hazard_ratio1'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors1'][i]
    )
    plt.errorbar(
        x = df['hazard_ratio2'][i], 
        y = i, 
        xerr=[[df['hazard_ratio2'][i] - df['ci_low2'][i]], [df['ci_high2'][i] - df['hazard_ratio2'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors2'][i]
    )
    plt.errorbar(
        x = df['hazard_ratio3'][i], 
        y = i-interval, 
        xerr=[[df['hazard_ratio3'][i] - df['ci_low3'][i]], [df['ci_high3'][i] - df['hazard_ratio3'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors3'][i]
    )
# Annotate the number of events to the right of the plot
index = 2.1
plt.text(index, len(df['event_counts1']) + 0.2, 'Events', ha='left', va='center', fontweight='bold')
for i, (count1, count2,count3) in enumerate(zip(df['event_counts1'],df['event_counts2'],df['event_counts3'])):

    plt.text(index, i+interval, f'{int(count1)}', ha='left', va='center', fontsize=9)
    plt.text(index, i, f'{int(count2)}', ha='left', va='center', fontsize=9)
    plt.text(index, i-interval, f'{int(count3)}', ha='left', va='center', fontsize=9)

    
# Annotate the p-values to the right of the plot
plt.text(index+0.2, len(df['p_value1']) + 0.2, 'P-value', ha='left', va='center', fontweight='bold')
for i, (count1, count2,count3) in enumerate(zip(df['p_value1'],df['p_value2'],df['p_value3'])):
    plt.text(index+0.2, i+interval, f'{count1:.2e}', ha='left', va='center', fontsize=9)
    plt.text(index+0.2, i, f'{count2:.2e}', ha='left', va='center', fontsize=9)
    plt.text(index+0.2, i-interval, f'{count3:.2e}', ha='left', va='center', fontsize=9)

#add legend
patch1 = mpatches.Patch(color='darkred', label='Model1',ls='--')
patch2 = mpatches.Patch(color='#14213d', label='Model2',ls='--')
patch3 = mpatches.Patch(color='#00a087ff', label='Model3',ls='--')
patch4 = mpatches.Patch(color='#bdbdbd', label='Not significant',ls='--')

plt.legend(handles=[patch1, patch2, patch3, patch4], loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=4)

plt.xlabel('Hazard ratio (95% CI)')
plt.yticks(range(len(df['hazard_ratio1'])), df['ncd_name'],fontsize=12)
plt.title(f'Multi-variate Cox model for metAgeGap in male', fontweight='bold')

#save as pdf
plt.savefig(f'path', format='pdf', dpi=800, bbox_inches='tight')


sorted_ncd_name = df.sort_values(by='hazard_ratio1', ascending=False)['ncd_name'].to_list()
# Combine ncd_name and ncd_type into dictionary
ncd_dict = dict(zip(cancer_name, cancer_type))

# Extract the sorted ncd_type
sorted_ncd_type = [ncd_dict[name] for name in sorted_ncd_name]


ordered_ncd_name = df.sort_values(by='hazard_ratio1', ascending=False)['ncd_name'].to_list()


# Create a dictionary to map names to tags
name_to_tag = dict(zip(cancer_name, cancer_type))


# Get corresponding tags based on the new order of names
ordered_ncd_type = [name_to_tag[name] for name in ordered_ncd_name]

print(ordered_ncd_type)
print(ordered_ncd_name)

df.sort_values(by='hazard_ratio1', ascending=False).to_csv('path')

df.sort_values(by='hazard_ratio1', ascending=False)



