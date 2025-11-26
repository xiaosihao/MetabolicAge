import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
from lifelines.plotting import add_at_risk_counts


### get top, middle, bottom 10%
def cut_by_thresholds(series):
    top_10_threshold = series.quantile(0.9)
    bottom_10_threshold = series.quantile(0.1)
    median_10_top = series.quantile(0.55)
    median_10_bottom = series.quantile(0.45)
 
    categories = pd.Series(['Other'] * len(series), index=series.index)
    categories[series > top_10_threshold] = 'Top 10%'
    categories[(series <= median_10_top) & (series >= median_10_bottom)] = 'Median 10%'
    categories[series < bottom_10_threshold] = 'Bottom 10%'
 
    return categories

### get top, middle, bottom 25%
def cut_by_thresholds25(series):
    top_10_threshold = series.quantile(0.75)
    bottom_10_threshold = series.quantile(0.25)
    median_10_top = series.quantile(0.65)
    median_10_bottom = series.quantile(0.35)
 
    categories = pd.Series(['Other'] * len(series), index=series.index)
    categories[series > top_10_threshold] = 'Top 25%'
    categories[(series <= median_10_top) & (series >= median_10_bottom)] = 'Median 25%'
    categories[series < bottom_10_threshold] = 'Bottom 25%'
 
    return categories

#Read metabolic age
all_preds_f = pd.read_feather('path')
#eid as string
all_preds_f['eid'] = all_preds_f['eid'].astype(str)
all_preds_f.set_index('eid', inplace=True)

all_preds_m = pd.read_feather('path')
#eid as string
all_preds_m['eid'] = all_preds_m['eid'].astype(str)
all_preds_m.set_index('eid', inplace=True)

mAge_df = {'Female':all_preds_f, 'Male':all_preds_m}

cox_df_f = pd.read_csv('path',index_col=0)
cox_df_m = pd.read_csv('path',index_col=0)

#select ncd_name only if p_value1, p_value2, p_value3 are less than 0.05 from cox_df_f
ncd_name_f = cox_df_f[cox_df_f['p_value1']<0.05][cox_df_f['p_value2']<0.05][cox_df_f['p_value3']<0.05]['ncd_name'].to_list()
ncd_name_m = cox_df_m[cox_df_m['p_value1']<0.05][cox_df_m['p_value2']<0.05][cox_df_m['p_value3']<0.05]['ncd_name'].to_list()
ncd_selected = [i for i in ncd_name_f if i in ncd_name_m]

ncd_selected += ['Colorectal cancer']

#For Common diseases
all_ncd = pd.read_feather('path')
#eid as string
all_ncd['eid'] = all_ncd['eid'].astype(str)
all_ncd.set_index('eid', inplace=True)

ncd_type = ['ACM', 'IHD', 'ischemic_stroke', 'all_stroke', 'COPD', 'liver', 'kidney', 'all_dementia', 'alzheimers', 'parkinsons', 'rheumatoid', 'macular', 'osteoporosis', 'osteoarthritis', 'vasc_dementia','Colorectal']
ncd_name = ['All-cause mortality','Ischemic heart disease','Ischemic stroke','All stroke','COPD','Chronic liver disease','Chronic kidney disease','All-cause dementia',"Alzheimer's disease","Parkinson's disease",'Rheumatoid arthritis','Macular degeneration','Osteoporosis','Osteoarthritis','Vascular dementia','Colorectal cancer']

ncd_dict = dict(zip(ncd_name, ncd_type))

import matplotlib.patches as mpatches
# Create a figure and subplots
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
# fig.suptitle(f"Cumulative risks of different disease types by deciles of pAD in men",y=1, fontsize=24)
fig.text(-0.01, 0.5, 'Cumulative risks',ha='center', rotation='vertical',size=18)
fig.text(0.5,-0.01, 'Chronological age',ha='center',size=18)

# Flatten the axes array to simplify indexing
axes = axes.flatten()
i=0

for name in ncd_selected:
    ncd_tag = ncd_dict[name]
    ax = axes[i]

    
    color = iter(['#3c5488ff','darkred'])
    for sex in ['Male','Female']:
        c = next(color)
        all_preds_df = all_ncd.join(mAge_df[sex], how='inner')
        if ncd_tag == 'ACM':
            temp_df_all = all_preds_df.copy()
            #fill na with 0 in {tag}_event
            temp_df_all[f'{ncd_tag}_event'] = temp_df_all[f'{ncd_tag}_event'].fillna(0)
        else:
            condition = (all_preds_df[f'incident_{ncd_tag}'] == 0) & (all_preds_df[f'{ncd_tag}_event'] == 1)
            # Use boolean indexing to remove rows where the condition is True
            temp_df_all = all_preds_df[~condition]

        temp_df_all['mAge_decile'] = cut_by_thresholds(temp_df_all['residual'])

        temp_df = temp_df_all[temp_df_all['mAge_decile'].isin(['Top 10%','Median 10%','Bottom 10%'])]

        style = iter(['-','--'])

        T = temp_df[f'{ncd_tag}_survival_time']

        T = T/365.25+temp_df['age_at_recruitment']

        E = temp_df[f'{ncd_tag}_event']
        #fill the missing value with 0
        E = E.fillna(0)

        #limit to 80 years old
        #if T>80 then set E=0 T=80
        E = np.where(T>80,0,E)
        T = np.where(T>80,80,T)

        groups = temp_df[f'mAge_decile']

        s = next(style)
        ix = (groups == 'Top 10%')
        kmf = KaplanMeierFitter()
        kmf.fit(T[ix], E[ix], label=f'{sex} top 10%')
        kmf.plot_cumulative_density(color=c,ax=ax,loc=slice(45,75),linestyle=s )

        s = next(style)
        ix = (groups == 'Bottom 10%')
        kmf = KaplanMeierFitter()
        kmf.fit(T[ix], E[ix], label=f'{sex} bottom 10%')
        kmf.plot_cumulative_density(color=c,ax=ax,loc=slice(45,75),linestyle=s ) 
    #set the x axis limit
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    n = int(temp_df_all[f'{ncd_tag}_event'].sum())
    ax.set_title(f'{name} (n={n})',fontsize=14)
    ax.legend().set_visible(False)

    #remove the legend
    i+=1       

#remove the last two empty plot
fig.delaxes(axes[-1])
fig.delaxes(axes[-2])
# fig.delaxes(axes[-3])
# Add a legend to the bottom
# Add a legend to the bottom
handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=14, bbox_to_anchor=(0.5, -0.15))

plt.tight_layout()
plt.savefig(f'path', format='pdf', dpi=800, bbox_inches='tight')


