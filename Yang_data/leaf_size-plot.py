import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','nature']) # sans-serif font

plt.close('all')

plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.rc('axes', labelsize=12)

params = {'legend.fontsize': 12,
         'axes.labelsize': 12,
         'axes.titlesize':12,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
plt.rcParams.update(params)


cmap=sns.color_palette("colorblind")

savepath = '/home/isabella/Documents/PLEN/x-ray/article_figs/figs/'

###############################################################################
#
#
#
###############################################################################
# Load data
df_leaf_area = pd.read_csv('/home/isabella/Documents/PLEN/x-ray/Yang_data/leaf size.csv')

df_leaf_area.loc[df_leaf_area['genotype'] == 'Col-0', 'genotype'] = 'WT'
df_leaf_area.columns
# Format columns
df_leaf_area['genotype'] = pd.Categorical(df_leaf_area['genotype'], categories=["WT", "ROP", "RIC"])
df_leaf_area['time'] = df_leaf_area['time'].str.replace('week', '')
df_leaf_area['time'] = pd.Categorical(df_leaf_area['time'], categories=["2", "3", "4", "5"], ordered=True)
df_leaf_area['order'] = df_leaf_area['order'].map({"6th": "6th leaf", "7th": "7th leaf", "8th": "8th leaf"})
df_leaf_area['order'] = pd.Categorical(df_leaf_area['order'], categories=["6th leaf", "7th leaf", "8th leaf"], ordered=True)


# Compute mean and SD
mean_sd = df_leaf_area.groupby(['genotype', 'order', 'time'], observed=True).agg(
    mean=('leaf_area', 'mean'),
    sd=('leaf_area', 'std')
).reset_index()

palette = {"WT": "darkturquoise", "RIC": "indigo", "ROP": "#FF6700"}

# Setup 3-panel mosaic figure
figsize = (8.27,3.2)

fig, axd = plt.subplot_mosaic("ABC", figsize=figsize)
leaf_orders = ["6th leaf", "7th leaf", "8th leaf"]
axes_labels = ['A', 'B', 'C']

for i, order in enumerate(leaf_orders):
    ax = axd[axes_labels[i]]
    df_subset = df_leaf_area[df_leaf_area['order'] == order]
    
    sns.stripplot(data=df_subset,
                  x='time', y='leaf_area', hue='genotype',
                  dodge=True, jitter=0.25, alpha=0.3,
                  ax=ax, palette=palette)

    # Mean ± SD overlay
    group_stats = df_subset.groupby(['time', 'genotype']).agg(
        mean=('leaf_area', 'mean'), sd=('leaf_area', 'std')).reset_index()

    for j, row in group_stats.iterrows():
        xpos = df_subset['time'].cat.categories.tolist().index(row['time'])
        offset = {"WT": -0.25, "ROP": 0, "RIC": 0.25}[row['genotype']]
        ax.errorbar(x=xpos + offset, y=row['mean'],
                    yerr=row['sd'], fmt='o', color=palette[row['genotype']],
                    capsize=5, lw=1.5)

    # Labels and appearance
    ax.set_title(order)
    ax.set_xlabel("Time (week)")
    if i == 0:
        ax.set_ylabel("Leaf area (cm²)")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    if i != 0:
        ax.get_legend().remove()
    else:
        ax.legend()

plt.tight_layout()
plt.savefig(savepath+"leaf_size_time_yang.pdf")