import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path

#####
# define paths
#####
try:
    # When running as a .py file
    script_dir = Path(__file__).resolve().parent
except NameError:
    # When running interactively (e.g., in Spyder or Jupyter)
    script_dir = Path(os.getcwd()).resolve()

main_path = (script_dir / '..' ).resolve()
savepath = (main_path / 'article_figs' / 'figs' ).resolve()

savepath.mkdir(parents=True, exist_ok=True)

#####
# Set plotting style
#####
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

###############################################################################
#
#
#
###############################################################################
# Load data
df_leaf_area = pd.read_csv(main_path / 'Yang_data/leaf size.csv')

'''
for col in df_leaf_area.columns:
    if col != 'leaf_area':
        unique_vals = df_leaf_area[col].dropna().unique()
        print(f"\nColumn: {col}")
        print(unique_vals)
'''

# Standardize genotype
df_leaf_area.loc[df_leaf_area['genotype'] == 'Col-0', 'genotype'] = 'WT'
df_leaf_area['genotype'] = pd.Categorical(df_leaf_area['genotype'],
                                          categories=["WT", "ROP", "RIC"])

# Remove space and "week" from time values
df_leaf_area['time'] = df_leaf_area['time'].str.replace(' week', '', regex=False)
df_leaf_area['time'] = pd.Categorical(df_leaf_area['time'],
                                      categories=["2", "3", "4", "5", "6"],
                                      ordered=True)

# Map and format leaf order
df_leaf_area['order'] = df_leaf_area['order'].map({
    "6th": "6th leaf",
    "7th": "7th leaf",
    "8th": "8th leaf"
})
df_leaf_area['order'] = pd.Categorical(df_leaf_area['order'],
                                       categories=["6th leaf", "7th leaf", "8th leaf"],
                                       ordered=True)


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
    group_stats = df_subset.groupby(['time', 'genotype'], observed=True).agg(
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
plt.savefig(savepath / "leaf_size_time_yang.pdf")




def detect_robust_plateau(group, initial_threshold=0.1, bump=0.05):
    """
    Returns:
        - plateau_time: timepoint just before plateau
        - threshold_used: which threshold was used
    """
    def _try_detect(threshold):
        for i in range(num_rows - 1):
            next_delta = deltas.iloc[i + 1]
            if next_delta <= threshold:
                print(f"Plateau detected at threshold={threshold:.2f}")
                return group.iloc[i]['time'], threshold
        return None, None

    group = group.sort_values('time_numeric').reset_index(drop=True)
    num_rows = len(group)

    if num_rows < 2:
        print("Not enough data")
        return pd.Series({'plateau_time': 'Not enough data', 'threshold_used': None})

    deltas = group['median_leaf_area'].pct_change().fillna(0)

    # Try initial threshold
    plateau, used_threshold = _try_detect(initial_threshold)
    if plateau is not None:
        return pd.Series({'plateau_time': plateau, 'threshold_used': used_threshold})

    # Try relaxed threshold
    relaxed_threshold = initial_threshold + bump
    plateau, used_threshold = _try_detect(relaxed_threshold)
    if plateau is not None:
        return pd.Series({'plateau_time': plateau, 'threshold_used': used_threshold})

    print(f"No plateau reached for group: {group.iloc[0]['genotype']}, {group.iloc[0]['order']} (max threshold={relaxed_threshold:.2f})")
    return pd.Series({'plateau_time': 'Not reached', 'threshold_used': relaxed_threshold})




# Apply across genotype × leaf order
median_growth = df_leaf_area.groupby(['genotype', 'order', 'time'], observed=True)['leaf_area'].median().reset_index(name='median_leaf_area')
median_growth['time_numeric'] = median_growth['time'].astype(int)

plateau_calls = (
    median_growth
    .sort_values(['genotype', 'order', 'time_numeric'])
    .groupby(['genotype', 'order'], observed=True)
    .apply(detect_robust_plateau)
    .reset_index()
)

print("\n=== Plateau Detection (Visual-Like Method) ===")
print(plateau_calls.to_string(index=False))

# Filter out rows where plateau was not reached
plateau_times_numeric = (
    plateau_calls[plateau_calls['plateau_time'].apply(lambda x: str(x).isdigit())]
    .copy()
)

# Convert plateau_time to numeric (int or float)
plateau_times_numeric['plateau_time'] = plateau_times_numeric['plateau_time'].astype(int)

# Compute average plateau time per genotype
avg_plateau_time_by_genotype = (
    plateau_times_numeric
    .groupby('genotype', observed=True)['plateau_time']
    .mean()
    .reset_index(name='avg_plateau_time')
)

print("\n=== Average Time of Plateau per Genotype ===")
print(avg_plateau_time_by_genotype.to_string(index=False))