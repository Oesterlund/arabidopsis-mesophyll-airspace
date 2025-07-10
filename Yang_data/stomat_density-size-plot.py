#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 18:58:33 2025

@author: isabella
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import bootstrap
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
plt.style.use(['science', 'nature'])

# Global plot settings
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)
params = {
    'legend.fontsize': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
}
plt.rcParams.update(params)

plt.close('all')

palette = {"WT": "darkturquoise", "RIC": "indigo", "ROP": "#FF6700"}

###############################################################################
#
# functions
#
###############################################################################

def pval_to_star(pval):
    if pval >= 0.05:
        return "ns"
    elif 0.01 <= pval < 0.05:
        return "*"
    elif 0.001 < pval < 0.01:
        return "**"
    elif 0.0001 < pval <= 0.001:
        return "***"
    else:
        return "****"

def plot_with_mixedmodel_tukey(df, value_col, ylabel, save_name, scaling=1.0):
    """
    df : dataframe
    value_col : str, column name with the values to plot
    ylabel : str, label for the y-axis
    save_name : str, name for saving the plot
    scaling : float, multiply value_col by this (default 1.0)
    """
    figsize = (9, 5)
    fig, axs = plt.subplot_mosaic([['week3_abaxial', 'week3_adaxial', 'week5_abaxial', 'week5_adaxial']],
                                   figsize=figsize)

    # Format categories
    df['genotype'] = pd.Categorical(df['genotype'], categories=["WT", "ROP", "RIC"], ordered=True)
    df['time'] = pd.Categorical(df['time'], categories=["week3", "week5"], ordered=True)
    df['side'] = pd.Categorical(df['side'], categories=["abaxial", "adaxial"], ordered=True)
    df['plant_num'] = df['plant_num'].astype(str)

    for (time, side), ax_name in zip(
        [("week3", "abaxial"), ("week3", "adaxial"), ("week5", "abaxial"), ("week5", "adaxial")],
        ['week3_abaxial', 'week3_adaxial', 'week5_abaxial', 'week5_adaxial']
    ):
        ax = axs[ax_name]
        subset = df[(df['time'] == time) & (df['side'] == side)].copy()

        subset[value_col] *= scaling
        
        sns.stripplot(data=subset, x='genotype', y=value_col,
              hue='genotype', dodge=False, jitter=0.25, alpha=0.2,
              ax=ax, palette=palette, legend=False)

        # Get actual xtick positions and labels
        xticks = ax.get_xticks()
        xticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
        tick_lookup = dict(zip(xticklabels, xticks))


        # mean ± 90% bootstrap confidence interval
        group_stats = []
        
        for genotype in ["WT", "ROP", "RIC"]:
            data = subset[subset['genotype'] == genotype][value_col].dropna().values
            if len(data) > 1:
                res = bootstrap((data,), np.mean, n_resamples=10000, confidence_level=0.90, method="basic")
                mean_val = np.mean(data)
                ci_lower, ci_upper = res.confidence_interval
                group_stats.append({"genotype": genotype, "mean": mean_val, "ci_lower": ci_lower, "ci_upper": ci_upper})
            else:
                # If only one point, CI cannot be computed
                mean_val = np.mean(data)
                group_stats.append({"genotype": genotype, "mean": mean_val, "ci_lower": mean_val, "ci_upper": mean_val})
        
        group_stats = pd.DataFrame(group_stats)

        for j, row in group_stats.iterrows():
            x = tick_lookup.get(row['genotype'], j)
            yerr_lower = row['mean'] - row['ci_lower']
            yerr_upper = row['ci_upper'] - row['mean']

            ax.errorbar(x=x, y=row['mean'],
                        yerr=[[yerr_lower], [yerr_upper]],
                        fmt='o', color=palette[row['genotype']],
                        capsize=5, lw=1.5)


        # Create dataframe for TukeyHSD
        subset_pred = subset[['genotype', value_col]].dropna()

        # Tukey's HSD on the raw data (approximated like emmeans in R)
        tukey = pairwise_tukeyhsd(endog=subset_pred[value_col],
                                  groups=subset_pred['genotype'],
                                  alpha=0.05)

        summary_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

        ymax = subset[value_col].max()

        for idx, row in summary_df.iterrows():
            group1 = row['group1']
            group2 = row['group2']
            pval = row['p-adj']
            star = pval_to_star(pval)

            xpos1 = tick_lookup[group1]
            xpos2 = tick_lookup[group2]
            x_center = (xpos1 + xpos2) / 2
            y = ymax + (idx + 1) * (ymax*0.05)  # vertical spacing

            # Plot brackets and stars
            ax.plot([xpos1, xpos1, xpos2, xpos2], [y-0.01*ymax, y, y, y-0.01*ymax], lw=1.5, color='black')
            ax.text(x_center, y+0.01*ymax, star, ha='center', va='bottom', fontsize=10)

        week_number = time.replace('week', 'Week ')  # adds space and capitalizes
        ax.set_title(f"{week_number} {side}")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(savepath, save_name), dpi=300)
    #plt.close()

###############################################################################
#
# data and creation of plots
#
###############################################################################

df_stomata_density = pd.read_csv(main_path / 'Yang_data/stomata density.csv')
df_stomata_size = pd.read_csv(main_path / 'Yang_data/stomata measurements.csv')

df_stomata_density.loc[df_stomata_density['genotype'] == 'Col-0', 'genotype'] = 'WT'
df_stomata_size.loc[df_stomata_size['genotype'] == 'Col-0', 'genotype'] = 'WT'

# Plot Stomata Size
plot_with_mixedmodel_tukey(
    df=df_stomata_size,
    value_col='area',
    ylabel=r"Stomata size ($\mu m^2$)",
    save_name="stomata_size_stats_tukey.pdf",
    scaling=1.0
)

# Plot Stomata Density
plot_with_mixedmodel_tukey(
    df=df_stomata_density,
    value_col='stomata_density',
    ylabel=r"Stomata density (mm$^{-2}$)",
    save_name="stomata_density_stats_tukey.pdf",
    scaling=1e6
)
