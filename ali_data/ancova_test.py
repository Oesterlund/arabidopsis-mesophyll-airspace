#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 20:14:20 2025

@author: isabella
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load data
data = pd.read_csv('/home/isabella/Documents/PLEN/x-ray/ali_data/data_for_isabella.csv')

# Filter out row(s) where Ca_setpoint is '410.r'
#data = data[data['Ca_setpoint'] != '410.r']
data['Ca_setpoint_clean'] = data['Ca_setpoint'].replace({'410.r': '410'})
data.drop(columns='Ca_setpoint', inplace=True)

# Convert genotype to a categorical variable
data['genotype'] = data['genotype'].astype('category')

# ANCOVA model 1: A ~ Ca + genotype
model1 = smf.ols('A ~ Ca + genotype', data=data).fit()
anova_results1 = sm.stats.anova_lm(model1, typ=3)
print("ANCOVA model 1 (A ~ Ca + genotype):")
print(anova_results1)

# Post hoc test (Tukey) for genotype
tukey1 = pairwise_tukeyhsd(endog=data['A'], groups=data['genotype'], alpha=0.05)
print("\nPost hoc Tukey HSD (A ~ genotype):")
print(tukey1.summary())

# ANCOVA model 2: gm ~ Ci + genotype
model2 = smf.ols('gm ~ Ci + genotype', data=data).fit()
anova_results2 = sm.stats.anova_lm(model2, typ=3)
print("\nANCOVA model 2 (gm ~ Ci + genotype):")
print(anova_results2)

# Post hoc test (Tukey) for genotype in model 2
tukey2 = pairwise_tukeyhsd(endog=data['gm'], groups=data['genotype'], alpha=0.05)
print("\nPost hoc Tukey HSD (gm ~ genotype):")
print(tukey2.summary())
