import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import r2_score
from scipy import stats

os.chdir('J:/NUTZER/GomezOspina.M/Paper_kidev/')


gwdata = pd.read_feather("./data/GWfilldatamod.feather")
dfr = pd.read_pickle("./Minimalbeispiel/cnn_paper_add/dfresults.pkl")

# Convert 'DATUM' to datetime if not already done
gwdata['DATUM'] = pd.to_datetime(gwdata['DATUM'])

# Extract year and month from 'DATUM'
gwdata['year'] = gwdata['DATUM'].dt.year
gwdata['month'] = gwdata['DATUM'].dt.month


# Create a dictionary to store the mean of each observation
mean_values = gwdata.groupby('wellid')['GWL'].mean().to_dict()
gwdata['GWL_centered'] = gwdata.apply(lambda row: row['GWL'] - mean_values[row['wellid']], axis=1)


simr_folder = "./reviews/resultsCNN/wihtoutRH/"
new_rows = []
for wellid in dfr['wellid'].unique():
    sim_df = pd.read_csv(simr_folder + f"ensemble_mean_values_CNN_{wellid}.txt", sep=';')
    sim_df['DATUM'] = pd.to_datetime(sim_df['dates'])
    sim_df['year'] = sim_df['DATUM'].dt.year
    sim_df['month'] = sim_df['DATUM'].dt.month
    
    sim_df['GWL'] = sim_df['Sim'] - sim_df['Sim'].mean()
    sim_df['wellid'] = wellid
    new_rows.append(sim_df)

new_simulated_df = pd.concat(new_rows, ignore_index=True)

# Filter relevant columns
sim_df = new_simulated_df[['wellid', 'year', 'month', 'GWL']]

# Merge simulated and observed data based on wellid, year, and month
merged_df = pd.merge(gwdata, sim_df, on=['wellid', 'year', 'month'], suffixes=('_observed', '_simulated'))


r = stats.linregress(merged_df['GWL_centered'], merged_df['GWL_simulated'])
R2 = r.rvalue ** 2

merged_df[['GWL_centered', 'GWL_simulated']].corr(method='pearson')**2

r2_score(merged_df['GWL_centered'], merged_df['GWL_simulated'])

plt.figure(figsize=(8, 8))
plt.scatter(merged_df['GWL_centered'], merged_df['GWL_simulated'], alpha=0.5)
plt.plot([merged_df['GWL_centered'].min(), merged_df['GWL_centered'].max()], 
         [merged_df['GWL_centered'].min(), merged_df['GWL_centered'].max()], 
         color='red', linestyle='--')

plt.xlabel('Mean-centered Observed GWL', fontsize=14)
plt.ylabel('Mean-centered Simulated GWL', fontsize=14)

# Add R² value text on the plot
plt.text(0.05, 0.95, f'$R^2 = {R2:.2f}$', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')

# Customize grid and spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.2)

# Increase the font size of the ticks
ax.tick_params(axis='both', which='major', labelsize=12)

plt.show()
