import os
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

os.chdir('J:/NUTZER/GomezOspina.M/Paper_kidev/Minimalbeispiel/')
well_list = pd.read_csv("./cnn_paper_add/wells_cnn.txt", header=None)

gwmap_path= gpd.read_file("./cnn_paper_add/shp/GWF2.shp") 
gwshp= gwmap_path[['MEST_ID', 'UTM_X', 'UTM_Y', 'geometry']]
years_for_testing_list = [3, 4, 5]  # Testing years

# Create a new DataFrame to store results for each well and year
results = pd.DataFrame()

for years_for_testing in years_for_testing_list:
    for i in range(len(well_list)):
        wellid = well_list.iloc[i][0]

        lminrow = []
        for ini in range(10):
            path_tvscores = f'./cnn_paper/CNN_results_t{years_for_testing}/trainednets/{wellid}/ini{ini}/'
            
            # Check if the directory exists to avoid errors
            if os.path.exists(path_tvscores):
                try:
                    trmetric = pd.read_csv(path_tvscores + f'history{wellid}ini{ini}.csv')
                    min_index = trmetric['val_mse'].idxmin()
                    row_with_min = trmetric.loc[min_index]

                    lminrow.append(row_with_min)
                except FileNotFoundError:
                    print(f"File not found: {path_tvscores}history{wellid}ini{ini}.csv")
                    continue
            else:
                print(f"Path does not exist: {path_tvscores}")
                continue

        if lminrow:  # Proceed if lminrow is not empty
            minrows = pd.DataFrame(lminrow)
            mean_values = minrows.mean()

            try:
                test_scores = pd.read_csv(f'./cnn_paper/CNN_results_t{years_for_testing}/scores_{wellid}.txt')
                test_scores = test_scores.rename(columns={'R2': 'r2', 'RMSE': 'rmse'})

                # Create a temporary DataFrame to store the results for the current well and year
                temp_df = pd.DataFrame({
                    'MEST_ID': [wellid],
                    'year': [years_for_testing],
                    'rmse': [mean_values['rmse']],
                    'val_rmse': [mean_values['val_rmse']],
                    'rmse_test': [test_scores['rmse'].values[0]]
                })

                # Append the temporary DataFrame to the results DataFrame
                results = pd.concat([results, temp_df], ignore_index=True)

            except FileNotFoundError:
                print(f"Test scores file not found for well ID: {wellid}")

# Calculate the 'val_test' column
results['val_test'] = abs(results['rmse_test'] - results['val_rmse'])

# Mapping code remains unchanged
pathshp = r'J:\NUTZER\GomezOspina.M\Paper_kidev\data\shp/'
waterbodies = gpd.read_file(pathshp + "waterbodiesND.shp")
waterways = gpd.read_file(pathshp + "waterwaysND.shp")
germany_states = gpd.read_file(pathshp + "DEU_adm1.shp")
ND = germany_states[germany_states.NAME_1 == "Niedersachsen"]
citiesND = gpd.read_file(pathshp + "citiesND2.shp")

proj_coor = 4647
waterbodies = waterbodies.to_crs(epsg=proj_coor)
waterways = waterways.to_crs(epsg=proj_coor)
cities = citiesND.to_crs(epsg=proj_coor)
germany_states = germany_states.to_crs(epsg=proj_coor)
ND = ND.to_crs(epsg=proj_coor)

sns.set_theme(style="ticks")

# Ensure 'gwshp' DataFrame has correct projection
gwshp = gwshp.to_crs(epsg=proj_coor)

# Merge 'gwshp' with 'results' DataFrame on 'MEST_ID'
merged_gwshp = gwshp.merge(results, on='MEST_ID')

cmap = LinearSegmentedColormap.from_list('mycmap', [(0, '#009ad8'), (0.7, '#d8dadc'), (1, '#d35555')])
norm = Normalize(vmin=0, vmax=0.3)

fig, gw = plt.subplots(1, 1, figsize=(10, 10))
gw = merged_gwshp.plot(ax=gw, figsize=(10, 10), column='val_test', cmap=cmap, norm=norm, legend=False, marker="v", facecolor=None, zorder=3)
NS = ND.boundary.plot(ax=gw, alpha=0.3, edgecolor='k', linewidth=1, zorder=1)
wb = waterbodies.plot(ax=gw, alpha=0.8, color='b', linewidth=0.8, zorder=1)
ww = waterways.plot(ax=gw, alpha=0.3, color='b', linewidth=.5, zorder=2)
cit = cities.plot(ax=gw, alpha=1, color='k', markersize=5, zorder=2)
cbar = gw.get_figure().colorbar(gw.get_children()[0], ticks=None, orientation='horizontal', pad=0.02, aspect=20, shrink=0.4)
cbar.set_label('Val rmse - Test rmse', fontsize=14)

for x, y, label in zip(cities.geometry.x, cities.geometry.y, cities.name):
    NS.annotate(label, xy=(x, y), xytext=(-15, -10), textcoords="offset points", fontsize=10, color="k", alpha=0.9)

#for x, y, label in zip(merged_gwshp.geometry.x, merged_gwshp.geometry.y, merged_gwshp['MEST_ID']):
#    gw.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", fontsize=8, color="black", alpha=0.7)


gw.spines['top'].set_visible(False)
gw.spines['right'].set_visible(False)
gw.spines['bottom'].set_visible(False)
gw.spines['left'].set_visible(False)
gw.get_xaxis().set_ticks([])
gw.get_yaxis().set_ticks([])
#plt.savefig(r"J:\NUTZER\GomezOspina.M\Paper_kidev\Minimalbeispiel\cnn_paper_add/map_val_test.jpg",bbox_inches="tight",dpi=200)
plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
sns.set_theme(style="white")

# Define distinct colors for each year
colors = sns.color_palette("Set2", len(years_for_testing_list))

for i, years_for_testing in enumerate(years_for_testing_list):
    # Subset data for the current year
    subset = results[results['year'] == years_for_testing]
    val_test_values = subset['val_test']
    num_wells = subset['MEST_ID'].nunique()
    
    # Plot the KDE scaled by the number of unique wells
    sns.kdeplot(
        data=val_test_values, 
        ax=ax, 
        label=f'{years_for_testing}', 
        color=colors[i],
        bw_adjust=1,  # Bandwidth adjustment for smoother curve
        alpha=1,  # Full opacity for line
    )

# Set x-axis limits to zoom
ax.set_xlim(-.5, 1.5)

# Increase font size
ax.set_xlabel('RMSE Validation - Testing', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.legend(title='Number of\n testing years', fontsize=12, title_fontsize=14)
ax.grid(alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Set tick parameters
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.show()