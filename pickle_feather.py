import pandas as pd


path= "J:/NUTZER/GomezOspina.M/Paper_kidev/reviews/"
path2= "J:/NUTZER/GomezOspina.M/Paper_kidev/data/"
gwdata=pd.read_pickle(path+"//GWfilldatamod2_seas.pkl")

df_list = []

for index, row in gwdata.iterrows():
    wellid = row['wellid']
    data = row['GW_NN']

    ts_df = pd.DataFrame({
        'wellid': wellid,
        'DATUM': data['DATUM'],
        'GWL': data[data.columns[-1]], 
        'filledtype': row['filledtype'],
        'fillratio': row['fillratio']
    })
    
    df_list.append(ts_df)


concatenated_df = pd.concat(df_list, ignore_index=True)
concatenated_df.to_feather(path2 + "GWfilldatamod.feather")


pr=pd.read_pickle(path2+"/dataprt.pkl")
tm=pd.read_pickle(path2+"/datatmt.pkl")

def process_data(data, value_col, agg_func):
    # Convert IDs and time once
    data['wellid'] = data['ID']
    df_list=[]
    for index, row in data.iterrows():
        wellid = row['wellid']
        data = pd.DataFrame({
            'wellid': int(wellid),
            'DATUM': pd.to_datetime( row['time']),
            value_col: row['cdata']
        })

        datam=data.resample("M", on="DATUM").agg({ value_col: agg_func, 'wellid': 'first'}).reset_index()
    
        df_list.append(datam) 

    concatenated_df = pd.concat(df_list, ignore_index=True)

    return concatenated_df


concatenated_dfpr = process_data(pr, 'pr', 'sum')
concatenated_dftm = process_data(tm, 'tm', 'mean')

concatenated_dfpr.to_feather(path2 + "pr.feather")
concatenated_dftm.to_feather(path2 + "tm.feather")


wellid = concatenated_df.unique()
unique_wellid_df = pd.DataFrame(wellid, columns=['wellid'])
unique_wellid_df.to_csv('wells_cnn.txt', index=False, header=False)