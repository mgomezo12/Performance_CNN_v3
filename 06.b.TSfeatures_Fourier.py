import pandas as pd
import numpy as np
from scipy import stats

path= "J:/NUTZER/GomezOspina.M/Paper_kidev/reviews/"
dfsw=pd.read_pickle(path+"//gwsinmodel2.pkl") #sin model simulation results 
gwdata=pd.read_pickle(path+"//GWfilldatamod2_seas.pkl")
dfr = pd.read_pickle(path+"//dfresults.pkl")
dfr.reset_index(inplace=True)

merged_df = pd.merge(gwdata, dfr, on='wellid')

gwdata['ffpower']=np.nan

for wl in range(len(gwdata)):
        
    ts = gwdata.GW_NN[wl][gwdata.GW_NN[wl].columns[-1]]
    # Compute the FFT
    fft_result = np.fft.fft(ts.values)
    fft_freq = np.fft.fftfreq(len(ts), d=1)

    # Keep only positive frequencies (since FFT is symmetric)
    positive_freq_mask = fft_freq > 0
    fft_result = fft_result[positive_freq_mask]
    fft_freq = fft_freq[positive_freq_mask]

    # Compute the power spectrum
    power_spectrum = np.abs(fft_result)**2
    index_one_year = np.argmin(np.abs(fft_freq - 1/12))
    gwdata['ffpower'][wl] =  power_spectrum[index_one_year]

sort_ffp=gwdata.sort_values(by=['ffpower'])['wellid']
sort_dfr=dfr.sort_values(by=['NSE'])['wellid']
 
merged_df2 = pd.merge(gwdata, dfr, on='wellid')

merged_df2.to_pickle(path+"gw_features.pkl")

performance=['NSE','r2','Bias']
for p in performance: 
    slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df2['ffpower'][:len(merged_df2)-10], np.abs(merged_df2[p][:len(merged_df2)-10]))
    print(slope, intercept, r_value, p_value, std_err)
#vlrvalue.append(round(r_value,2)) if p_value <= 0.1 else vlrvalue.append(0) 