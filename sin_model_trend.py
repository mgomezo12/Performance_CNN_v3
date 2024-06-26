import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
#from tsfeatures import tsfeatures
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from scipy import stats
from sklearn.linear_model import LinearRegression

""" Code for model GWL time series in Lower Saxony adjusting a sinus function plus the 
last 9 months precipitation trend, we apply the same optimization  method and function #
(maximizing the sum of NSE and R2) as in the CNN model """


path_feather= "J:/NUTZER/GomezOspina.M/Paper_kidev/data/"
gwdata= pd.read_feather(path_feather+"GWfilldatamod.feather")

pr= pd.read_feather(path_feather+"pr.feather")

gwdata['DATUM'] = gwdata['DATUM'].dt.to_period('M')
pr['DATUM'] = pr['DATUM'].dt.to_period('M')
merged_df = pd.merge(gwdata[['wellid','DATUM','GWL']], pr[['wellid','DATUM','pr']], on=['DATUM', 'wellid'], how='inner')

def compute_9_month_trend(df,index):
    if index >= 8: 
        # Select the range of data for the last 9 months
        start_index = index - 8
        end_index = index
        X = df.iloc[start_index:end_index + 1]['DATUMp_ordinal'].values.reshape(-1, 1)
        y = df.iloc[start_index:end_index + 1]['pr'].values
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0]  
    else:
        return 0
    

def sinus_function(x, amplitude, frequency, phase, offset):
    return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset 


def sin_bayopt( amplitude, frequency, phase):
    dfs = merged_df[merged_df['wellid']== wellid]
    df = dfs.copy()
    cutoff_period = pd.Period('2012-01', freq='M')
    df=df[df['DATUM']<cutoff_period]
    df['datenum'] = pd.factorize(df['DATUM'])[0] + 1

    x = df["datenum"].values
    scaler_gwl = MinMaxScaler(feature_range=(-1, 1))
    yn = scaler_gwl.fit_transform(df['GWL'].values.reshape(-1, 1))
    y = np.array(yn).flatten()

    df['DATUMp'] = df['DATUM'].apply(lambda x: x.start_time)
    df['DATUMp_ordinal'] = df['DATUMp'].apply(lambda x: x.toordinal())

    df['9_month_trend'] = np.nan  
    for i in range(len(df)):
        df.at[df.index[i], '9_month_trend'] = compute_9_month_trend(df,i) 

    sim=sinus_function(x, amplitude, frequency, phase, df['9_month_trend'].values)
    err=sim-y
    meanobs=np.mean(y)
    errnse= y - meanobs
    r=stats.linregress( sim , y)

    mse = -np.mean(err**2) 

    return mse  

def simulate_all( amplitude, frequency, phase):
    dfs = merged_df[merged_df['wellid']== wellid]
    df = dfs.copy()

    cutoff_period = pd.Period('2012-01', freq='M')
    df=df[df['DATUM']>cutoff_period]
    df['datenum'] = pd.factorize(df['DATUM'])[0] + 1

    x = df["datenum"].values
    scaler_gwl = MinMaxScaler(feature_range=(-1, 1))
    yn = scaler_gwl.fit_transform(df['GWL'].values.reshape(-1, 1))
    y = np.array(yn).flatten()

    df['DATUMp'] = df['DATUM'].apply(lambda x: x.start_time)
    df['DATUMp_ordinal'] = df['DATUMp'].apply(lambda x: x.toordinal())

    df['9_month_trend'] = np.nan  
    for i in range(len(df)):
        df.at[df.index[i], '9_month_trend'] = compute_9_month_trend(df,i) 

    sim=sinus_function(x, amplitude, frequency, phase, df['9_month_trend'].values)
    err=sim-y
    err_rel= (sim-y) / (np.max(y)-np.min(y))
    meanobs=np.mean(y)
    errnse= y - meanobs
    r=stats.linregress( sim , y)

    NSE= 1- ((np.sum(err ** 2)) / (np.sum((errnse) ** 2)))
    R2= r.rvalue**2
    RMSE =  np.sqrt(np.mean(err ** 2))
    rRMSE = np.sqrt(np.mean(err_rel ** 2)) * 100
    Bias = np.mean(err)
    rBias = np.mean(err_rel) * 100    

    scores = pd.DataFrame(np.array([[NSE, R2, RMSE, rRMSE, Bias, rBias]]),
                   columns=['NSE','R2','RMSE','rRMSE','Bias','rBias'])

    return scores, y , x, df['DATUM'].values, sim, dfs.columns[-1][6:]

#Only test period
def simulate( amplitude, frequency, phase):
    dfs = merged_df[merged_df['wellid']== wellid]
    df = dfs.copy()

    cutoff_period = pd.Period('2012-01', freq='M')
    df=df[df['DATUM']>cutoff_period]
    df['datenum'] = pd.factorize(df['DATUM'])[0] + 1

    x = df["datenum"].values
    scaler_gwl = MinMaxScaler(feature_range=(-1, 1))
    yn = scaler_gwl.fit_transform(df['GWL'].values.reshape(-1, 1))
    y = np.array(yn).flatten()

    df['DATUMp'] = df['DATUM'].apply(lambda x: x.start_time)
    df['DATUMp_ordinal'] = df['DATUMp'].apply(lambda x: x.toordinal())

    df['9_month_trend'] = np.nan  
    for i in range(len(df)):
        df.at[df.index[i], '9_month_trend'] = compute_9_month_trend(df,i) 

    sim=sinus_function(x, amplitude, frequency, phase, df['9_month_trend'].values)
    err=sim-y
    err_rel= (sim-y) / (np.max(y)-np.min(y))
    meanobs=np.mean(y)
    errnse= y - meanobs
    r=stats.linregress( sim , y)

    NSE= 1- ((np.sum(err ** 2)) / (np.sum((errnse) ** 2)))
    R2= r.rvalue**2
    RMSE =  np.sqrt(np.mean(err ** 2))
    rRMSE = np.sqrt(np.mean(err_rel ** 2)) * 100
    Bias = np.mean(err)
    rBias = np.mean(err_rel) * 100    

    scores = pd.DataFrame(np.array([[NSE, R2, RMSE, rRMSE, Bias, rBias]]),
                   columns=['NSE','R2','RMSE','rRMSE','Bias','rBias'])

    return scores, y , x, df['DATUM'].values, sim, dfs.columns[-1][6:]


#%%

lwdf , lwellid, lscores  = [] , [] , []

wellid= [9700232, 40510681, 100000718, 100000926 ]

for pp in range(len(merged_df['wellid'].unique())):
#for pp in range(2):
    wellid= merged_df['wellid'].unique()[pp]
    # Define the search space for Bayesian optimization
    param_space = { 
        "amplitude": (0.4,1),
        "frequency": (1/14,1/10),
        "phase": (0, 2 * np.pi)
    }

    # Create Bayesian optimization optimizer
    optimizer = BayesianOptimization(
        f = sin_bayopt,
        pbounds = param_space,
        random_state=1, 
        verbose = 0
    )

    ei_function = UtilityFunction(kind='ei')
    optimizer.maximize(
            init_points=20, #steps of random exploration (random starting points before bayesopt(?))
            n_iter=0, # steps of bayesian optimization
            acquisition_function=ei_function
            )
    counter1 = 150
    counter2 = 15
        
    # optimize while improvement during last 10 steps
    current_step = len(optimizer.res)
    beststep = False
    step = -1
    while not beststep:
        step = step + 1
        beststep = optimizer.res[step] == optimizer.max #search for best iteration


    while current_step < counter1:
            current_step = len(optimizer.res)
            beststep = False
            step = -1
            while not beststep:
                step = step + 1
                beststep = optimizer.res[step] == optimizer.max
            print("\nbeststep {}, current step {}".format(step+1, current_step+1))
            optimizer.maximize(
                init_points=0, #steps of random exploration (random starting points before bayesopt(?))
                n_iter=1, # steps of bayesian optimization

                )



    amplitude_best = optimizer.max.get("params").get("amplitude")
    frequency_best = optimizer.max.get("params").get("frequency")
    phase_best = optimizer.max.get("params").get("phase")

    wellid=merged_df['wellid'].unique()[pp]
    scoresopt, yobsopt, xdatesopt , datesopt, ysimopt, wellidopt = simulate_all( amplitude_best, frequency_best, phase_best)
    scores, yobs, xdates , dates, ysim, wellid = simulate( amplitude_best, frequency_best, phase_best)


    dates2 = [p.start_time for p in dates]
     
    plt.figure(figsize=(19, 6))
    line_obs, = plt.plot( dates2, yobs, color='slateblue', label='Observed', linewidth = 1.7,alpha=0.9)
    line_sim, = plt.plot(dates2, ysim,  'r', label='Fitted sin model',linewidth = 1.7)

    plt.legend(fontsize=13,bbox_to_anchor=(.14, .25),loc='upper right',fancybox = False)
    line_obs.set_label('_nolegend_')
    line_sim.set_label('_nolegend_')

    plt.xlabel('Date',size=16)
    plt.ylabel('GWL [masl]', size=15)
    plt.grid( which='major', color='grey', alpha = 0.3, linestyle='-')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    text_box = plt.text(0.895, 0.92, f'R$^2 = {scores.R2.values[0]:.2f}$    NSE = {scores.NSE.values[0]:.2f}',
                    ha='center', va='center', transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'), fontsize=15)
    wellid=merged_df['wellid'].unique()[pp]
    plt.savefig(r"J:/NUTZER/GomezOspina.M/Paper_kidev/Minimalbeispiel/cnn_paper_add/figs_sin/"+str(wellid)+"_test_period_SIN.pdf")

    lwdf.append( pd.DataFrame( {"dates":datesopt, "obs":yobsopt, "sim":ysimopt} ) )
    lwellid.append(wellid)
    lscores.append(scores)
    #plt.show()

    dfsw=pd.DataFrame({"wellid":lwellid, "scores": lscores, "lwdf":lwdf})


dfsw['NSE'] = np.full(len(dfsw), np.nan)
dfsw['R2'] = np.full(len(dfsw), np.nan)
dfsw['Bias'] = np.full(len(dfsw), np.nan)
for i in range(len(dfsw)): 
     dfsw.at[i, 'NSE'] = dfsw.scores[i].NSE
     dfsw.at[i, 'R2'] = dfsw.scores[i].R2
     dfsw.at[i, 'Bias'] = dfsw.scores[i].Bias

dfsw.to_pickle(r"J:\NUTZER\GomezOspina.M\Paper_kidev\Minimalbeispiel\cnn_paper_add/"+"//gwsinmodel.pkl")


