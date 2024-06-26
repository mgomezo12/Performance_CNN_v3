# -*- coding: utf-8 -*-
"""
Adjusted by Mariana Gomez (BGR)

Original credits: 
Created on Sun Nov 15 10:59:14 2020
@author: Andreas Wunsch (KIT)

"""
#reproducability
from numpy.random import seed
seed(1+347823)
from tensorflow import random
random.set_seed(1+63493)

#import packages
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
from bayes_opt.event import Events
import os
import pandas as pd
import datetime
from scipy import stats
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import backend as K

os.chdir('J:/NUTZER/GomezOspina.M/Paper_kidev/Minimalbeispiel/')

gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
                                         

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

#%% functions
def load_data(Well_ID):
    
    path_feather= "J:/NUTZER/GomezOspina.M/Paper_kidev/data/"
    gwl= pd.read_feather(path_feather+"GWfilldatamod.feather")

    pr= pd.read_feather(path_feather+"pr.feather")
    tm= pd.read_feather(path_feather+"tm.feather")
    meteo = pd.merge(pr, tm, on=['wellid', 'DATUM'])

    Well_ID=int(Well_ID)
    dgwl=gwl[gwl['wellid']== Well_ID].copy()
    dmet=meteo[meteo['wellid']== Well_ID].copy()


    dmet.loc[:, 'YearMonth'] = dmet['DATUM'].dt.to_period('M')
    dgwl.loc[:, 'YearMonth'] = dgwl['DATUM'].dt.to_period('M')

    dataset = pd.merge(dgwl, dmet, on=['wellid', 'YearMonth'],how='left',suffixes=('', '_met'))
    dataset.dropna(subset=['pr'], inplace=True)

    dldataset=pd.DataFrame({'dates': dataset['DATUM'], 
                            'GWL' : dataset['GWL'], 
                            'pr': dataset['pr'], 
                            'tm' :dataset['tm'] })
    dataset=dldataset.set_index('dates')

    
    return  dataset 


def split_data(data, GLOBAL_SETTINGS):
    dataset = data[(data.index < GLOBAL_SETTINGS["test_start"])] #Testdaten abtrennen
    
    TrainingData = dataset[0:round(0.8 * len(dataset))]
    StopData = dataset[round(0.8 * len(dataset))+1:round(0.9 * len(dataset))]#split data
    StopData_ext = dataset[round(0.8 * len(dataset))+1-GLOBAL_SETTINGS["seq_length"]:round(0.9 * len(dataset))] #extend data according to dealys/sequence length
    OptData = dataset[round(0.9 * len(dataset))+1:]#split data
    OptData_ext = dataset[round(0.9 * len(dataset))+1-GLOBAL_SETTINGS["seq_length"]:] #extend data according to dealys/sequence length
    
    TestData = data[(data.index >= GLOBAL_SETTINGS["test_start"]) & (data.index <= GLOBAL_SETTINGS["test_end"])] #split data
    TestData_ext = pd.concat([dataset.iloc[-GLOBAL_SETTINGS["seq_length"]:], TestData], axis=0) # extend Testdata to be able to fill sequence later                                              

    return TrainingData, StopData, StopData_ext, OptData, OptData_ext, TestData, TestData_ext

def to_supervised(data, GLOBAL_SETTINGS):
    X, Y = list(), list()
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # find the end of this pattern
        end_idx = i + GLOBAL_SETTINGS["seq_length"]
        # check if we are beyond the dataset
        if end_idx >= len(data):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = data[i:end_idx, 1:], data[end_idx, 0]
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)

class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
    
def predict_distribution(X, model, n):
    # preds = [model(X, training=True) for _ in range(n)]
    preds = [model(X) for _ in range(n)]
    return np.hstack(preds)

def gwmodel(ini,GLOBAL_SETTINGS,X_train, Y_train,X_stop, Y_stop):
    # define model
    seed(ini+872527)
    tf.random.set_seed(ini+87747)
    inp = tf.keras.Input(shape=(GLOBAL_SETTINGS["seq_length"], X_train.shape[2]))
    cnn = tf.keras.layers.Conv1D(filters=GLOBAL_SETTINGS["filters"],
                                         kernel_size=3,
                                         activation='relu',
                                         padding='same')(inp)
    
    if GLOBAL_SETTINGS["batchnorm"]:
        cnn = tf.keras.layers.BatchNormalization()(cnn)
        
    cnn = tf.keras.layers.MaxPool1D(padding='same')(cnn)
    cnn = tf.keras.layers.Dropout(GLOBAL_SETTINGS["dropout_rate"])(cnn)
    
    cnn = tf.keras.layers.Flatten()(cnn)
    
    if GLOBAL_SETTINGS["dense_size"] > 0:
        cnn = tf.keras.layers.Dense(GLOBAL_SETTINGS["dense_size"], activation='relu')(cnn)
        
    output1 = tf.keras.layers.Dense(1, activation='linear')(cnn)
    
    # tie together
    model = tf.keras.Model(inputs=inp, outputs=output1)
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=GLOBAL_SETTINGS["learning_rate"], epsilon=10E-3, clipnorm=GLOBAL_SETTINGS["clip_norm"])
    
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse','mae',rmse,r2])
    
    
    # early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=GLOBAL_SETTINGS["patience"],restore_best_weights = True)
    
    # fit network
    history = model.fit(X_train, Y_train, validation_data=(X_stop, Y_stop),
                        epochs=GLOBAL_SETTINGS["epochs"], verbose=0,
                        batch_size=GLOBAL_SETTINGS["batch_size"], callbacks=[es])
        
    return model, history

class newJSONLogger(JSONLogger) :

      def __init__(self, path):
            self._path=None
            super(JSONLogger, self).__init__()
            self._path = path if path[-5:] == ".json" else path + ".json"
            
def bayesOpt_function(seqlength,densesize,batchsize,filters,dropout_rate,batchnorm):
    
    seqlength = int(seqlength)
    
    if int(densesize) == 3:
        densesize = 0
    else:
        densesize = 2**int(densesize)
        
    batchsize = 2**int(batchsize)
    filters = 2**int(filters)
    
    dropout_rate = 0.1*int(dropout_rate)

    batchnorm = int(round(batchnorm))


    return bayesOpt_function_discrete(seqlength,densesize,batchsize,filters,dropout_rate,batchnorm)

def bayesOpt_function_discrete(seqlength,densesize,batchsize,filters,dropout_rate,batchnorm):
    
    print("...optimizing: "+Well_ID)
    
    GLOBAL_SETTINGS = {
        'batch_size': batchsize, #16-128
        'dense_size': densesize, 
        'filters': filters, 
        'seq_length': seqlength,
        'batchnorm': batchnorm,
        'clip_norm': True,
        'epochs': 100,
        'patience': 15,
        'dropout_rate': dropout_rate,
        'learning_rate': 1e-3,
        'test_years': years_for_testing,
        'test_end': pd.to_datetime('30122015', format='%d%m%Y')
    }

    ## load data
    data = load_data(Well_ID)
    
    if GLOBAL_SETTINGS["test_end"] > data.index[-1]:
        GLOBAL_SETTINGS["test_end"] = data.index[-1]

    GLOBAL_SETTINGS["test_start"] = GLOBAL_SETTINGS["test_end"] - datetime.timedelta(days=(365*GLOBAL_SETTINGS["test_years"]))
        
    #scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler = StandardScaler()
    scaler_gwl = MinMaxScaler(feature_range=(-1, 1))
    scaler_gwl.fit(pd.DataFrame(data['GWL']))
    data_n = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

    #split data
    TrainingData, StopData, StopData_ext, OptData, OptData_ext, TestData, TestData_ext = split_data(data, GLOBAL_SETTINGS)
    TrainingData_n, StopData_n, StopData_ext_n, OptData_n, OptData_ext_n, TestData_n, TestData_ext_n = split_data(data_n, GLOBAL_SETTINGS)
    
    #sequence data
    X_train, Y_train = to_supervised(TrainingData_n.values, GLOBAL_SETTINGS)
    X_stop, Y_stop = to_supervised(StopData_ext_n.values, GLOBAL_SETTINGS) 
    X_opt, Y_opt = to_supervised(OptData_ext_n.values, GLOBAL_SETTINGS)
    # X_test, Y_test = to_supervised(TestData_ext_n.values, GLOBAL_SETTINGS) 

    #build and train model with different initializations
    inimax = 2
    n_distr = 20 # no of MCDropout predictions
    sim_members = np.zeros((len(X_opt), inimax*n_distr))
    sim_members[:] = np.nan
    
    for ini in range(inimax):
        # print("optimizing: "+Well_ID+" - ini: "+str(ini))
        model, history = gwmodel(ini,GLOBAL_SETTINGS,X_train, Y_train, X_stop, Y_stop)  

        print("(training loss:{}, validation loss:{}) BayesOpt-Iteration {} - ini-Ensemblemember {}".format(history.history['loss'], history.history['val_loss'], len(optimizer.res)+1, ini+1))

        y_pred_distribution = predict_distribution(X_opt, model, n_distr)
        sim = scaler_gwl.inverse_transform(y_pred_distribution)
        sim_members[:,ini*n_distr:ini*n_distr+n_distr]=sim
        
    
    sim_mean = np.nanmedian(sim_members,axis = 1)

    # get scores
    sim = np.asarray(sim_mean.reshape(-1,1))
    obs = np.asarray(scaler_gwl.inverse_transform(Y_opt.reshape(-1,1)))

    err = sim-obs
    # err_nash = obs - np.mean(np.asarray(data['GWL'][(data.index < GLOBAL_SETTINGS["test_start"])]))
    err_nash = obs - np.mean(obs)

    NSE = 1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2)))
    r = stats.linregress(sim[:,0], obs[:,0])
    R2 = r.rvalue ** 2
    # RMSE =  np.sqrt(np.mean(err ** 2))
    # Bias = np.mean(err)
    print("[NSE+R2] step result "+str(round(NSE+R2,4)))
    
    return NSE+R2

def simulate_testset(seqlength,densesize,batchsize,filters,dropout_rate,batchnorm):
    
    # fixed settings for all experiments (müssen gleich sein wie bei Optimierung)
    GLOBAL_SETTINGS = {
        'batch_size': batchsize, #16-128
        'dense_size': densesize, 
        'filters': filters, 
        'seq_length': seqlength,
        'batchnorm': batchnorm,
        'clip_norm': True,
        'epochs': 100,
        'patience': 15,
        'dropout_rate': dropout_rate,
        'learning_rate': 1e-3,
        'test_years': years_for_testing,
        'test_end': pd.to_datetime('30122015', format='%d%m%Y')
    }

    ## load data
    data = load_data(Well_ID)
    
    if GLOBAL_SETTINGS["test_end"] > data.index[-1]:
        GLOBAL_SETTINGS["test_end"] = data.index[-1]

    GLOBAL_SETTINGS["test_start"] = GLOBAL_SETTINGS["test_end"] - datetime.timedelta(days=(365*GLOBAL_SETTINGS["test_years"]))

        
        
    #scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler = StandardScaler()
    scaler_gwl = MinMaxScaler(feature_range=(-1, 1))
    scaler_gwl.fit(pd.DataFrame(data['GWL']))
    data_n = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

    #split data
    TrainingData, StopData, StopData_ext, OptData, OptData_ext, TestData, TestData_ext = split_data(data, GLOBAL_SETTINGS)
    TrainingData_n, StopData_n, StopData_ext_n, OptData_n, OptData_ext_n, TestData_n, TestData_ext_n = split_data(data_n, GLOBAL_SETTINGS)
    
    #sequence data
    X_train, Y_train = to_supervised(TrainingData_n.values, GLOBAL_SETTINGS)
    X_stop, Y_stop = to_supervised(StopData_ext_n.values, GLOBAL_SETTINGS) 
    X_opt, Y_opt = to_supervised(OptData_ext_n.values, GLOBAL_SETTINGS)
    X_test, Y_test = to_supervised(TestData_ext_n.values, GLOBAL_SETTINGS) 

    #build and train model with different initializations
    inimax = 10 # no of models
    n_distr = 100 # no of MCDropout predictions
    
    sim_members = np.zeros((len(X_test), inimax))
    sim_members[:] = np.nan
    
    sim_members_train = np.zeros((len(X_train), inimax))
    sim_members_train[:] = np.nan
    sim_members_stop = np.zeros((len(X_stop), inimax))
    sim_members_stop[:] = np.nan
    sim_members_opt = np.zeros((len(X_opt), inimax))
    sim_members_opt[:] = np.nan
    
    testresults_members = np.zeros((len(X_test), inimax*n_distr))

    for ini in range(inimax):
        print("testing: "+Well_ID+" - ini: "+str(ini))
        modelpath = "./trainednets/"+Well_ID+"/ini"+str(ini)
        if not os.path.isdir(modelpath):
            print("training: "+Well_ID+" - ini: "+str(ini))
            model, history = gwmodel(ini,GLOBAL_SETTINGS,X_train, Y_train, X_stop, Y_stop)              
            model.save(modelpath)
            pd.DataFrame(history.history).to_csv(modelpath + '/history'+Well_ID+'ini'+str(ini)+'.csv', index=False)
        else:
            print("loading: "+Well_ID+" - ini: "+str(ini))
            model = tf.keras.models.load_model(modelpath)  

        y_pred_distribution = predict_distribution(X_test, model, n_distr)
        test_sim = scaler_gwl.inverse_transform(y_pred_distribution)
        testresults_members[:,ini*n_distr:ini*n_distr+n_distr]=test_sim
        
        #just for simple plt of all sections (no MCDropout)
        y_pred = model.predict(X_test)
        sim = scaler_gwl.inverse_transform(y_pred)
        sim_members[:, ini]= sim.reshape(-1,)
        
        y_pred = model.predict(X_train)
        sim = scaler_gwl.inverse_transform(y_pred)
        sim_members_train[:, ini]= sim.reshape(-1,)
        
        y_pred = model.predict(X_stop)
        sim = scaler_gwl.inverse_transform(y_pred)
        sim_members_stop[:, ini]= sim.reshape(-1,)
        
        y_pred = model.predict(X_opt)
        sim = scaler_gwl.inverse_transform(y_pred)
        sim_members_opt[:, ini]= sim.reshape(-1,)


    sim_mean = np.mean(testresults_members,axis = 1)    
    sim = np.asarray(sim_mean.reshape(-1,1))
    sim_uncertainty = [np.quantile(testresults_members, 0.05, axis=1),np.quantile(testresults_members, 0.95, axis=1)]


    sim_train = np.nanmedian(sim_members_train,axis = 1)
    sim_stop = np.nanmedian(sim_members_stop,axis = 1)
    sim_opt = np.nanmedian(sim_members_opt,axis = 1)

    # get scores
    sim = np.asarray(sim_mean.reshape(-1,1))
    obs = np.asarray(scaler_gwl.inverse_transform(Y_test.reshape(-1,1)))

    err = sim-obs
    # err_nash = obs - np.mean(np.asarray(data['GWL'][(data.index < GLOBAL_SETTINGS["test_start"])]))
    err_nash = obs - np.mean(obs)

    NSE = 1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2)))
    r = stats.linregress(sim[:,0], obs[:,0])
    R2 = r.rvalue ** 2
    RMSE =  np.sqrt(np.mean(err ** 2))
    Bias = np.mean(err)
    scores = pd.DataFrame(np.array([[NSE, R2, RMSE, Bias]]),
                   columns=['NSE','R2','RMSE','Bias'])
    print(scores)

    return scores, TestData, sim, obs, inimax, sim_uncertainty, Well_ID, sim_train, sim_stop, sim_opt, TrainingData, StopData, OptData


#%% start

with tf.device("/cpu"): # change to cpu if no gpu available or problems with CUDA occur
    
    time1 = datetime.datetime.now()
    basedir = './'
    os.chdir(basedir)
    
    well_list = pd.read_csv("./wells_cnn.txt")
    years_for_testing = 4 # length og testset in years
    
    #for w in range(well_list.shape[0]): #loop all wells
    for index, row in well_list.iterrows():
        Well_ID=str(row.values[0])
        print("well={}  ".format(row.values[0]))
        if index>1:
            break 

        seed(1)
        tf.random.set_seed(1)
        
        skip = False # default setting
        
        #skip if already optimized and forecasted (existing "scores" file)
        if os.path.isfile('./scores_'+Well_ID+'.txt'):
            skip = True
            
        
    #%% Hyperparameter
        counter1 = 5 #min number of iterations
        counter2 = 5 #stop if > counter1 and if no improvement for counter2 steps
        counter3 = 30 #max number of iterations
        
        #values partly are surrogates for real values  (e.g. batchsize 8 == 2**8)
        pbounds = {'seqlength': (1, 12), 
                    'densesize': (3, 7),
                    'batchsize': (4, 8),
                    'filters': (4,8),
                    'dropout_rate':(1,1),
                    'batchnorm':(0,1)} 
        
        optimizer = BayesianOptimization(
            f= bayesOpt_function, #Funktion die optimiert wird
            pbounds=pbounds, #Wertebereiche in denen optimiert wird
            random_state=1, 
            verbose = 0 # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent, verbose = 2 prints everything
            )
    
        # #load existing optimize
        log_already_available = 0
        
        if os.path.isfile("./logs_CNN_"+Well_ID+".json"):
            load_logs(optimizer, logs=["./logs_CNN_"+Well_ID+".json"]);
            print("\nExisting optimizer is already aware of {} points. (w={})".format(len(optimizer.space),Well_ID))
            log_already_available = 1
            
        #%% Optimization
        # random exploration
        if (skip == False) & (log_already_available == 0):
            
            #Save progress
            logger = newJSONLogger(path="./logs_CNN_"+Well_ID+".json")
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            
            if log_already_available == 0:
                optimizer.maximize(
                        init_points=20, #steps of random exploration 
                        n_iter=0, # steps of bayesian optimization
                        acq="ei",# ei  = expected improvmenet (probably the most common acquisition function) 
                        xi=0.05  #  Prefer exploitation (xi=0.0) / Prefer exploration (xi=0.1)
                        )
    
                        
        if (skip == False):
    
            # optimize while improvement during last counter2 steps
            current_step = len(optimizer.res)
            beststep = False
            step = -1
            while not beststep:
                step = step + 1
                beststep = optimizer.res[step] == optimizer.max #aktuell beste Iteration suchen
        
            while current_step < counter1: #für <counter1 Interationen kein Abbruchskriterium
                    current_step = len(optimizer.res)
                    beststep = False
                    step = -1
                    while not beststep:
                        step = step + 1
                        beststep = optimizer.res[step] == optimizer.max
                    print("\nbeststep {}, current step {}".format(step+1, current_step+1))
                    optimizer.maximize(
                        init_points=0, #steps of random exploration (
                        n_iter=1, # steps of bayesian optimization
                        acq="ei",# ei  = expected improvmenet (probably the most common acquisition function) 
                        xi=0.05  #  Prefer exploitation (xi=0.0) / Prefer exploration (xi=0.1)
                        )
                    
            # while (step + counter2 > current_step and current_step < counter3): # Abbruch bei 150 Iterationen oder wenn seit 20 keine Verbesserung mehr eingetreten ist
            #         current_step = len(optimizer.res)
            #         beststep = False
            #         step = -1
            #         while not beststep:
            #             step = step + 1
            #             beststep = optimizer.res[step] == optimizer.max
                        
            #         print("\nbeststep {}, current step {}".format(step+1, current_step+1))
            #         optimizer.maximize(
            #             init_points=0, #steps of random exploration 
            #             n_iter=1, # steps of bayesian optimization
            #             acq="ei",# ei  = expected improvmenet (probably the most common acquisition function) 
            #             xi=0.05  #  Prefer exploitation (xi=0.0) / Prefer exploration (xi=0.1)
            #             )
    
                
            print("\nBEST:\t{}".format(optimizer.max))

    
        if skip:
            print("skipped")
    
    
    #%% testing
        if not skip :
                
            #get best values from optimizer
            seqlength = int(optimizer.max.get("params").get("seqlength"))
            
            densesize = optimizer.max.get("params").get("densesize")
            if int(densesize) == 3:
                densesize = 0
            else:
                densesize = 2**int(densesize)
                
            batchsize = 2**int(optimizer.max.get("params").get("batchsize"))
            filters = 2**int(optimizer.max.get("params").get("filters"))        
            
            dropout_rate = 0.1*int(optimizer.max.get("params").get("dropout_rate"))
            
            batchnorm = int(round(optimizer.max.get("params").get("batchnorm")))
            
            HPs = {'batch_size': batchsize,
                    'kernel_size': 3,
                    'dense_size': densesize, 
                    'filters': filters, 
                    'seq_length': seqlength,
                    'batchnorm': batchnorm,
                    'dropout_rate': dropout_rate
                    }
            
            pd.DataFrame(HPs, index = [0]).to_csv('./hyper_params_'+Well_ID+'.txt', sep=';', index=False)
        
            # start testing
            scores, TestData, sim, obs, inimax, sim_uncertainty, Well_ID, sim_train, sim_stop, sim_opt, TrainingData, StopData, OptData = simulate_testset(seqlength,densesize,batchsize,filters,dropout_rate,batchnorm)
            
        #%% plot test
            pyplot.figure(figsize=(20,6))
            
            lb = sim_uncertainty[0]
            ub = sim_uncertainty[1]
            
            pyplot.fill_between(TestData.index, lb,
                            ub, facecolor = (1,0.7,0,0.99),
                            label ='90% confidence',linewidth = 0.8,
                            edgecolor = (1,0.7,0,0.99))
            
            pyplot.plot(TestData.index, sim, color = 'r', label ="simulated mean", alpha=1,linewidth=1.2)
            
            pyplot.plot(TestData.index, obs, 'k', label ="observed", linewidth=1.2,alpha=1)
            
            pyplot.title("CNN Model Test: "+Well_ID, size=17,fontweight = 'bold')
            pyplot.ylabel('GWL [m asl]', size=15)
            pyplot.xlabel('Date',size=15)
            pyplot.legend(fontsize=15,bbox_to_anchor=(1.18, 1),loc='upper right',fancybox = False, framealpha = 1, edgecolor = 'k')
            pyplot.tight_layout()
            pyplot.grid(b=True, which='major', color='#666666', alpha = 0.3, linestyle='-')
            pyplot.xticks(fontsize=14)
            pyplot.yticks(fontsize=14)
            
            s = """NSE = {:.2f}\nR²  = {:.2f}\nRMSE = {:.2f}\nBias = {:.2f}""".format(scores.NSE[0],scores.R2[0],scores.RMSE[0],scores.Bias[0])
            
            pyplot.figtext(0.9, 0.5, s, bbox=dict(facecolor='white'),fontsize = 15)
            
            pyplot.rc("pdf", fonttype=42)
            pyplot.savefig('./Test_'+Well_ID+'_Graph0_CNN.pdf', bbox_inches='tight')            
            # pyplot.show()
            
            
            beststep = False
            step = -1
            while not beststep:
                step = step + 1
                beststep = optimizer.res[step] == optimizer.max 
                
            # print log summary file
            f = open('./log_summary_CNN_'+Well_ID+'.txt', "w")
            print("BEST:", file = f)
            print(HPs,file = f)
            print("\nbest iteration = {}\n".format(step+1), file = f)
            print("max iteration = {}\n".format(len(optimizer.res)), file = f)
            for i, res in enumerate(optimizer.res):
                print("Iteration {}: \t{}".format(i+1, res), file = f) 
            f.close()
            
            # print scores
            scores.to_csv('./scores_'+Well_ID+'.txt',float_format='%.3f')
            
            printdf = pd.DataFrame(data=np.c_[obs,sim,lb,ub],index=TestData.index)
            printdf = printdf.rename(columns={0: 'Obs', 1: 'Sim', 2:'lb:0.05', 3:'ub:95'})
            printdf.to_csv('./results_'+Well_ID+'.txt',sep=';', float_format = '%.6f')

            
            
            #%% plots
            pyplot.figure(figsize=(20,6))
        
            pyplot.plot(TrainingData.index[seqlength:], sim_train, 'b', label ="simulated mean", linewidth = 1.7)
            pyplot.plot(TrainingData.index, TrainingData['GWL'], 'k', label ="observed", linewidth=1.7,alpha=0.9)
        
            pyplot.plot(StopData.index, sim_stop, 'y', label ="simulated mean", linewidth = 1.7)
            pyplot.plot(StopData.index, StopData['GWL'], 'k', label ="observed", linewidth=1.7,alpha=0.9)
            
            pyplot.plot(OptData.index, sim_opt, 'g', label ="simulated mean", linewidth = 1.7)
            pyplot.plot(OptData.index, OptData['GWL'], 'k', label ="observed", linewidth=1.7,alpha=0.9)
            
            pyplot.plot(TestData.index, sim, 'r', label ="simulated mean", linewidth = 1.7)
            pyplot.plot(TestData.index, obs, 'k', label ="observed", linewidth=1.7,alpha=0.9)
            
            pyplot.title("CNN Model full: "+Well_ID, size=17,fontweight = 'bold')
            pyplot.ylabel('GWL [m asl]', size=15)
            pyplot.xlabel('Date',size=15)
            pyplot.legend(fontsize=15,bbox_to_anchor=(1.18, 1),loc='upper right',fancybox = False, framealpha = 1, edgecolor = 'k')
            pyplot.tight_layout()
            pyplot.grid(b=True, which='major', color='#666666', alpha = 0.3, linestyle='-')
            pyplot.xticks(fontsize=14)
            pyplot.yticks(fontsize=14)
            
            s = """NSE = {:.2f}\nR²  = {:.2f}\nRMSE = {:.2f}\nBias = {:.2f}""".format(scores.NSE[0],scores.R2[0],scores.RMSE[0],scores.Bias[0])
            
            pyplot.figtext(0.9, 0.3, s, bbox=dict(facecolor='white'),fontsize = 15)
            # pyplot.savefig('./refined/Test_'+Well_ID+'_CNN.png', dpi=300)
            pyplot.savefig('./Test_'+Well_ID+'_Graph2_CNN.png', dpi=300)            
            # pyplot.show()
            
            
        #%% plots
            pyplot.figure(figsize=(20,6))
        
            pyplot.plot(StopData.index, sim_stop, 'y', label ="simulated mean", linewidth = 1.7)
            pyplot.plot(StopData.index, StopData['GWL'], 'k', label ="observed", linewidth=1.7,alpha=0.9)
            
            pyplot.plot(OptData.index, sim_opt, 'g', label ="simulated mean", linewidth = 1.7)
            pyplot.plot(OptData.index, OptData['GWL'], 'k', label ="observed", linewidth=1.7,alpha=0.9)
            
            pyplot.plot(TestData.index, sim, 'r', label ="simulated mean", linewidth = 1.7)
            pyplot.plot(TestData.index, obs, 'k', label ="observed", linewidth=1.7,alpha=0.9)
            
            pyplot.title("CNN Model part: "+Well_ID, size=17,fontweight = 'bold')
            pyplot.ylabel('GWL [m asl]', size=15)
            pyplot.xlabel('Date',size=15)
            pyplot.legend(fontsize=15,bbox_to_anchor=(1.18, 1),loc='upper right',fancybox = False, framealpha = 1, edgecolor = 'k')
            pyplot.tight_layout()
            pyplot.grid(b=True, which='major', color='#666666', alpha = 0.3, linestyle='-')
            pyplot.xticks(fontsize=14)
            pyplot.yticks(fontsize=14)
            
            s = """NSE = {:.2f}\nR²  = {:.2f}\nRMSE = {:.2f}\nBias = {:.2f}""".format(scores.NSE[0],scores.R2[0],scores.RMSE[0],scores.Bias[0])
            
            pyplot.figtext(0.9, 0.3, s, bbox=dict(facecolor='white'),fontsize = 15)
            pyplot.savefig('./Test_'+Well_ID+'_Graph1_CNN.png', dpi=300)            
            # pyplot.show()
            
            
        

# %%
