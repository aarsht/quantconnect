import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout

class LSTM5Min(QCAlgorithm):
  
    def Initialize(self):
  
        self.SetStartDate(2019,2,1)
        self.SetEndDate(2019,5,1)
        self.SetCash(100000)
        self.resolution_period = 5
        self.model_warmup = 100000
        self.indicator_warmup = 30
        self.equity = 'AAPL'
        self.AddEquity(self.equity, Resolution.Minute)
        # setting data resolution to 5 minutes.
        consolidator = TradeBarConsolidator(self.resolution_period)
        consolidator.DataConsolidated += self.OnDataConsolidated
        self.SubscriptionManager.AddConsolidator(self.equity, consolidator)
        self.rsi = RelativeStrengthIndex(9)
        self.RegisterIndicator(self.equity, self.rsi,consolidator)
        self.mfi = MoneyFlowIndex(14)
        self.RegisterIndicator(self.equity, self.mfi,consolidator)
        self.ema = ExponentialMovingAverage(9)
        self.RegisterIndicator(self.equity, self.ema, consolidator)
        self.bb = BollingerBands(30,2,2)
        self.RegisterIndicator(self.equity, self.bb, consolidator)
        self.ppo = PercentagePriceOscillator(12,26)
        self.RegisterIndicator(self.equity, self.ppo, consolidator)
        self.cci = CommodityChannelIndex(20)
        self.RegisterIndicator(self.equity, self.cci, consolidator)
        self.dema = DoubleExponentialMovingAverage(9)
        self.RegisterIndicator(self.equity, self.dema, consolidator)
        self.tema = TripleExponentialMovingAverage(9)
        self.RegisterIndicator(self.equity, self.tema, consolidator)
        self.SetWarmup(self.model_warmup*self.resolution_period)
        self.count = 0
        self.tech_data = pd.DataFrame()
        self.scaled_data = pd.DataFrame()
     

    def OnData(self, data):

        pass
   
    # The event will be triggered each 5 min interval to pump the data.
    def OnDataConsolidated(self, sender, bar):
        self.count+=1
        # warming up for model fitting.
        if self.IsWarmingUp:
            # warming up upto max indicator warming up period.
            if self.count > self.indicator_warmup:
                if self.tech_data.empty:
                    self.tech_data = pd.DataFrame([[bar.Open,bar.Close,bar.Low,bar.High,bar.Volume,float(self.mfi.ToString()),float(self.rsi.ToString()),float(self.ema.ToString()),float(self.bb.ToString()),float(self.ppo.ToString()),float(self.cci.ToString()),float(self.dema.ToString()),float(self.tema.ToString())]],columns=['open','close','low','high','volume','mfi','rsi','ema','bb','ppo','cci','dema','tema'])
                else:
                    self.tech_data = self.tech_data.append(pd.DataFrame([[bar.Open,bar.Close,bar.Low,bar.High,bar.Volume,float(self.mfi.ToString()),float(self.rsi.ToString()),float(self.ema.ToString()),float(self.bb.ToString()),float(self.ppo.ToString()),float(self.cci.ToString()),float(self.dema.ToString()),float(self.tema.ToString())]],columns=['open','close','low','high','volume','mfi','rsi','ema','bb','ppo','cci','dema','tema']))  
                    # model fitting at the last time step of warm up period.
                    if self.count == self.model_warmup:
                        self.Debug('tech data shape:')
                        self.Debug(self.tech_data.shape)
                        # difference between non indicator data.
                        self.tech_data.iloc[:,:5] = self.tech_data.iloc[:,:5] - self.tech_data.iloc[:,:5].shift(1)
                        self.tech_data.dropna(how='any',inplace=True)
                        for column in self.tech_data:
                            self.scaled_data[column] = np.tanh((self.tech_data[column].values-np.mean(self.tech_data[column].values))/np.std(self.tech_data[column].values))
                        self.x_train = np.array(self.scaled_data.tail(-1))
                        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0],1,13))
                        # categorizing target data
                        self.y_train = pd.cut(self.tech_data['close'].shift(-1),[-np.inf,0,np.inf],labels=[0,1])
                        self.y_train = np.array(self.y_train.dropna())
                        self.model = Sequential()
                        self.model.add(LSTM(units=80, return_sequences=True, input_shape=(1,13)))
                        self.model.add(Dropout(0.2))
                        self.model.add(LSTM(units=80, return_sequences=True))
                        self.model.add(Dropout(0.2))
                        self.model.add(LSTM(units=80))
                        self.model.add(Dropout(0.2))
                        self.model.add(Dense(units=1))

                        # Compiling the RNN
                        self.model.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['acc','mse'])
                        # Fitting to the training set
                        history_callback = self.model.fit(self.x_train,self.y_train,epochs=10,batch_size=32,verbose=2)
                        self.Debug(history_callback.history["loss"])
        else:
            # Fitting model further due to time limit of 10 minutes at each time step.
            if self.count == self.model_warmup+1:
                history_callback1 = self.model.fit(self.x_train,self.y_train,epochs=10,batch_size=32,verbose=2)
                self.Debug(history_callback1.history["loss"])
                y_sample = pd.DataFrame(self.model.predict(self.x_train))
                self.Debug('y_pred')
                self.Debug(y_sample.describe)
            x_live = pd.DataFrame([[bar.Open,bar.Close,bar.Low,bar.High,bar.Volume,float(self.mfi.ToString()),float(self.rsi.ToString()),float(self.ema.ToString()),float(self.bb.ToString()),float(self.ppo.ToString()),float(self.cci.ToString()),float(self.dema.ToString()),float(self.tema.ToString())]],columns=['open','close','low','high','volume','mfi','rsi','ema','bb','ppo','cci','dema','tema'])
            history_tradebar = self.History(self.Symbol(self.equity), self.resolution_period+1, Resolution.Minute)
            # difference between non indicator data.
            x_live.iloc[:,:5] = x_live.iloc[:,:5] - history_tradebar.iloc[0,:]
            #applying same tanh transformation on live data
            for column in x_live:
                x_live[column] = np.tanh((x_live[column].values-np.mean(self.tech_data[column].values))/np.std(self.tech_data[column].values))
            x_live = np.reshape(np.array(x_live), (1,1,13))
            y_predicted = self.model.predict(x_live)
            if y_predicted >= 0.5:
                self.SetHoldings(self.equity, 1.0)
            else:
                self.SetHoldings(self.equity, 0)
