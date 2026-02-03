import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pandas_ta as ta
from backtesting import Backtest, Strategy


present = datetime.today().strftime('%Y-%m-%d')

spy = yf.download("QQQ", start="2001-01-01", end=present)
spy.columns = [col[0] for col in spy.columns]


spy["SMA_200"] = ta.sma(spy["Close"], length=200)
spy["SMA_9"]   = ta.sma(spy["Close"], length=9)
spy["SMA_5"]   = ta.sma(spy["Close"], length=9)
spy["ATR"] = ta.atr(spy["High"],spy["Low"],spy["Close"], length=14)

#spy = spy.reset_index()


# tqqq = yf.download("QQQ", start="2001-01-01", end=present)
# tqqq.columns = [col[0] for col in tqqq.columns]
# tqqq["SMA_200"] = ta.sma(tqqq["Close"], length=200)
# tqqq["SMA_9"]   = ta.sma(tqqq["Close"], length=9)
#tqqq = tqqq.reset_index()

#spy.index = pd.to_datetime(spy.index)
#spy = spy.loc["2022-01-01":"2023-12-31"]





class IvanSherman(Strategy):
    
    def init(self):
       
        self.risk_prc = 0.01
        self.sma200 = self.I(
            lambda x: ta.sma(pd.Series(x), length=200).to_numpy(),
            self.data.Close
        )

        # self.sma9 = self.I(
        #     lambda x: ta.sma(pd.Series(x), length=9).to_numpy(),
        #     self.data.Close
        # )
        
        self.sma5 = self.I(
            lambda x: ta.sma(pd.Series(x), length=5).to_numpy(),
            self.data.Close
        )
        
        
        self.above_sma200 = False
        self.three_consecutive_red = False
        
    def close_above_sma5(self):
        if len(self.data.Close) < 2:
            return False
        return self.data.Close[-1] > self.data.SMA_5[-1] and self.data.Close[-2] <= self.data.SMA_5[-2]
        
    def check_three_consecutive_red(self):
        if len(self.data.Close) < 5:
            return False
        return (self.data.Close[-1] < self.data.Open[-1] and self.data.Close[-1] <= self.data.Close[-2] and
                self.data.Close[-2] < self.data.Open[-2] and self.data.Close[-2] <= self.data.Close[-3] and
                self.data.Close[-3] < self.data.Open[-3]) and self.data.Close[-3] <= self.data.Close[-4] 
        
        
    def check_tp(self):
        
        if len(self.data.Close) < 5 or self.take_profit is None:
            return False
        
        return self.data.Close[-1] >= self.take_profit
    
    def check_stop_loss(self):
        
        if len(self.data.Close) < 5 or self.stop_price is None:
            return False
        
        return self.data.Close[-1] < self.stop_price
        

    def next(self):
        #self.above_sma200 = self.data.Close[-1] > self.sma200[-1]
        self.above_sma200 = self.data.Close[-1] > self.data.SMA_200[-1]
        self.three_consecutive_red = self.check_three_consecutive_red()
        
        
        #print(f"Above SMA200: {self.above_sma200}, Three Red: {self.three_consecutive_red},  position: {self.position}")
        
        if not self.position:
            if self.above_sma200 and  self.three_consecutive_red:
              
                self.stop_price  = self.data.Close[-1] - self.data.ATR[-1] * 2
                self.take_profit  = self.data.Close[-1] + self.data.ATR[-1] * 2
                amount = self.equity * self.risk_prc
                size = amount // (self.data.Close[-1] - self.stop_price)
                self.buy(size=size)
                #print(f"Going long {size} shares at {self.data.Close[-1]}, stop at {self.stop_price}")
                
        if self.position and (self.check_tp() or self.check_stop_loss()) :
            #print("closing position")
            self.position.close()
            self.stop_price = None
            self.take_profit = None
            
        
        pass

bt = Backtest(spy, IvanSherman, cash=25_000)

# run the strategy
stats = bt.run()
bt.plot()
print(stats)
stats['_trades'].to_csv("trades_qqq.csv", index=False)
tra = pd.read_csv("trades_qqq.csv")
#x = tra[["Size","EntryBar","ExitBar","EntryPrice","ExitPrice","SL","TP","PnL","Commission","ReturnPct","EntryTime","ExitTime"]]
#print(x)