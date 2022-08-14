from pickle import TRUE
from typing import List, Dict
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import false
from vnpy_portfoliostrategy import StrategyTemplate, StrategyEngine
from vnpy.trader.object import TickData,OrderData
from statsmodels.formula.api import ols
from vnpy.trader.constant import Status,Direction
import statsmodels.tsa.stattools as ts
import time
class TESTccs(StrategyTemplate):

    author = "用Python的交易员"
    leg1_symbol = ""
    leg2_symbol = ""
    variables = ["leg1_symbol", "leg2_symbol", "open_position_1", "open_position_2", "waiting_open", "waiting_close", "flag", "flag2", "complusary_closing_time", "close_one", "close_two","open_one","open_two"]

    def __init__(
            self,
            strategy_engine: StrategyEngine,
            strategy_name: str,
            vt_symbols: List[str],
            setting: dict
    ):

        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)
        self.leg1_symbol, self.leg2_symbol = vt_symbols
        self.main_time = -10
        self.second_time = -100
        self.send_open_time = 1e7
        self.send_close_time = 1e7
        self.flag = False #判断交易还是建模
        self.flag2 = True #判断是否进入第二次线性回归
        self.waiting_open = False #等待订单成交
        self.waiting_close = False
        self.tracked = False
        self.running = True
        self.open_time = 1e7
        self.close_time = 1e7
        self.chasing = True
        self.send = False
        self.send_track = False
        self.send_track_open = False
        #储存建模所用数据
        self.main_price, self.second_price= [], []
        self.model_ols = {'price1': self.main_price, 'price2': self.second_price}
        #判断开仓状态
        self.open_position_1 = 0 #多一空二
        self.open_position_2 = 0 #多二空一
        #收盘前半小时不开仓
        self.complusary_open_time_day = 142000
        self.complusary_open_time_night = 224000
        self.complusary_open_time = 1e6
        #强制锁仓时间
        self.complusary_closing_time_day = 145000
        self.complusary_closing_time_night = 225000
        self.complusary_closing_time = 1e6
        #计数器
        self.count = 0
        #订单状态
        self.close_one,self.close_two,self.open_one,self.open_two = [],[],[],[]
        self.not_active = set(
            [Status.SUBMITTING, Status.NOTTRADED, Status.PARTTRADED, Status.CANCELLED, Status.REJECTED])
        self.active = set([Status.ALLTRADED])
        
    def on_init(self):
        self.write_log("策略初始化")

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.main_time = -10
        self.second_time = -100
        self.send_open_time = 1e7
        self.send_close_time = 1e7
        self.flag = False #判断交易还是建模
        self.flag2 = True #判断是否进入第二次线性回归
        self.waiting_open = False #等待订单成交
        self.waiting_close = False
        self.tracked = False
        self.running = True
        self.open_time = 1e7
        self.close_time = 1e7
        self.chasing = True
        self.send = False
        self.send_track = False
        self.send_track_open = False
        #储存建模所用数据
        self.main_price, self.second_price= [], []
        self.model_ols = {'price1': self.main_price, 'price2': self.second_price}
        #判断开仓状态
        self.open_position_1 = 0 #多一空二
        self.open_position_2 = 0 #多二空一
        #收盘前半小时不开仓
        self.complusary_open_time_day = 142000
        self.complusary_open_time_night = 224000
        self.complusary_open_time = 1e6
        #强制锁仓时间
        self.complusary_closing_time_day = 145000
        self.complusary_closing_time_night = 225000
        self.complusary_closing_time = 1e6
        #计数器
        self.count = 0
        #订单状态
        self.close_one,self.close_two,self.open_one,self.open_two = [],[],[],[]
        self.not_active = set(
            [Status.SUBMITTING, Status.NOTTRADED, Status.PARTTRADED, Status.CANCELLED, Status.REJECTED])
        self.active = set([Status.ALLTRADED])
        
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")
    
    def linear_regression(self):
        self.model_ols_done=pd.DataFrame(self.model_ols)
        self.lm = ols('price2~price1', data=self.model_ols_done).fit()

    def trade_1(self):
        self.open_one = self.buy(self.leg1_symbol,self.main_ask_price,22)
        self.open_two = self.short(self.leg2_symbol,self.second_bid_price,20)
        #self.open_position_1 = True
        self.open_time = self.count
        self.open_resid_1 = self.resid2
        self.open_a1 = self.main_ask_price
        self.open_b2 = self.second_bid_price
        self.open_price1 = self.main_bid_price
        self.open_price2 = self.second_ask_price
        self.waiting_open = True
        self.tracked = True
        self.send_open_time = self.count
        self.a1,self.b1,self.a2,self.b2 = self.main_ask_price,self.main_bid_price,self.second_ask_price,self.second_bid_price
        self.buffer = self.main_ask_bid+self.second_ask_bid
        self.open_position_1 += 1
        self.write_log(['开仓',self.resid2,self.diff])
    def trade_2(self):
        self.open_two = self.buy(self.leg2_symbol,self.second_ask_price,22)
        self.open_one = self.short(self.leg1_symbol,self.main_bid_price,20)
        #self.open_position_2 = True
        self.open_time = self.count
        self.open_resid_2 = self.resid2
        self.open_a2 = self.second_ask_price
        self.open_b1 = self.main_bid_price
        self.open_price1 = self.main_ask_price
        self.open_price2 = self.second_bid_price
        self.waiting_open = True
        self.tracked = True
        self.send_open_time = self.count
        self.open_position_2 += 1
        self.a1,self.b1,self.a2,self.b2 = self.main_ask_price,self.main_bid_price,self.second_ask_price,self.second_bid_price
        self.buffer = self.main_ask_bid+self.second_ask_bid
        self.write_log(['开仓',self.resid2,self.diff])

    def closing_1(self):
        self.close_one = self.sell(self.leg1_symbol,self.main_bid_price,22)
        self.close_two = self.cover(self.leg2_symbol,self.second_ask_price,20)
        # self.open_position_1 = False
        self.a1,self.b1,self.a2,self.b2 = self.main_ask_price,self.main_bid_price,self.second_ask_price,self.second_bid_price
        self.buffer = self.main_ask_bid+self.second_ask_bid
        self.close_time = self.count
        self.waiting_close = True
        self.close_price1 = self.main_bid_price
        self.close_price2 = self.second_ask_price
        self.send_close_time = self.count
        self.write_log(['平仓',self.resid2,self.diff])
    def closing_2(self):
        self.close_two = self.sell(self.leg2_symbol,self.second_bid_price,22)
        self.close_one = self.cover(self.leg1_symbol,self.main_ask_price,20)
        # self.open_position_2 = False
        self.close_time = self.count
        self.waiting_close = True
        self.close_price1 = self.main_ask_price
        self.close_price2 = self.second_bid_price
        self.a1,self.b1,self.a2,self.b2 = self.main_ask_price,self.main_bid_price,self.second_ask_price,self.second_bid_price
        self.buffer = self.main_ask_bid+self.second_ask_bid
        self.chasing = True
        self.send_close_time = self.count
        self.write_log(['平仓',self.resid2,self.diff])
    def track_open(self):
        self.open_order_id_1 = self.open_one[0]
        self.open_order_id_2 = self.open_two[0]
        self.status_1 = self.get_order(self.open_order_id_1)
        self.status_2 = self.get_order(self.open_order_id_2)
        if (self.status_1.status in self.active) and (self.status_2.status in self.not_active) and (self.count - self.open_time  > 2400 or abs(self.open_price1 - self.price1)>10):
            self.cancel_order(self.open_order_id_2)
            self.write_log(['订单一成交'])
            self.send_track_open = True
        elif (self.status_1.status in self.not_active) and (self.status_2.status in self.active) and (self.count - self.open_time  > 2400 or abs(self.open_price2 - self.price2)>10):
            self.cancel_order(self.open_order_id_1)
            self.write_log(['订单二成交'])
            self.send_track_open = True
        elif (self.status_1.status in self.not_active) and (self.status_2.status in self.not_active) and (self.count - self.open_time  > 2400):
            self.cancel_order(self.open_order_id_1)
            self.cancel_order(self.open_order_id_2)
            self.write_log(['订单均未成交'])
            self.send_track_open = True
        elif (self.status_1.status in self.active) and (self.status_2.status in self.active):
            self.write_log(['订单成交'])
            self.waiting_open = False
    def send_open(self):
        self.open_order_id_1 = self.open_one[0]
        self.open_order_id_2 = self.open_two[0]
        self.status_1 = self.get_order(self.open_order_id_1)
        self.status_2 = self.get_order(self.open_order_id_2)
        if (self.status_1.status in self.active) and (self.status_2.status in self.not_active):
            num = self.status_2.volume - self.status_2.traded
            if self.status_1.direction==Direction.LONG:
                self.open_two = self.short(self.leg2_symbol,self.second_bid_price,num)
                
            else:
                self.open_two = self.buy(self.leg2_symbol,self.main_ask_price,num)
               
            self.send_open_time = self.count
            self.send_track_open = False
        elif (self.status_1.status in self.not_active) and (self.status_2.status in self.active):
            num = self.status_1.volume - self.status_1.traded
            if self.status_2.direction==Direction.LONG:
                self.open_one = self.short(self.leg1_symbol,self.second_bid_price,num)
               
            else:
                self.open_one = self.buy(self.leg1_symbol,self.second_ask_price,num)
            self.send_open_time = self.count
            self.send_track_open = False
        elif (self.status_1.status in self.not_active) and (self.status_2.status in self.not_active):
            num2 = self.status_2.volume - self.status_2.traded
            num1 = self.status_1.volume - self.status_1.traded
            if self.status_1.direction==Direction.LONG:  
                self.open_one = self.buy(self.leg1_symbol,self.main_ask_price,num1)
                self.open_two = self.short(self.leg2_symbol,self.second_bid_price,num2)
           
            else:
                self.open_one = self.short(self.leg1_symbol,self.main_bid_price,num1)
                self.open_two = self.buy(self.leg2_symbol,self.second_ask_price,num2)
            self.send_open_time = self.count
            self.send_track_open = False
       
    def track_close(self):
        self.close_order_id_1 = self.close_one[0]
        self.close_order_id_2 = self.close_two[0]
        self.status_1 = self.get_order(self.close_order_id_1)
        self.status_2 = self.get_order(self.close_order_id_2)
        if (self.status_1.status in self.active) and (self.status_2.status in self.not_active) and (self.count - self.close_time  > 2400 or abs(self.close_price1 - self.price1)>10):
            self.cancel_order(self.close_order_id_2)
            self.send_track = True
            self.write_log(['订单一成交'])
        elif (self.status_1.status in self.not_active) and (self.status_2.status in self.active) and (self.count - self.close_time  > 2400 or abs(self.close_price2 - (self.price2/1.2))>10):
            self.cancel_order(self.close_order_id_1)
            self.send_track = True
            self.write_log(['订单二成交'])
        elif (self.status_1.status in self.not_active) and (self.status_2.status in self.not_active) and self.count - self.close_time  > 2400:
            self.cancel_order(self.close_order_id_1)
            self.cancel_order(self.close_order_id_2)
            self.send_track = True
            self.write_log(['订单均未成交'])
        elif (self.status_1.status in self.active) and (self.status_2.status in self.active):
            if self.status_2.direction == Direction.LONG:
                self.open_position_1 = 0
            else:
                self.open_position_2 = 0
            self.write_log(['订单成交'])
            self.waiting_close = False

    def send_close(self):
        self.close_order_id_1 = self.close_one[0]
        self.close_order_id_2 = self.close_two[0]
        self.status_1 = self.get_order(self.close_order_id_1)
        self.status_2 = self.get_order(self.close_order_id_2)
        if (self.status_1.status in self.active) and (self.status_2.status in self.not_active):
            if self.active_orderids == set() and self.get_pos(self.leg2_symbol)!=0:    
                if self.status_1.direction == Direction.LONG:
                    self.close_two = self.sell(self.leg2_symbol, self.second_bid_price, self.status_2.volume-self.status_2.traded)
                    self.open_position_2 = 0
                else:
                    self.close_two = self.cover(self.leg2_symbol, self.second_ask_price, self.status_2.volume-self.status_2.traded)
                    self.open_position_1 = 0
                self.send_track = False
                self.send_close_time = self.count
        elif (self.status_1.status in self.not_active) and (self.status_2.status in self.active):
            if self.active_orderids == set() and self.get_pos(self.leg1_symbol)!=0:
                if self.status_2.direction == Direction.LONG:
                    self.close_one = self.sell(self.leg1_symbol, self.main_bid_price, self.status_1.volume-self.status_1.traded)
                    self.open_position_1 = 0
                else:
                    self.close_one = self.cover(self.leg1_symbol, self.main_ask_price, self.status_1.volume-self.status_1.traded) 
                    self.open_position_2 = 0
                self.send_track = False
                self.send_close_time = self.count
        elif (self.status_1.status in self.not_active) and (self.status_2.status in self.not_active):
            if self.active_orderids == set() and self.get_pos(self.leg2_symbol)!=0 and self.get_pos(self.leg1_symbol)!=0:
                if self.status_2.direction == Direction.LONG:
                    self.close_one = self.sell(self.leg1_symbol, self.main_bid_price, self.status_1.volume-self.status_1.traded)
                    self.close_two = self.cover(self.leg2_symbol, self.second_ask_price, self.status_2.volume-self.status_2.traded)
                    self.open_position_1 = 0
                else:
                    self.trading = True
                    self.close_one = self.sell(self.leg2_symbol, self.second_bid_price, self.status_2.volume-self.status_2.traded)
                    self.close_two = self.cover(self.leg1_symbol, self.main_ask_price, self.status_1.volume-self.status_1.traded)
                    self.open_position_2 = 0
                self.send_track = False
                self.send_close_time = self.count
        self.write_log('发单')
        
    

        
    def on_tick(self, tick: TickData):
        if tick.bid_price_1 > 1000000 or tick.bid_price_1 < 10:
            pass
        elif tick.ask_price_1 > 1000000 or tick.ask_price_1 < 10:
            pass
        else:
            #记录当前时间
            self.tick_time = int(time.strftime("%H%M%S"))
            if 0 <= self.tick_time <= 23000:
                self.tick_time += 240000
            if 90000<=self.tick_time<=151000:
                self.complusary_open_time = self.complusary_open_time_day
                self.complusary_closing_time = self.complusary_closing_time_day
            if 210000<=self.tick_time<=270000:
                self.complusary_open_time = self.complusary_open_time_night
                self.complusary_closing_time = self.complusary_closing_time_night

            self.count += 1
            #计算
            if 90000 <= self.tick_time and self.flag == False and self.waiting_open == False and self.waiting_close == False and self.running == True:
                if tick.vt_symbol == self.leg1_symbol:
                    self.main_time=float(datetime.strftime(tick.datetime,'%H%M%S.%f'))
                    self.deal_price1 =(tick.bid_price_1 + tick.ask_price_1) *1.1/ 2
                    
                if tick.vt_symbol == self.leg2_symbol:
                    
                    self.second_time=float(datetime.strftime(tick.datetime,'%H%M%S.%f'))
                    self.deal_price2 = (tick.bid_price_1  + tick.ask_price_1) / 2

                if -0.5<=self.second_time-self.main_time<=0.5 and self.flag == False:

                    self.main_price.append(self.deal_price1)
                    self.second_price.append(self.deal_price2)
                    self.write_log('adding')
                if len(self.main_price) >3600:
                    self.linear_regression()
                    p = ts.adfuller(self.lm.resid, 1)[1]
                    
                    
                    # with open(self.path,'a',newline='') as myfile:
                    #     writer = csv.writer(myfile)
                    #     writer.writerow([(p),np.std(self.lm.resid)])s
                    #     writer.writerow([self.lm.params['price1'],self.lm.params['Intercept']])
                    if p < 0.05 and np.std(self.lm.resid)>0.15:
                        self.write_log([(p,np.std(self.lm.resid))])
                        self.write_log([self.lm.params['price1'],self.lm.params['Intercept']])
                        self.flag = True
                        self.mean = np.mean(self.lm.resid)
                        self.std = np.std(self.lm.resid)

            #交易
            if tick.vt_symbol == self.leg1_symbol:
            # self.price1 = np.log((tick.ask_price_1 + tick.bid_price_1) / 2)
                self.main_ask_bid = tick.ask_price_1 - tick.bid_price_1
                # self.main_bid_price = np.log(tick.bid_price_1)
                # self.main_ask_price = np.log(tick.ask_price_1)
                self.price1 = 1.1*(tick.ask_price_1 + tick.bid_price_1) / 2
                self.main_bid_price = tick.bid_price_1
                self.main_ask_price = tick.ask_price_1
                self.main_time = float(datetime.strftime(tick.datetime, '%H%M%S.%f'))
                # self.main_bid_price5 = np.log(tick.bid_price_5)
                # self.main_ask_price5 = np.log(tick.ask_price_5)
            else:
                # self.price2 = np.log(1.2*(tick.ask_price_1 + tick.bid_price_1) / 2)
                self.second_ask_bid = tick.ask_price_1 - tick.bid_price_1
                # self.second_bid_price = np.log(tick.bid_price_1)
                # self.second_ask_price = np.log(tick.ask_price_1)
                self.second_time = float(datetime.strftime(tick.datetime, '%H%M%S.%f'))
                self.second_bid_price = tick.bid_price_1
                self.second_ask_price = tick.ask_price_1
                self.price2 = (tick.ask_price_1 + tick.bid_price_1) / 2
                
                # self.second_bid_price5 = np.log(tick.bid_price_5)
                # self.second_ask_price5 = np.log(tick.ask_price_5)
            if -0.5<=self.second_time-self.main_time<=0.5 and self.running == True:
                if self.flag == True and self.waiting_open == False and self.waiting_close == False:
                    self.resid = self.price2 - self.lm.params['price1'] * self.price1 - self.lm.params['Intercept']
                    self.resid2 = (self.resid - self.mean) / self.std
                    self.diff = self.price2 - self.price1
                    self.write_log([self.resid2,self.diff])
                   
                    #强制平仓
                    if self.tick_time > self.complusary_closing_time:  
                        if self.open_position_1 > 0:
                            self.closing_1()
                        if self.open_position_2 > 0:
                            self.closing_2()
                    if self.main_ask_bid+self.second_ask_bid <= 3:
                        if self.resid2 < -9/self.std and self.open_position_2 == 0 and self.open_position_1 == 0 and self.tick_time <=  self.complusary_open_time:
                            self.trade_2()
                        if self.open_position_2 > 0:
                            if self.resid2 - self.open_resid_2 > (10/self.std):
                                self.closing_2()    
                        if self.resid2 > (9/self.std) and self.open_position_1 == 0 and self.open_position_2 == 0 and self.tick_time <=  self.complusary_open_time:
                            self.trade_1()
                        if self.open_position_1 > 0:
                            if self.open_resid_1 - self.resid2 > (10/self.std):
                                self.closing_1()
                        if self.open_position_1 > 0:
                            if 7200 < self.count-self.open_time <= 14400:# or (self.resid2 > (3.4/self.std)): 
                                if (self.main_bid_price-self.open_a1)*110-(self.second_ask_price-self.open_b2)*100 >= 120:
                                    self.closing_1()
                                    
                            if self.count-self.open_time > 14400:
                                self.closing_1() 
                                if self.flag2 == True:
                                    self.flag = False
                                    self.flag2 = False
                                    self.main_price.clear()
                                    self.second_price.clear()
                                else:self.running = False
                        if self.open_position_2 > 0:
                            if 7200 < self.count-self.open_time < 14400:
                                if (self.second_bid_price-self.open_a2)*100-(self.main_ask_price-self.open_b1)*110 >= 120:
                                    self.closing_2()
                                  
                            if self.count-self.open_time > 14400:# or (self.resid2 < (-3.4/self.std)):
                                self.closing_2()
                                if self.flag2 == True:
                                    self.flag = False
                                    self.flag2 = False
                                    self.main_price.clear()
                                    self.second_price.clear()
                                else:self.running = False
        

                if self.waiting_open == True:
                    if self.send_track_open == True:
                        self.send_open()
                    if self.count - self.open_time  > 10 and self.count - self.send_open_time > 2:
                        self.track_open()
                if self.waiting_close == True:
        
                    if self.send_track == True:
                        self.send_close()
                    if self.count - self.close_time  > 10 and self.count - self.send_close_time > 2:
                        self.track_close()
                        
            self.put_event()