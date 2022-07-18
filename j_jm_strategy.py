from typing import List, Dict
from datetime import datetime
import time
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import false
from vnpy_portfoliostrategy import StrategyTemplate, StrategyEngine
from vnpy.trader.utility import BarGenerator
from vnpy.trader.object import TickData, OrderData
from statsmodels.formula.api import ols
from vnpy.trader.constant import Status, Direction
import winsound
import statsmodels.tsa.stattools as ts
import csv

class JJmStrategy(StrategyTemplate):
    author = "用Python的交易员"
    leg1_symbol = ""
    leg2_symbol = ""
    daytime = False  # 交易时间
    wrong = True  # 不合理数据报错
    variables = ["leg1_symbol", "leg2_symbol", "open_position_1", "open_position_2", "waiting_open", "waiting_close",
                 "flag", "flag2", "complusary_closing_time", "daytime", "wrong"]

    def __init__(
            self,
            strategy_engine: StrategyEngine,
            strategy_name: str,
            vt_symbols: List[str],
            setting: dict
    ):

        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)
        self.leg1_symbol, self.leg2_symbol = vt_symbols
        self.main_time = 9999
        self.second_time = 999999
        self.flag = False  # 判断交易还是建模
        self.flag2 = True  # 判断是否进入第二次线性回归
        self.waiting_open = False  # 等待订单成交
        self.waiting_close = False
        self.running = True
        self.open_time = 1e5
        self.close_time = 1e5

        # 储存建模所用数据
        self.main_price, self.second_price= [], []
        self.model_ols = {'price1': self.main_price, 'price2': self.second_price}
        # 判断开仓状态
        self.open_position_1 = False  # 多一空二
        self.open_position_2 = False  # 多二空一
        # 强制锁仓时间
        self.complusary_closing_time_day = 145000
        self.complusary_closing_time_night = 225000
        self.complusary_closing_time = 1e5
        # 计数器
        self.count = 0
        # 订单状态
        self.not_active = set(
            [Status.SUBMITTING, Status.NOTTRADED, Status.PARTTRADED, Status.CANCELLED, Status.REJECTED])
        self.active = set([Status.ALLTRADED])
        ##权重
        self.coefficient_1 = 1.2
        self.coefficient_2 = 1

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        print("策略初始化")

    def on_start(self):
        """
        Callback when strategy is started.
        """
        print("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        print("策略停止")

    def linear_regression(self):
        self.model_ols=pd.DataFrame(self.model_ols)
        self.lm = ols('price2~price1', data=self.model_ols).fit()
       
    def trade_1(self, amount):
        self.open_one = self.buy(self.leg1_symbol,self.main_ask_price,2)
        self.open_two = self.short(self.leg2_symbol,self.second_bid_price,1)
        self.open_time = self.count
        self.open_resid_1 = self.resid2
        self.open_a1 = self.main_ask_price
        self.open_b1 = self.second_bid_price
        self.waiting_open = True
        self.write_log(['开仓',self.resid2,self.diff])
        
    def trade_2(self, amount):
        self.open_two = self.buy(self.leg2_symbol, self.second_ask_price, 1)
        self.open_one = self.short(self.leg1_symbol, self.main_bid_price, 2)
        # self.open_position_2 = True
        self.open_time = self.count
        self.open_resid_2 = self.resid2
        self.open_a2 = self.second_ask_price
        self.open_b2 = self.main_bid_price
        self.waiting_open = True
        self.write_log(['开仓',self.resid2,self.diff])
        
    def closing_1(self, amount):

        self.close_one = self.sell(self.leg1_symbol,self.main_bid_price,2)
        self.close_two = self.cover(self.leg2_symbol,self.second_ask_price,1)
        # self.open_position_1 = False
        self.close_time = self.count
        self.waiting_close = True
        self.write_log(['平仓',self.resid2,self.diff])

    def closing_2(self, amount):
        self.close_two = self.sell(self.leg2_symbol,self.second_bid_price,1)
        self.close_one = self.cover(self.leg1_symbol,self.main_ask_price,2)
        # self.open_position_2 = False
        self.close_time = self.count
        self.waiting_close = True
        self.write_log(['平仓',self.resid2,self.diff])
       
    def track_open(self):
        self.open_order_id_1 = self.open_one[0]
        self.open_order_id_2 = self.open_two[0]
        self.status_1 = self.get_order(self.open_order_id_1)
        self.status_2 = self.get_order(self.open_order_id_2)
        if (self.status_1.status in self.active) and (self.status_2.status in self.not_active):
            self.cancel_order(self.open_order_id_2)
            if self.status_1.direction==Direction.LONG:
                self.sell(self.leg1_symbol,self.main_bid_price,2)
            else:
                self.cover(self.leg1_symbol,self.main_ask_price,2)
            self.write_log(['订单一成交'])
        elif (self.status_1.status in self.not_active) and (self.status_2.status in self.active):
            self.cancel_order(self.open_order_id_1)
            if self.status_2.direction==Direction.LONG:
                self.sell(self.leg2_symbol,self.second_bid_price,1)
            else:
                self.cover(self.leg2_symbol,self.second_ask_price,1)
            self.write_log(['订单二成交'])
        elif (self.status_1.status in self.not_active) and (self.status_2.status in self.not_active):
            self.cancel_order(self.open_order_id_1)
            self.cancel_order(self.open_order_id_2)
            self.write_log(['订单均未成交'])
        else:
            self.write_log(['订单成交'])
            if self.status_1.direction==Direction.LONG:
                self.open_position_1 = True
            else:
                self.open_position_2 = True
        self.waiting_open = False

    def track_close(self):
        self.close_order_id_1 = self.close_one[0]
        self.close_order_id_2 = self.close_two[0]
        self.status_1 = self.get_order(self.close_order_id_1)
        self.status_2 = self.get_order(self.close_order_id_2)
        if (self.status_1.status in self.active) and (self.status_2.status in self.not_active):
            self.cancel_order(self.close_order_id_2)
            if self.status_1.direction == Direction.LONG:
                self.close_two = self.sell(self.leg2_symbol, self.second_bid_price5, 1)
            else:
                self.close_two = self.cover(self.leg2_symbol, self.second_bid_price5, 1)
            self.write_log(['订单一成交'])
            self.close_time = self.count
        elif (self.status_1.status in self.not_active) and (self.status_2.status in self.active):
            self.cancel_order(self.close_order_id_1)
            if self.status_2.direction == Direction.LONG:
                self.close_one = self.sell(self.leg1_symbol, self.main_bid_price5, 2)
            else:
                self.close_one = self.cover(self.leg1_symbol, self.main_ask_price5, 2)
            self.write_log(['订单二成交'])
            self.close_time = self.count
        elif (self.status_1.status in self.not_active) and (self.status_2.status in self.not_active):
            self.cancel_order(self.close_order_id_1)
            self.cancel_order(self.close_order_id_2)
            if self.status_2.direction == Direction.LONG:
                self.close_one = self.sell(self.leg1_symbol, self.main_bid_price5, 2)
                self.close_two = self.cover(self.leg2_symbol, self.second_ask_price5, 1)
            else:
                self.close_two = self.sell(self.leg2_symbol, self.second_bid_price5, 1)
                self.close_one = self.cover(self.leg1_symbol, self.main_ask_price5, 2)
            self.write_log(['订单均未成交'])
            self.close_time = self.count
        else:
            if self.status_2.direction == Direction.LONG:
                self.open_position_1 = False
            else:
                 self.open_position_2 = False
            self.write_log(['订单成交'])
            self.waiting_close = False

    def on_tick(self, tick: TickData):
        # 记录当前时间
        ##self.tick_time = int(time.strftime("%H%M%S"))
        if  self.running == True:
            self.tick_time = float(datetime.strftime(tick.datetime, '%H%M%S.%f'))

            if 90000 <= self.tick_time <= 151000:
                self.complusary_closing_time = self.complusary_closing_time_day
                self.open_timing=143000
            if 210000 <= self.tick_time <= 240000:
                self.open_timing=223000
                self.complusary_closing_time = self.complusary_closing_time_night
            self.count += 1
        # 计算
            if 90000 <= self.tick_time and self.flag == False and self.waiting_open == False and self.waiting_close == False :
                if tick.vt_symbol == self.leg1_symbol:
                    self.main_time=float(datetime.strftime(tick.datetime,'%H%M%S.%f'))
                    self.deal_price1 =(tick.bid_price_1 + tick.ask_price_1) * self.coefficient_1 / 2
                    
                if tick.vt_symbol == self.leg2_symbol:                    
                    self.second_time=float(datetime.strftime(tick.datetime,'%H%M%S.%f'))
                    self.deal_price2 = (tick.bid_price_1  + tick.ask_price_1) * self.coefficient_2 / 2

                if -0.5<=self.second_time-self.main_time<=0.5:
                    self.main_price.append(self.deal_price1)
                    self.second_price.append(self.deal_price2)
                if len(self.main_price)>7200:
                    self.running=False
                    self.write_log('t检验无法通过检验停止检验')
                if 7200>len(self.main_price) > 4200:
                        self.linear_regression()
                        p = ts.adfuller(self.lm.resid, 1)[1]
                        self.write_log([p,np.std(self.lm.resid)])                                  
                        if p < 0.05 and np.std(self.lm.resid)>0.15:
                            self.flag = True
                            self.mean = np.mean(self.lm.resid)
                            self.std = np.std(self.lm.resid)

            if tick.vt_symbol == self.leg1_symbol:
                self.main_ask_bid = tick.ask_price_1 - tick.bid_price_1
                self.price1 = (tick.ask_price_1 + tick.bid_price_1) * self.coefficient_1/ 2
                self.main_bid_price = tick.bid_price_1
                self.main_ask_price = tick.ask_price_1
                self.main_bid_price5=tick.bid_price_5
                self.main_ask_price5=tick.ask_price_5
                self.main_time = float(datetime.strftime(tick.datetime, '%H%M%S.%f'))
            else:
                self.second_ask_bid = tick.ask_price_1 - tick.bid_price_1
                self.second_time = float(datetime.strftime(tick.datetime, '%H%M%S.%f'))
                self.second_bid_price = tick.bid_price_1
                self.second_ask_price = tick.ask_price_1
                self.second_ask_price5=tick.ask_price_5
                self.second_bid_price5=tick.bid_price_5
                self.price2 = (tick.ask_price_1 + tick.bid_price_1) *self.coefficient_2 / 2

            if -0.5<=self.second_time-self.main_time<=0.5:

                if self.flag == True and self.waiting_open == False and self.waiting_close == False:
                    self.resid = self.price2 - self.lm.params['price1'] * self.price1 - self.lm.params['Intercept']
                    self.resid2 = (self.resid - self.mean) / self.std
                    self.diff = (self.price2) - (self.price1)
                    self.write_log([self.resid2, self.diff])
                    if self.main_ask_bid + self.second_ask_bid < 3:
                        # 强制平仓
                        if self.tick_time > self.complusary_closing_time:
                            if self.open_position_1 == True:
                                self.closing_1(1)
                            if self.open_position_2 == True:
                                self.closing_2(1)
                        if self.resid2 > 2 and self.open_position_1 == False and self.tick_time <= self.open_timing:
                            self.trade_1(1)
                        if self.open_position_1 == True:
                            
                            if (self.open_resid_1 - self.resid2 > 3) or ((self.main_bid_price - self.open_a1)*120 + (self.open_b1 -
                                self.second_ask_price)*100 >= 700):
                                self.closing_1(1)
                            
                            self.write_log((self.main_bid_price - self.open_a1)*120 + (self.open_b1 -
                                self.second_ask_price)*100)
                        if self.resid2 < -2 and self.open_position_2 == False and self.tick_time <= self.open_timing:
                            self.trade_2(1)
                        if self.open_position_2 == True:
                            if (self.resid2 - self.open_resid_2 > 3) or ((self.second_bid_price - self.open_a2)*100 + (self.open_b2-self.main_ask_price)*120 >= 700):
                                self.closing_2(1)
                            
                            self.write_log((self.second_bid_price - self.open_a2)*100 + (self.open_b2-self.main_ask_price)*120)
                        if self.open_position_1 == True:
                            if 7200 < self.count - self.open_time < 14400:
                                if (self.main_bid_price - self.open_a1)*120 + (self.open_b1 -
                                self.second_ask_price)*100 >= 300:
                                    self.closing_1(1)
                            if self.count - self.open_time > 14400:
                                self.closing_1(1)
                                if self.flag2 == True:
                                    self.flag = False
                                    self.flag2 = False
                                    self.main_price.clear()
                                    self.second_price.clear()
                                else:
                                    self.running = False
                        if self.open_position_2 == True:
                            if 7200 < self.count - self.open_time < 14400:
                                if (self.second_bid_price - self.open_a2)*100 + (self.open_b2-self.main_ask_price)*120 >= 300:
                                    self.closing_2(1)
                            if self.count - self.open_time > 14400:
                                self.closing_2(1)
                                if self.flag2 == True:
                                    self.flag = False
                                    self.flag2 = False
                                    self.main_price.clear()
                                    self.second_price.clear()
                                else:
                                    self.running = False
                if self.waiting_open == True:
                    if self.count - self.open_time > 1000:
                        self.track_open()
                if self.waiting_close == True:
                    if self.count - self.close_time > 1000:
                        self.track_close()
            self.put_event()