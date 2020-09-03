import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt
import time
from datetime import datetime, date, time, timedelta
from sqlalchemy import create_engine
from datetime import datetime
import psycopg2
import DataFeed
# 正常显示画图时出现的中文和负号
from pylab import mpl
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo
import logging

log = logging.getLogger(__name__)

# 以下引入脚本是个人的数据库文件，导入其他数据请注释掉
# from update_sql import update_sql
# 更新数据库
# update_sql(table_name='daily_data')

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

engine = create_engine('postgresql+psycopg2://postgres:140888@localhost:5432/postgres')

# 常用大盘指数
index = {'上证综指': '000001.SH', '深证成指': '399001.SZ', '沪深300': '000300.SH', '创业板指': '399006.SZ', '上证50': '000016.SH',
         '中证500': '000905.SH', '中小板指': '399005.SZ', '上证180': '000010.SH'}


def get_data(code, date='20200108'):
    begin_date = (datetime.strptime(date, "%Y%m%d") + timedelta(days=-200)).strftime("%Y%m%d")
    # code = DataFeed.get_stock_code()
    # 如果代码在字典index里，则取的是指数数据
    if code in index.values():
        data = DataFeed.get_index_from_sql(code=code, start_date=begin_date, end_date=date, engine=engine)
        data.index = pd.to_datetime(data.trade_date)
        data = data.sort_index()
        data['volume'] = data.vol
        data['datetime'] = pd.to_datetime(data.trade_date)
        data = data[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']]
        data = data.fillna(0)
    # 代码为股票
    else:
        data = DataFeed.get_stock_from_sql(code=code, start_date=begin_date, end_date=date, engine=engine)
        data.index = pd.to_datetime(data.trade_date)
        data = data.sort_index()
        data['volume'] = data.vol
        data['openinterest'] = 0
        data['datetime'] = pd.to_datetime(data.trade_date)
        data = data[['datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest']]
        data = data.fillna(0)
    return data


def get_code_list(date='20200108'):
    # 默认2020年开始回测
    # dd = pro.daily_basic(trade_date=date)
    dd = DataFeed.choose_stock_from_sql(date, engine=engine)
    x1 = dd.close < 100
    # 流通市值低于300亿大于50亿
    x2 = dd.circ_mv > 2500000
    x3 = dd.circ_mv < 3000000
    # 市盈率低于80
    x4 = dd.pe_ttm < 80
    # 股息率大于2%
    x5 = dd.dv_ttm > 3
    x = x1 & x2 & x3 & x4 & x5
    stock_list = dd[x].ts_code.values
    return stock_list


# 通过价格、市值、市盈率和股息率指标的设置，选择了24只个股进行量化回测。
print(len(get_code_list()))


class MyStrategy(bt.Strategy):
    #  策略参数
    params = dict(
        period=20,  #  均线周期
        look_back_days=30,
        printlog=False
    )

    def __init__(self):
        self.mas = dict()
        # 遍历所有股票,计算20日均线
        for data in self.datas:
            self.mas[data._name] = bt.ind.SMA(data.close, period=self.p.period)

    def next(self):
        # 计算截面收益率
        rate_list = []
        for data in self.datas:
            if len(data) > self.p.look_back_days:
                p0 = data.close[0]
                pn = data.close[-self.p.look_back_days]
                rate = (p0 - pn) / pn
                rate_list.append([data._name, rate])

        # 股票池   
        long_list = []
        sorted_rate = sorted(rate_list, key=lambda x: x[1], reverse=True)
        long_list = [i[0] for i in sorted_rate[:10]]

        #  得到当前的账户价值
        total_value = self.broker.getvalue()
        p_value = total_value * 0.9 / 10
        for data in self.datas:
            # 获取仓位
            pos = self.getposition(data).size
            if not pos and data._name in long_list and \
                    self.mas[data._name][0] > data.close[0]:
                size = int(p_value / 100 / data.close[0]) * 100
                self.buy(data=data, size=size)

            if pos != 0 and data._name not in long_list or \
                    self.mas[data._name][0] < data.close[0]:
                self.close(data=data)

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()},{txt}')

    # 记录交易执行情况（可省略，默认不输出结果）
    def notify_order(self, order):
        #  如果order为submitted/accepted,返回空
        if order.status in [order.Submitted, order.Accepted]:
            return
        #  如果order为buy/sell executed,报告价格结果
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入:\n价格:{order.executed.price:.2f},\
                成本:{order.executed.value:.2f},\
                手续费:{order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'卖出:\n价格：{order.executed.price:.2f},\
                成本: {order.executed.value:.2f},\
                手续费{order.executed.comm:.2f}')

            self.bar_executed = len(self)

        #  如果指令取消/交易失败, 报告结果
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('交易失败')
        self.order = None

    # 记录交易收益情况（可省略，默认不输出结果）
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'策略收益：\n毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}')


class BaseStrategy(bt.Strategy):
    #  策略参数
    params = dict(
        period=20,  #  均线周期
        look_back_days=30,
        printlog=False
    )

    def __init__(self):
        self.mas = dict()
        # 遍历所有股票,计算20日均线
        for data in self.datas:
            self.mas[data._name] = bt.ind.SMA(data.close, period=self.p.period)

    def next(self):
        # 计算截面收益率
        rate_list = []
        for data in self.datas:
            if len(data) > self.p.look_back_days:
                p0 = data.close[0]
                pn = data.close[-self.p.look_back_days]
                rate = (p0 - pn) / pn
                rate_list.append([data._name, rate])

        # 股票池   
        long_list = []
        sorted_rate = sorted(rate_list, key=lambda x: x[1], reverse=True)
        long_list = [i[0] for i in sorted_rate[:10]]

        #  得到当前的账户价值
        total_value = self.broker.getvalue()
        p_value = total_value * 0.9 / 10
        for data in self.datas:
            # 获取仓位
            pos = self.getposition(data).size
            if not pos and data._name in long_list and \
                    self.mas[data._name][0] > data.close[0]:
                size = int(p_value / 100 / data.close[0]) * 100
                self.buy(data=data, size=size)

            if pos != 0 and data._name not in long_list or \
                    self.mas[data._name][0] < data.close[0]:
                self.close(data=data)

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()},{txt}')

    # 记录交易执行情况（可省略，默认不输出结果）
    def notify_order(self, order):
        #  如果order为submitted/accepted,返回空
        if order.status in [order.Submitted, order.Accepted]:
            return
        #  如果order为buy/sell executed,报告价格结果
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入:\n价格:{order.executed.price:.2f},\
                成本:{order.executed.value:.2f},\
                手续费:{order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'卖出:\n价格：{order.executed.price:.2f},\
                成本: {order.executed.value:.2f},\
                手续费{order.executed.comm:.2f}')

            self.bar_executed = len(self)

        #  如果指令取消/交易失败, 报告结果
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('交易失败')
        self.order = None

    # 记录交易收益情况（可省略，默认不输出结果）
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'策略收益：\n毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}')


# 均线交叉策略
class SMACross(bt.Strategy):
    params = dict(
        sma_lower=10,  # period for lower SMA
        sma_higher=50,  # period for higher SMA
    )

    def __init__(self):
        # 10日SMA计算
        sma1 = bt.ind.SMA(period=self.p.sma_lower)
        # 50日SMA计算
        sma2 = bt.ind.SMA(period=self.p.sma_higher)

        # 均线交叉, 1是上穿，-1是下穿
        self.crossover = bt.ind.CrossOver(sma1, sma2)

    def next(self):
        close = self.data.close[0]
        date = self.data.datetime.date(0)
        if not self.position:
            if self.crossover > 0:
                log.info("buy created at {} - {}".format(date, close))
                self.buy()          # 买入

        elif self.crossover < 0:
            log.info("sell created at {} - {}".format(date, close))
            self.close()            # 卖出


# 计算Heikin Ashi蜡烛线
def cal_hadata(name):
    df = get_data(name)
    # 计算修正版K线
    df['ha_close'] = (df.close + df.open + df.high + df.low) / 4.0
    ha_open = np.zeros(df.shape[0])
    ha_open[0] = df.open[0]
    for i in range(1, df.shape[0]):
        ha_open[i] = (ha_open[i - 1] + df['ha_close'][i - 1]) / 2
        df.insert(1, 'ha_open', ha_open)
        df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
        df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
        df = df.iloc[1:]
        return df


def plot_result(cerebro, cerebro2, s='中国平安', ktype=0):

    if ktype == 0:
        df = get_data(s)
        k_value = df[['open', 'close', 'low', 'high']].values
    else:
        df = cal_hadata(s)  # 画K线图数据
        k_value = df[['ha_open', 'ha_close', 'ha_low', 'ha_high']].values

    date = df.index.strftime('%Y%m%d').tolist()

    # 引入pyecharts画图使用的是0.5.11版本，新版命令需要重写
    from pyecharts import Kline, Line, Bar, EffectScatter, Overlap, Grid, Page
    # K线
    kline = Kline(s + '行情走势')
    kline.add('日K线图', date, k_value, is_datazoom_show=True, is_splitline_show=False)
    # 加入5、20日均线
    df['ma20'] = df.close.rolling(20).mean()
    df['ma5'] = df.close.rolling(5).mean()
    line = Line()
    v0 = df['ma5'].round(2).tolist()
    v = df['ma20'].round(2).tolist()
    line.add('5日均线', date, v0, is_symbol_show=False, line_width=2)
    line.add('20日均线', date, v, is_symbol_show=False, line_width=2)
    # 买卖点
    scatter = EffectScatter('买卖点')
    buy_order_date = []
    buy_order_price = []
    sell_order_date = []
    sell_order_price = []
    total_cash = []
    total_benifit = []
    for order in cerebro[0].broker.orders:
        if order.isbuy():
            buy_date = bt.num2date(order.dteos).isoformat()
            buy_date = ''.join(buy_date.split('-'))[0:8]
            buy_order_date.append(buy_date)
            buy_order_price.append(int(order.executed.price))
            total_cash.append(int(cerebro[0].broker.startingcash - order.executed.value))
            total_benifit.append(int(cerebro[0].broker.cash) + int(order.executed.value))
        else:
            sell_date = bt.num2date(order.dteos).isoformat()
            sell_date = ''.join(sell_date.split('-'))[0:8]
            sell_order_date.append(sell_date)
            sell_order_price.append(int(order.executed.price))
            total_cash.append(int(cerebro[0].broker.startingcash + order.executed.value))
            total_benifit.append(int(cerebro[0].broker.cash) + int(order.executed.value))

    # 设置散点的形状（cricle,rect,pin,triangle）
    scatter.add('Buy', buy_order_date, buy_order_price, symbol='triangle', symbol_size=16, color='aaa')
    scatter.add('Sell', sell_order_date, sell_order_price, symbol='cricle', symbol_size=16,  color='111')
    # 加说明
    # statistic = "收益率：" + str(cerebro.broker.getvalue()/(cerebro.broker.startingcash))

    # 成交量
    bar = Bar()
    bar.add('成交量', date, df['volume'], tooltip_tragger='axis', is_legend_show=False, is_yaxis_show=False,
            yaxis_max=5 * max(df['volume']))
    overlap = Overlap()
    overlap.add(kline)
    overlap.add(line, )
    overlap.add(bar, yaxis_index=1, is_add_yaxis=True)
    overlap.add(scatter)
    line2 = Line()
    # line2.add('账户现金', sorted(buy_order_date+sell_order_date), total_cash, is_symbol_show=True, line_width=2)
    result_date = cerebro[0].analyzers.transactions.get_analysis()
    line2.add('账户现金', result_date['date'],  (cerebro[0].analyzers.transactions.get_analysis())['value'], is_symbol_show=True, line_width=2)
    line2.add('沪深300账户现金', (cerebro2[0].analyzers.transactions.get_analysis())['date'],  (cerebro2[0].analyzers.transactions.get_analysis())['value'], is_symbol_show=True, line_width=2)
    page = Page()
    page.add(overlap)
    page.add(line2)
    return page.render(s+'.html')


# 加载数据
cerebro = bt.Cerebro()
# 自动按照条件选取一组股票进行回测
for s in get_code_list():
    feed = bt.feeds.PandasData(dataname=get_data(s))
    cerebro.adddata(feed, name=s)

feed = bt.feeds.PandasData(dataname=get_data(s))
cerebro.adddata(feed, name=s)
cerebro.broker.get_notification()
# 回测设置，起始资金
startcash = 100000.0
cerebro.broker.setcash(startcash)
#  设置佣金为千分之一
cerebro.broker.setcommission(commission=0.001)
#  添加策略
cerebro.addstrategy(MyStrategy, printlog=True)
# 添加分析器，夏普比例
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name = 'SharpeRatio')
# 最大回撤
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DrawDown')
# 年收益率
cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annual_returns")
# 交易记录表
cerebro.addanalyzer(bt.analyzers.Transactions, _name="transactions")
results = cerebro.run()
# 获取回测结束后的总资金
portvalue = cerebro.broker.getvalue()
pnl = portvalue - startcash

# 打印Analyzer结果到日志
for result in results:
    annual_returns = result.analyzers.annual_returns.get_analysis()
    log.info("annual returns:".format('%.2f'))
    for year, ret in annual_returns.items():
            log.info("\t {} {}%, ".format(year, round(ret * 100, 2)))

    draw_down = result.analyzers.DrawDown.get_analysis()
    log.info(
            "drawdown={drawdown}%, moneydown={moneydown}, drawdown len={len}, "
            "max.drawdown={max.drawdown}, max.moneydown={max.moneydown}, "
            "max.len={max.len}".format(**draw_down))

    transactions = result.analyzers.transactions.get_analysis()
    log.info("transactions")


# 打印结果
print(f'总资金: {round(portvalue, 2)}')
print(f'净收益: {round(pnl, 2)}')
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
print('SR夏普率:', results[0].analyzers.SharpeRatio.get_analysis())
# 最大回撤
print('最大回撤DW:', results[0].analyzers.DrawDown.get_analysis())


# 选取一只股票进行回测
cerebro2 = bt.Cerebro()
baseline = '000300.SH'
feed = bt.feeds.PandasData(dataname=get_data(baseline))
cerebro2.adddata(feed, name=baseline)
cerebro2.broker.get_notification()
# 回测设置
cerebro2.broker.setcash(startcash)
#  设置佣金为千分之一
cerebro2.broker.setcommission(commission=0.001)
#  添加策略
cerebro2.addstrategy(BaseStrategy, printlog=True)
results2 = cerebro2.run()
# 获取回测结束后的总资金
portvalue2 = cerebro2.broker.getvalue()
pnl2 = portvalue2 - startcash


# 打印结果
print(f'沪深300 总资金: {round(portvalue2, 2)}')
print(f'沪深300 净收益: {round(pnl2, 2)}')
print('沪深300Final Portfolio Value: %.2f' % cerebro2.broker.getvalue())


# 绘制图像
b = Bokeh(style='bar',plot_mode='multi',scheme=Tradimo())


# cerebro.plot(b)
# cerebro.plot()
plot_result(results, results2, s)
