# 先引入后面可能用到的包（package）
import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt
import backtrader as bt
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from datetime import datetime
import psycopg2
import DataFeed
# 正常显示画图时出现的中文和负号
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

engine = create_engine('postgresql+psycopg2://postgres:140888@localhost:5432/postgres')


# 设置token
# token = 'd7d08f66cd206206a4d85ee7bd0f2073da3caeeac531418f1194c1aa'
# ts.set_token(token)
# pro = ts.pro_api(token)


class MyStrategy(bt.Strategy):
    params = (('maperiod', 15),
              ('printlog', False),)


def __init__(self):
    # 指定价格序列
    self.dataclose = self.datas[0].close

    #  初始化交易指令、买卖价格和手续费
    self.order = None
    self.buyprice = None
    self.buycomm = None

    # 添加移动均线指标
    self.sma = bt.indicators.SimpleMovingAverage(
        self.datas[0], period=self.params.maperiod)


# 策略核心，根据条件执行买卖交易指令（必选）
def next(self):
    # 记录收盘价
    # self.log(f'收盘价, {dataclose[0]}')
    if self.order:
        # 检查是否有指令等待执行 
        return
    # 检查是否持仓  
    if not self.position:  #  没有持仓
        # 执行买入条件判断：收盘价格上涨突破15日均线
        if self.dataclose[0] > self.sma[0]:
            self.log('BUY CREATE, %.2f' % self.dataclose[0])
            # 执行买入
            self.order = self.buy()
        else:
            # 执行卖出条件判断：收盘价格跌破15日均线
            if self.dataclose[0] < self.sma[0]:
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
            # 执行卖出
            self.order = self.sell()


# 交易记录日志（可省略，默认不输出结果）
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
            self.log(f'买入:\n价格:{order.executed.price},\
            成本:{order.executed.value},\
            手续费:{order.executed.comm}')
            self.buyprice = order.executed.price
            self.buycomm = order.executed.comm
        else:
            self.log(f'卖出:\n价格：{order.executed.price},\
                成本: {order.executed.value},\
                手续费{order.executed.comm}')
            self.bar_executed = len(self)
    # 如果指令取消/交易失败, 报告结果
    elif order.status in [order.Canceled, order.Margin, order.Rejected]:
        self.log('交易失败')
    self.order = None


# 记录交易收益情况（可省略，默认不输出结果）
def notify_trade(self, trade):
    if not trade.isclosed:
        return
    self.log(f'策略收益：\n毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}')


# 回测结束后输出结果（可省略，默认输出结果）
def stop(self):
    self.log('(MA均线： %2d日) 期末总资金 %.2f' %
             (self.params.maperiod, self.broker.getvalue()), doprint=True)


def main(code, start, end='', startcash=10000, qts=500, com=0.001):
    # 创建主控制器
    cerebro = bt.Cerebro()
    # 导入策略参数寻优
    cerebro.optstrategy(MyStrategy, maperiod=range(3, 31))
    # 获取数据
    # df = ts.get_k_data(code, autype='qfq', start=start, end=end)
    df = DataFeed.get_stock_from_sql(code=code, start=start, end=end, engine=engine)
    df.index = pd.to_datetime(df.date)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    # 将数据加载至回测系统
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    # broker设置资金、手续费
    cerebro.broker.setcash(startcash)
    cerebro.broker.setcommission(commission=com)
    # 设置买入设置，策略，数量
    cerebro.addsizer(bt.sizers.FixedSize, stake=qts)
    print('期初总资金: %.2f' % cerebro.broker.getvalue())
    cerebro.run(maxcpus=1)
    print('期末总资金: %.2f' % cerebro.broker.getvalue())


def plot_stock(code, title, start, end):
    # dd = pro.index_daily(ts_code=code, start_date=start, end_date=end)
    dd = DataFeed.get_index_from_sql(code=code, start_date=start, end_date=end, engine=engine)
    dd.index = pd.to_datetime(dd.trade_date)
    dd.close.plot(figsize=(14, 6), color='r')
    plt.title(title + '价格走势\n' + start + ':' + end, size=1500)
    plt.annotate(f'期间累计涨幅:{(dd.close[-1] / dd.close[0] - 1) * 100:.2f}%', xy=(dd.index[-15], dd.close.mean()),
                 xytext=(dd.index[-16], dd.close.min()), bbox=dict(boxstyle='round,pad=0.5',
                                                                   fc='yellow', alpha=0.5),
                 arrowprops=dict(facecolor='green', shrink=0.05), fontsize=12)
    plt.show()


# 常用大盘指数
index = {'上证综指': '000001.SH', '深证成指': '399001.SZ', '沪深300': '000300.SH', '创业板指': '399006.SZ', '上证50': '000016.SH',
         '中证500': '000905.SH', '中小板指': '399005.SZ', '上证180': '000010.SH'}


# 获取代码
def get_code(name):
    df = DataFeed.get_code_from_sql(engine)
    codes = df.ts_code.values
    names = df.name.values
    stock = dict(zip(names, codes))
    # 合并指数和个股成一个字典
    stocks = dict(stock, **index)
    return stocks[name]


# 默认设定时间周期为当前时间往前推300个交易日
# 日期可以根据需要自己改动
def get_data(name, n=300):
    t = datetime.now()
    t0 = t - timedelta(n)
    start = t0.strftime('%Y%m%d')
    end = t.strftime('%Y%m%d')
    code = get_code(name)
    # 如果代码在字典index里，则取的是指数数据
    if code in index.values():
        # df = pro.index_daily(ts_code=code, start_date=start, end_date=end)
        df = DataFeed.get_index_from_sql(code=code, start_date=start, end_date=end, engine=engine)
        # 否则取的是个股数据，使用前复权
    else:
        # df = ts.pro_bar(ts_code=code, adj='qfq', start_date=start, end_date=end)
        df = DataFeed.get_stock_from_sql(code=code, start_date=start, end_date=end, engine=engine)
        # 将交易日期设置为索引值
        df.index = pd.to_datetime(df.trade_date)
        df = df.sort_index()

    return df


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


def kline_plot(name, ktype=0):
    df = cal_hadata(name)  # 画K线图数据
    # date = df.index.strftime('%Y%m%d').tolist()
    date = df.trade_date.tolist()
    # date = df['trade_date'].values.dt.strftime('%Y.%m.%d').tolist
    # date = df.index.tolist()
    if ktype == 0:
        k_value = df[['open', 'close', 'low', 'high']].values
    else:
        k_value = df[['ha_open', 'ha_close', 'ha_low', 'ha_high']].values

    # 引入pyecharts画图使用的是0.5.11版本，新版命令需要重写
    from pyecharts import Kline, Line, Bar, Scatter, Overlap
    kline = Kline(name + '行情走势')
    kline.add('日K线图', date, k_value, is_datazoom_show=True, is_splitline_show=False)
    # 加入5、20日均线
    df['ma20'] = df.close.rolling(20).mean()
    df['ma5'] = df.close.rolling(5).mean()
    line = Line()
    v0 = df['ma5'].round(2).tolist()
    v = df['ma20'].round(2).tolist()
    line.add('5日均线', date, v0, is_symbol_show=False, line_width=2)
    line.add('20日均线', date, v, is_symbol_show=False, line_width=2)
    # 成交量
    bar = Bar()
    bar.add('成交量', date, df['vol'], tooltip_tragger='axis', is_legend_show=False, is_yaxis_show=False,
            yaxis_max=5 * max(df['vol']))
    overlap = Overlap()
    overlap.add(kline)
    overlap.add(line, )
    overlap.add(bar, yaxis_index=1, is_add_yaxis=True)
    # display(HTML(overlap._repr_html_()))
    # warnings.filterwarnings("ignore")
    return overlap


# if __name__ == '__main__':
#     main(code, start, end='', startcash=10000, qts=500, com=0.001)


# plot_stock('000300.SH', '沪深300', '20200101', '20200330')
kline_plot('中国平安', ktype=0).render()
