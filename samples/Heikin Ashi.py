import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt
# %matplotlib inline
# 正常显示画图时出现的中文和负号
from pylab import mpl
from datetime import datetime, timedelta

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 设置token
token = 'd7d08f66cd206206a4d85ee7bd0f2073da3caeeac531418f1194c1aa'
ts.set_token(token)
pro = ts.pro_api(token)

# 常用大盘指数
index = {'上证综指': '000001.SH', '深证成指': '399001.SZ', '沪深300': '000300.SH', '创业板指': '399006.SZ', '上证50': '000016.SH',
         '中证500': '000905.SH', '中小板指': '399005.SZ', '上证180': '000010.SH'}


# 获取代码
def get_code(name):
    df = pro.stock_basic(exchange='')
    codes = df.ts_code.values
    names = df.name.values
    stock = dict(zip(names, codes))
    # 合并指数和个股成一个字典
    stocks = dict(stock, **index)
    return stocks[name]
    # 默认设定时间周期为当前时间往前推300个交易日
    # #日期可以根据需要自己改动


def get_data(name, n=300):

    t = datetime.now()
    t0 = t - timedelta(n)
    start = t0.strftime('%Y%m%d')
    end = t.strftime('%Y%m%d')
    code = get_code(name)
    # 如果代码在字典index里，则取的是指数数据
    if code in index.values():
        df = pro.index_daily(ts_code=code, start_date=start, end_date=end)
        # 否则取的是个股数据，使用前复权
    else:
        df = ts.pro_bar(ts_code=code, adj='qfq', start_date=start, end_date=end)
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
        # df.insert(1, 'ha_open', ha_open[i])
        df['ha_open'] = ha_open[i]
        df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
        df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
        # df = df.iloc[1:]

    return df


def plot_stock(code, title, start, end):
    dd = pro.index_daily(ts_code=code, start_date=start, end_date=end)
    dd.index = pd.to_datetime(dd.trade_date)
    dd.close.plot(figsize=(14, 6), color='r')
    plt.title(title + '价格走势\n' + start + ':' + end, size=1500)
    plt.annotate(f'期间累计涨幅:{(dd.close[-1] / dd.close[0] - 1) * 100:.2f}%', xy=(dd.index[-15], dd.close.mean()),
                 xytext=(dd.index[-16], dd.close.min()), bbox=dict(boxstyle='round,pad=0.5',
                                                                    fc='yellow', alpha=0.5),
                 arrowprops=dict(facecolor='green', shrink=0.05), fontsize=12)
    plt.show()


def kline_plot(name, ktype=0):
    df = cal_hadata(name)  # 画K线图数据
    # date = df.index.strftime('%Y%m%d').tolist()
    date = df.trade_date.tolist()
    #date = df['trade_date'].values.dt.strftime('%Y.%m.%d').tolist
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

    return overlap

# plot_stock('000300.SH', '沪深300', '20200101', '20200825')

kline_plot('沪深300', ktype=0).render()
