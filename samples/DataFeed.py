import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import psycopg2
import time

# 正常显示画图时出现的中文和负号
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 设置token
token = 'd7d08f66cd206206a4d85ee7bd0f2073da3caeeac531418f1194c1aa'
ts.set_token(token)
pro = ts.pro_api()
engine = create_engine('postgresql+psycopg2://postgres:140888@localhost:5432/postgres')

# 常用大盘指数
index = {'上证综指': '000001.SH', '深证成指': '399001.SZ', '沪深300': '000300.SH', '创业板指': '399006.SZ', '上证50': '000016.SH',
         '中证500': '000905.SH', '中小板指': '399005.SZ', '上证180': '000010.SH'}


# 数据获取函数，默认时间可以随时改动
# 如果报错，把tushare升级到最新
def get_stock_data(code, start='20190101', end='20190131'):
    dd = ts.pro_bar(ts_code=code, adj='qfq', start_date=start, end_date=end)
    return dd


# 交易代码获取函数，获取最新交易日的代码
# 获取当前交易日最新的股票代码和简称
def get_stock_code():
    # codes = pro.stock_basic(list_status='L', fields='ts_code').values
    codes = pro.stock_basic(list_status='L').ts_code.values
    # 剔除st股
    # codes = codes[-codes['name'].apply(lambda x: x.startswith('*ST'))]
    # codes = codes[-codes['name'].apply(lambda x: x.startswith('ST'))]
    return codes


def insert_sql(data, db_name, if_exists='append'):
    # 使用try...except..continue避免出现错误，运行崩溃
    try:
        # access the database created
        data.to_sql(db_name, engine, index=False, if_exists=if_exists)
        print(data + '写入数据库成功')
    except:
        pass


# 股票每日行情数据获取
def get_stock_daily(cal_date='20200101'):
    for _ in range(3):
        try:
            if cal_date:
                # df = self.pro.daily(ts_code=ts_code, trade_date=trade_date)
                df = pro.daily_basic(trade_date=cal_date)
            else:
                # df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                df = pro.daily_basic(trade_date=cal_date)
        except:
            time.sleep(1)
        else:
            return df


# 股票每日行情，详细数据更新
def update_daily_detail(start='20200201', end='20200828', db_name='stock_data_detail'):
    # 获取20200101～20200401之间所有有交易的日期
    cal = pro.trade_cal(exchange='SSE', is_open='1',
                        start_date=start,
                        end_date=end,
                        fields='cal_date')

    print(cal.head())

    for date in cal['cal_date'].values:
        df = get_stock_daily(cal_date=date)
        insert_sql(df, db_name)

    return


# 指数每日行情
def update_daily_index(start='20200101', end='20200131', db_name='index_data', index='000300.SH'):
    for code in index:
        data = pro.index_daily(ts_code=code, start_date=start, end_date=end)
        insert_sql(data, db_name)

    return


# 读取整张表数据
# df=pd.read_sql('stock_data',engine)
# print(len(df))
# 选取ts_code=000001.SZ的股票数据
# df=pd.read_sql("select * from stock_data where ts_code='000001.SZ'",engine)
# print(len(df))


# 构建一个数据更新函数，可以下载和插入其他时间周期的数据。2018年1月1日至2019年4月25日，数据就已达到108万条。
# 更新数据或下载其他期间数据
def update_stock_daily(start, end, db_name):
    for code in get_stock_code():
        data = get_stock_data(code, start, end)
        insert_sql(data, db_name)
    print(f'{start}:{end}期间数据已成功更新')


# 整体每日更新函数入口
def update_daily(db_name='stock_data', db_name2='stock_data_detail', index='000300.SH'):
    from datetime import datetime, timedelta
    today = time.strftime("%Y%m%d", time.localtime())
    max_date = pd.read_sql("select max(trade_date) from stock_data", engine).values
    max_date = max_date[0, 0]
    if today == max_date:
        print('已经更新到最新数据')
    else:
        # 日线基本行情, daily_basic
        update_stock_daily(max_date, today, db_name)
        # 日线详细行情,pro_bar
        update_daily_detail(max_date, today, db_name2)
        # 指数更新
        update_daily_index(start=max_date, end=today, index=index.values())
        print('%s更新了记录' % today)


# update_daily()


# 获取基本信息
def stock_basic(db_name):
    data = pro.stock_basic(exchange='', list_status='L')
    insert_sql(data, db_name)
    return


# stock_basic(db_name='stock_basic')


# 对外接口，从数据库中获取股票数据
def get_stock_from_sql(code, start_date, end_date, engine):
    # select * from stock_data left JOIN stock_data_detail using (ts_code,trade_date) where
    # stock_data.ts_code='000001.SZ' and stock_data.trade_date between '20190425' and '20200425' order by
    # stock_data.ts_code
    data = pd.read_sql(f"select * from stock_data left JOIN stock_data_detail using (ts_code,trade_date,close)"
                       f"where stock_data.ts_code='{code}' and stock_data. "
                       f"trade_date between '{start_date}' and '{end_date}' order by stock_data.trade_date",
                       engine)
    # 计算20日均线
    # data['ma20']=data.close.rolling(20).mean()
    return data


# 对外接口，从数据库中获取指数数据
def get_index_from_sql(code, start_date, end_date, engine):
    data = pd.read_sql(
        f"select * from index_data where ts_code='{code}' and trade_date between '{start_date}' and '{end_date}' order by trade_date",
        engine)
    return data


# 对外接口，从数据库中获取code
def get_code_from_sql(engine):
    # select DISTINCT stock_data.ts_code, stock_basic.name from stock_data
    # INNER JOIN stock_basic
    # on (stock_data.ts_code)=(stock_basic.ts_code)
    # order by  stock_data.ts_code
    code = pd.read_sql(f"select DISTINCT stock_data.ts_code, stock_basic.name from stock_data INNER JOIN stock_basic "
                       f"on (stock_data.ts_code)=(stock_basic.ts_code) order by  stock_data.ts_code",
                       engine)
    return code

# 对外接口，从数据库中获取code
def choose_stock_from_sql(trade_date, engine):
    code = pd.read_sql(f"select * from stock_data_detail where trade_date='{trade_date}'",
                       engine)
    return code



def plot_data(condition, title):
    from pyecharts import Bar
    from sqlalchemy import create_engine
    engine1 = create_engine('postgresql+psycopg2://postgres:140888@localhost:5432/postgres')
    data = pd.read_sql("select * from stock_data where+" + condition, engine1)
    count_ = data.groupby('trade_date')['ts_code'].count()
    attr = count_.index
    v1 = count_.values
    bar = Bar(title, title_text_size=15)
    bar.add('', attr, v1, is_splitline_show=False, linewidth=2)
    return bar

# 查詢股價低於2元個股數據分布
# c1 = "close<2"
# t1 = "股價低於2元個股時間分布"
# plot_data(c1, t1)
#
# c2 = "pct_chg>9.5"
# t2 = "股價漲幅超過9.5%個股時間分布"
# plot_data(c2, t2)
#
# c3 = "pct_chg<-9.5"
# t3 = "股價跌幅超過-9.5%個股時間分布"
# plot_data(c3, t3)
