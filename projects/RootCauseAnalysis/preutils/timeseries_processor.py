import datetime
import time
import pandas as pd


def timestamp_datetime(ts):
    """timestamp转datetime 
    :param ts:timestamp数据
    :return: datetime格式数据
    """
    if isinstance(ts, (int, float, str)):
        try:
            ts = int(ts)
        except ValueError:
            raise

        if len(str(ts)) == 13:
            ts = int(ts / 1000.0)
        if len(str(ts)) != 10:
            raise ValueError
    else:
        raise ValueError()
    return datetime.datetime.fromtimestamp(ts) #.strftime("%Y/%m/%d %H:%M:%S")


def datetime_timestamp(dt, type='ms'):
    """datetime2timestamp
    :param dt:datetime格式数据
    :param type: 时间戳类型，ms即millisecond timestamp
    :return: timestamp格式数据
    """
    if isinstance(dt, str):
        try:
            if len(dt) == 10:
                dt = datetime.datetime.strptime(dt, '%Y/%m/%d')
            elif len(dt) == 19:
                dt = datetime.datetime.strptime(dt, '%Y/%m/%d %H:%M:%S')
            elif len(dt) == 18:
                dt = datetime.datetime.strptime(dt, '%Y/%m/%d %H:%M:%S')
            elif len(dt) == 17:
                dt = datetime.datetime.strptime(dt, '%Y/%m/%d %H:%M:%S')
            else:
                raise ValueError()
        except ValueError as e:
            raise ValueError(
                "{0} is not supported datetime format." \
                "dt Format example: 'yyyy/mm/dd' or yyyy/mm/dd HH:MM:SS".format(dt)
            )

    if isinstance(dt, time.struct_time):
        dt = datetime.datetime.strptime(time.stftime('%Y/%m/%d %H:%M:%S', dt), '%Y/%m/%d %H:%M:%S')

    if isinstance(dt, datetime.datetime):
        if type == 'ms':
            ts = int(dt.timestamp()) * 1000
        else:
            ts = int(dt.timestamp())
    else:
        raise ValueError(
            "dt type not supported. dt Format example: 'yyyy/mm/dd' or yyyy/mm/dd HH:MM:SS"
        )
    return ts

def get_tsdata(filepath,itemid,bomcid,indexname):
    """从csv中获取kpi数据
    :param filepath:文件路径
    :param itemid: 大类
    :param bomcid: 网元id 
    :param indexname: 指标名称
    :return: timeseriesdata
    dta:series 格式的时间序列
    check_value:需要检测的值
    """
    alldata = pd.read_csv(filepath)
    #cols = list(alldata.columns.values)
    data = alldata[(alldata['itemid']==itemid) & (alldata['bomc_id']==bomcid) & (alldata['name']==indexname)]
    
    timestamp_list = []
    value_list = []
    for  timestamp, value in zip(data['timestamp'], data['value']): 
        ts = timestamp_datetime(timestamp)
        timestamp_list.append(ts)
        value_list.append(value)
    dta = pd.Series(value_list[:-1]) #转为series   
    dta = dta.fillna(dta.mean())  #用均值填充空值
    dta.index = pd.Index(timestamp_list[:-1])
    dta = dta.sort_index()
    #dta.index = pd.DatetimeIndex(dta.index)  #时间为索引
    # 最后一个点为检测点
    #check_value = value_list[-1]
    return dta#, check_value