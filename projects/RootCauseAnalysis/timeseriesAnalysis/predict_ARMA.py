#导入库

import pandas as pd 
import numpy as np
from  statsmodels.graphics.tsaplots import plot_acf,plot_pacf  #acf ,pacf 展示分析
from statsmodels.tsa.stattools import adfuller #adf检验库
from statsmodels.stats.diagnostic import acorr_ljungbox #随机性检验库
from statsmodels.tsa.arima_model import ARMA #ARMA，因为ARIMA最多二阶差分，所以实用性不大好，多将数据平稳处理后直接给ARMA调用
import matplotlib.pyplot as plt #Matplotlib图形展示库
import prettytable #导入表格库

#多次用到的表格

def pre_table(table_name ,table_rows):
   '''
    :param table_name: 表格名称，字符串列表
    :param talbe_row: 表格内容，嵌套列表
    :return: 展示表格对象
   '''
   table = prettytable.PrettyTable() #创建表格实例
   table.field_names  = table_name #定义表格列
   for i in table_rows:  # 循环读多条数据
        table.add_row(i)  #增加数据
   return table

#数据平稳处理，取log,也可以其他方法
def get_best_log(ts,max_log=5,rule1=True,rule2=True):
    '''
    :param ts: 时间序列数据，Series
    :param max_log: 最大log处理的次数，int型
    :param rule1: rule1规则布尔值
    :param rule2: rule2规则布尔值
    :return : 达到平稳处理的最佳次数值和处理后的时间序列
    '''
    if rule1 and rule2: #
        return 0,ts # 直接返回0和原始时间序列数据
    else: #只要有一个规则不满足
        for i in range(1,max_log): #循环做log处理
            ts = np.log(ts) # log处理
            lbvalue,pvalue1 = acorr_ljungbox(ts,lags=1) #白噪声检验结果
            adf ,pvalue2 ,usedlag,nobs,critical_values,icbest = adfuller(ts)#adf 检验
            rule_1 = (adf<critical_values['1%'] and adf < critical_values['5%']
                and adf < critical_values['10%'] and pvalue1< 0.01) #稳定性检验
            rule_2 = (pvalue2 < 0.05)  #白噪声检测
            rule_3 = (i<5)
        if rule_1 and rule_2 and rule_3: #如果同时满足条件
            print('The best log n is :(0)'.format(i)) #打印输出最佳次数
            return i,ts #返回最佳次数和处理后的时间序列
     
#还原经过平稳处理的数据
def rev_log(ts,log_n):
    '''
    :param ts: 经过log方法平稳处理的时间序列，Series类型
    :param log_n: log方法处理的次数，Series类型
    :return: 还原后的时间序列
    '''
    for i in range(1,log_n+1): 
        ts  = np.exp(ts) #log 还原方法
    return ts 


#平稳性检测
def adf_val(ts ,ts_title,acf_title,pacf_title):
    '''
    :param ts:时间序列数据，Series
    :param ts_title:时间序列图的标题名称，字符串
    :param acf_title:acf图的标题名称，字符串
    :param pacf_title:PACF图的标题名称，字符串
    :return: adf值，adf的p值，三种状态的检测值
    '''
    plt.figure()
    plt.plot(ts) 
    plt.title(ts_title)  
    plt.show()
    plot_acf(ts,lags=20,title=acf_title).show() #自相关检测
    plot_pacf(ts,lags=20,title=pacf_title).show() #偏相关检测
    adf,pvalue,usedlag,nobs,critical_values,icbest = adfuller(ts)  #平稳性检测
    table_name = ['adf','pvalue','usedlag','nobs','critical_values','icbest'] #表格列名列表
    table_rows = [[adf,pvalue,usedlag,nobs,critical_values,icbest]]
    adf_table = pre_table(table_name,table_rows)  #获得平稳性展示表格对象
    print('stochastic score') #打印标题
    print(adf_table)
    return adf,pvalue,critical_values

#白噪声（随机性）检验
def acorr_val(ts):
    '''
    :param ts: 时间序列 ，Series
    :return :白噪声检验的p值和展示数据表格对象
    '''
    lbvalue,pvalue = acorr_ljungbox(ts,lags=1)  #白噪声检验结果
    table_name = ['lbvalue','pvalue'] 
    table_rows = [[lbvalue,pvalue]]
    acorr_ljungbox_table = pre_table(table_name,table_rows) #获得白噪声检验展示表格对象
    print('stationarity score')
    print(acorr_ljungbox_table)
    return pvalue  #返回白噪声检验的p值和展示数据表格对象


# arma最优模型训练
def arma_fit(ts):
    '''
    :param ts: 时间序列数据，Series类型
    :return: 最优状态下的p值、q值、arma模型对象、pdq数据框和展示参数表格对象
    '''
    max_count = int(len(ts) / 10)  # 最大循环次数最大定义为记录数的10%
    bic = float('inf')  # 初始值为正无穷
    tmp_score = []  # 临时p、q、aic、bic和hqic的值的列表
    for tmp_p in range(max_count + 1):  # p循环max_count+1次
        for tmp_q in range(max_count + 1):  # q循环max_count+1次
            model = ARMA(ts, order=(tmp_p, tmp_q))  # 创建ARMA模型对象
            try:
                results_ARMA = model.fit(disp=-1, method='css')  # ARMA模型训练
            except:
                continue  # 遇到报错继续
            finally:
                tmp_aic = results_ARMA.aic  # 模型的获得aic
                tmp_bic = results_ARMA.bic  # 模型的获得bic
                tmp_hqic = results_ARMA.hqic  # 模型的获得hqic
                tmp_score.append([tmp_p, tmp_q, tmp_aic, tmp_bic, tmp_hqic])  # 追加每个模型的训练参数和结果
                if tmp_bic < bic:  # 如果模型bic小于最小值，那么获得最优模型ARMA的下列参数：
                    p = tmp_p  # 最优模型ARMA的p值
                    q = tmp_q  # 最优模型ARMA的q值
                    model_arma = results_ARMA  # 最优模型ARMA的模型对象
                    aic = tmp_bic  # 最优模型ARMA的aic
                    bic = tmp_bic  # 最优模型ARMA的bic
                    hqic = tmp_bic  # 最优模型ARMA的hqic
    pdq_metrix = np.array(tmp_score)  # 将嵌套列表转换为矩阵
    pdq_pd = pd.DataFrame(pdq_metrix, columns=['p', 'q', 'aic', 'bic', 'hqic'])  # 基于矩阵创建数据框
    table_name = ['p', 'q', 'aic', 'bic', 'hqic']  # 表格列名列表
    table_rows = [[p, q, aic, bic, hqic]]  # 表格行数据，嵌套列表
    parameter_table = pre_table(table_name, table_rows)  # 获得最佳ARMA模型结果展示表格对象
    print ('each p/q traning record')  # 打印标题
    print (pdq_pd)  # 打印输出每次ARMA拟合结果，包含p、d、q以及对应的AIC、BIC、HQIC
    print ('best p and q')  # 打印标题
    print (parameter_table)  # 输出最佳ARMA模型结果展示表格对象
    return model_arma  # 最优状态下的arma模型对象


# 模型训练和效果评估
def train_test(model_arma, ts, log_n, rule1=True, rule2=True):
    '''
    :param model_arma: 最优ARMA模型对象
    :param ts: 时间序列数据，Series类型
    :param log_n: 平稳性处理的log的次数，int型
    :param rule1: rule1规则布尔值，布尔型
    :param rule2: rule2规则布尔值，布尔型
    :return: 还原后的时间序列
    '''
    train_predict = model_arma.predict()  # 得到训练集的预测时间序列
    if not (rule1 and rule2):  # 如果两个条件有任意一个不满足
        train_predict = recover_log(train_predict, log_n)  # 恢复平稳性处理前的真实时间序列值
        ts = recover_log(ts, log_n)  # 时间序列还原处理
    ts_data_new = ts[train_predict.index]  # 将原始时间序列数据的长度与预测的周期对齐
    RMSE = np.sqrt(np.sum((train_predict - ts_data_new) ** 2) / ts_data_new.size)  # 求RMSE
    # 对比训练集的预测和真实数据
    plt.figure()  # 创建画布
    train_predict.plot(label='predicted data', style='--')  # 以虚线展示预测数据
    ts_data_new.plot(label='raw data')  # 以实线展示原始数据
    plt.legend(loc='best')  # 设置图例位置
    plt.title('raw data and predicted data with RMSE of %.2f' % RMSE)  # 设置标题
    plt.show()  # 展示图像
    return ts  # 返回还原后的时间序列


# 预测未来指定时间项的数据
def predict_data(model_arma, ts, log_n, start, end, rule1=True, rule2=True):
    '''
    :param model_arma: 最优ARMA模型对象
    :param ts: 时间序列数据，Series类型
    :param log_n: 平稳性处理的log的次数，int型
    :param start: 要预测数据的开始时间索引
    :param end: 要预测数据的结束时间索引
    :param rule1: rule1规则布尔值，布尔型
    :param rule2: rule2规则布尔值，布尔型
    :return: 无
    '''
    predict_ts = model_arma.predict(start=start, end=end)  # 预测未来指定时间项的数据
    print ('-----------predict data----------')  # 打印标题
    if not (rule1 and rule2):  # 如果两个条件有任意一个不满足
        predict_ts = rev_log(predict_ts, log_n)  # 还原数据
    print (predict_ts)  # 展示预测数据
    # 展示预测趋势
    plt.figure()  # 创建画布
    ts.plot(label='raw time series')  # 设置推向标签
    predict_ts.plot(label='predicted data', style='--')  # 以虚线展示预测数据
    plt.legend(loc='best')  # 设置图例位置
    plt.title('predicted time series')  # 设置标题
    plt.show()  # 展示图像
    return predict_ts

#预测未来指定个数数据
def forecast_data(model_arma, ts, log_n, forecast_num = 1 ,rule1=True, rule2=True):
    '''
    :param model_arma: 最优ARMA模型对象
    :param ts: 时间序列数据，Series类型
    :param log_n: 平稳性处理的log的次数，int型
    :param forecast_num: 预测未来多少数据
    :param rule1: rule1规则布尔值，布尔型
    :param rule2: rule2规则布尔值，布尔型
    :return: 无
    '''
    predict_ts = model_arma.forecast(forecast_num)[0]  # 预测未来指定时间项的数据
    
    print ('-----------predict data----------')  # 打印标题
    if not (rule1 and rule2):  # 如果两个条件有任意一个不满足
        predict_ts = rev_log(predict_ts, log_n)  # 还原数据
    print (predict_ts)  # 展示预测数据
    predict_ts=pd.Series(predict_ts)
    # 展示预测趋势
    plt.figure()  # 创建画布
    ts.plot(label='raw time series')  # 设置推向标签
    predict_ts.plot(label='predicted data', style='--')  # 以虚线展示预测数据
    plt.legend(loc='best')  # 设置图例位置
    plt.title('predicted time series')  # 设置标题
    plt.show()  # 展示图像
    return predict_ts

            