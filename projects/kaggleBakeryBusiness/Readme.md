# BakeryBusinessModel

## 背景
这是来自kaggle的一个数据集，主要是来自一个面包店的两万多条交易记录。提供的数据可以用于商业分析，产品营销推荐等，没有task，一般是自己定义的问题。https://www.kaggle.com/sulmansarwar/transactions-from-a-bakery  

每条记录有日期，时间、交易项等，交易项是一个商品，比如Tea、bread等，也就是表示每个商品被销售的时间。
这里主要学习关联规则，参考https://www.kaggle.com/bbhatt001/bakery-business-model-association-rules  
主要熟悉mlxtend机器学习扩展库和networkx网络图算法库

## 数据探索分析


```python
import numpy as np 
import pandas as pd 
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules,fpgrowth
import networkx as nx
import warnings
warnings.filterwarnings('ignore')  #忽略警告
```


```python
data=pd.read_csv('input//BreadBasket_DMS.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>Transaction</th>
      <th>Item</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2016-10-30</td>
      <td>09:58:11</td>
      <td>1</td>
      <td>Bread</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2016-10-30</td>
      <td>10:05:34</td>
      <td>2</td>
      <td>Scandinavian</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2016-10-30</td>
      <td>10:05:34</td>
      <td>2</td>
      <td>Scandinavian</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2016-10-30</td>
      <td>10:07:57</td>
      <td>3</td>
      <td>Hot chocolate</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2016-10-30</td>
      <td>10:07:57</td>
      <td>3</td>
      <td>Jam</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.Item.unique()  #查看多少种商品
```




    array(['Bread', 'Scandinavian', 'Hot chocolate', 'Jam', 'Cookies',
           'Muffin', 'Coffee', 'Pastry', 'Medialuna', 'Tea', 'NONE',
           'Tartine', 'Basket', 'Mineral water', 'Farm House', 'Fudge',
           'Juice', "Ella's Kitchen Pouches", 'Victorian Sponge', 'Frittata',
           'Hearty & Seasonal', 'Soup', 'Pick and Mix Bowls', 'Smoothies',
           'Cake', 'Mighty Protein', 'Chicken sand', 'Coke',
           'My-5 Fruit Shoot', 'Focaccia', 'Sandwich', 'Alfajores', 'Eggs',
           'Brownie', 'Dulce de Leche', 'Honey', 'The BART', 'Granola',
           'Fairy Doors', 'Empanadas', 'Keeping It Local', 'Art Tray',
           'Bowl Nic Pitt', 'Bread Pudding', 'Adjustment', 'Truffles',
           'Chimichurri Oil', 'Bacon', 'Spread', 'Kids biscuit', 'Siblings',
           'Caramel bites', 'Jammie Dodgers', 'Tiffin', 'Olum & polenta',
           'Polenta', 'The Nomad', 'Hack the stack', 'Bakewell',
           'Lemon and coconut', 'Toast', 'Scone', 'Crepes', 'Vegan mincepie',
           'Bare Popcorn', 'Muesli', 'Crisps', 'Pintxos', 'Gingerbread syrup',
           'Panatone', 'Brioche and salami', 'Afternoon with the baker',
           'Salad', 'Chicken Stew', 'Spanish Brunch',
           'Raspberry shortbread sandwich', 'Extra Salami or Feta',
           'Duck egg', 'Baguette', "Valentine's card", 'Tshirt',
           'Vegan Feast', 'Postcard', 'Nomad bag', 'Chocolates',
           'Coffee granules ', 'Drinking chocolate spoons ',
           'Christmas common', 'Argentina Night', 'Half slice Monster ',
           'Gift voucher', 'Cherry me Dried fruit', 'Mortimer', 'Raw bars',
           'Tacos/Fajita'], dtype=object)




```python
#看有多少无效数据
data[data['Item']=='NONE'].count()
```




    Date           786
    Time           786
    Transaction    786
    Item           786
    dtype: int64



去掉空值


```python
data_drop = data.drop(data[data['Item']=='NONE'].index)
```


```python
data_drop.shape
```




    (20507, 4)




```python
data_drop['Item'].nunique()  #商品种类
```




    94



查看数据集里销售量排名前十的商品有那几个,并画条形图



```python
data_drop['Item'].value_counts().sort_values(ascending=False).head(10)
```




    Coffee           5471
    Bread            3325
    Tea              1435
    Cake             1025
    Pastry            856
    Sandwich          771
    Medialuna         616
    Hot chocolate     590
    Cookies           540
    Brownie           379
    Name: Item, dtype: int64




```python
fig, ax=plt.subplots(figsize=(6,4))
data_drop['Item'].value_counts().sort_values(ascending=False).head(10).plot(kind='bar')
plt.ylabel('Number of transactions')
plt.xlabel('Items')
#ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.title('Best sellers')
```




    Text(0.5, 1.0, 'Best sellers')




![png](output_13_1.png)


这个统计说明，面包店主要卖的好的是咖啡、面包、茶和蛋糕等，当然这也是显而易见的，我们需要更深入的分析，比如什么日期段，什么时间段什么商品卖的好，然后看哪些一起卖好，也就是什么时间摆哪些组合，这就是店长需要的数据分析，以根据具体情况设置商品的销售和促销活动等等。  
我们先按月份看，各个月卖的好的商品有什么，这需要对date数据特征做特征提取，取出月日。相对于月份，我们还关注每周星期几的销售情况，比如周几卖什么比较多。所以要提取星期几作为新特征



```python
data_drop['Date'].head()
data_drop['Date_Time'] = pd.to_datetime(data_drop['Date']+' '+data_drop['Time'])  #转为日期加时间的标准格式
data_drop['Day']=data_drop['Date_Time'].dt.day_name()      #获取日期，这里是星期几的英文
data_drop['Month']=data_drop['Date_Time'].dt.month          #获取月份（数值）
data_drop['Month_name']=data_drop['Date_Time'].dt.month_name()     #获取月份的英文名
data_drop['Year']=data_drop['Date_Time'].dt.year             # 获取年份数字
data_drop['Year_Month']=data_drop['Year'].apply(str)+' '+data_drop['Month_name'].apply(str)  #转为字符串
data_drop1 = data_drop.drop(['Date'], axis=1 )       #去掉原特征字段

 
```


```python
data_drop1.head()
 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>Transaction</th>
      <th>Item</th>
      <th>Date_Time</th>
      <th>Day</th>
      <th>Month</th>
      <th>Month_name</th>
      <th>Year</th>
      <th>Year_Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>09:58:11</td>
      <td>1</td>
      <td>Bread</td>
      <td>2016-10-30 09:58:11</td>
      <td>Sunday</td>
      <td>10</td>
      <td>October</td>
      <td>2016</td>
      <td>2016 October</td>
    </tr>
    <tr>
      <td>1</td>
      <td>10:05:34</td>
      <td>2</td>
      <td>Scandinavian</td>
      <td>2016-10-30 10:05:34</td>
      <td>Sunday</td>
      <td>10</td>
      <td>October</td>
      <td>2016</td>
      <td>2016 October</td>
    </tr>
    <tr>
      <td>2</td>
      <td>10:05:34</td>
      <td>2</td>
      <td>Scandinavian</td>
      <td>2016-10-30 10:05:34</td>
      <td>Sunday</td>
      <td>10</td>
      <td>October</td>
      <td>2016</td>
      <td>2016 October</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10:07:57</td>
      <td>3</td>
      <td>Hot chocolate</td>
      <td>2016-10-30 10:07:57</td>
      <td>Sunday</td>
      <td>10</td>
      <td>October</td>
      <td>2016</td>
      <td>2016 October</td>
    </tr>
    <tr>
      <td>4</td>
      <td>10:07:57</td>
      <td>3</td>
      <td>Jam</td>
      <td>2016-10-30 10:07:57</td>
      <td>Sunday</td>
      <td>10</td>
      <td>October</td>
      <td>2016</td>
      <td>2016 October</td>
    </tr>
  </tbody>
</table>
</div>




```python
#可视化显示面包店不同月份的销售情况
data_drop1.groupby('Year_Month')['Item'].count().plot(kind='bar')
plt.ylabel('Number of transactions')
plt.title('Business during the past months')

```




    Text(0.5, 1.0, 'Business during the past months')




![png](output_17_1.png)



```python
#看下2016 十月的具体情况
data_drop1.loc[data_drop1['Year_Month']=='2016 October'].nunique()
```




    Time           175
    Transaction    175
    Item            30
    Date_Time      175
    Day              2
    Month            1
    Month_name       1
    Year             1
    Year_Month       1
    dtype: int64




```python
#查看2017 4月 
data_drop1.loc[data_drop1['Year_Month']=='2017 April'].nunique()
```




    Time           502
    Transaction    509
    Item            49
    Date_Time      509
    Day              7
    Month            1
    Month_name       1
    Year             1
    Year_Month       1
    dtype: int64



看下具体每个月的销售量，统计每个月哪些商品畅销，也是运营最关心的。
咖啡在所有月份都是最畅销的。


```python
data2=data_drop1.pivot_table(index='Month_name',columns='Item', aggfunc={'Item':'count'}).fillna(0)
data2['Max']=data2.idxmax(axis=1)
data2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="20" halign="left">Item</th>
      <th>Max</th>
    </tr>
    <tr>
      <th>Item</th>
      <th>Adjustment</th>
      <th>Afternoon with the baker</th>
      <th>Alfajores</th>
      <th>Argentina Night</th>
      <th>Art Tray</th>
      <th>Bacon</th>
      <th>Baguette</th>
      <th>Bakewell</th>
      <th>Bare Popcorn</th>
      <th>Basket</th>
      <th>...</th>
      <th>The Nomad</th>
      <th>Tiffin</th>
      <th>Toast</th>
      <th>Truffles</th>
      <th>Tshirt</th>
      <th>Valentine's card</th>
      <th>Vegan Feast</th>
      <th>Vegan mincepie</th>
      <th>Victorian Sponge</th>
      <th></th>
    </tr>
    <tr>
      <th>Month_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>April</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>24.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>8.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>(Item, Coffee)</td>
    </tr>
    <tr>
      <td>December</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>45.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>65.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>33.0</td>
      <td>0.0</td>
      <td>(Item, Coffee)</td>
    </tr>
    <tr>
      <td>February</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>112.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>60.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>12.0</td>
      <td>26.0</td>
      <td>72.0</td>
      <td>37.0</td>
      <td>21.0</td>
      <td>12.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>(Item, Coffee)</td>
    </tr>
    <tr>
      <td>January</td>
      <td>0.0</td>
      <td>15.0</td>
      <td>39.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>15.0</td>
      <td>36.0</td>
      <td>79.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>(Item, Coffee)</td>
    </tr>
    <tr>
      <td>March</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>61.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>57.0</td>
      <td>80.0</td>
      <td>48.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>(Item, Coffee)</td>
    </tr>
    <tr>
      <td>November</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>141.0</td>
      <td>0.0</td>
      <td>18.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>(Item, Coffee)</td>
    </tr>
    <tr>
      <td>October</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>(Item, Coffee)</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 95 columns</p>
</div>




```python
#将日期时间转为新增索引
data_drop1.index=data_drop1['Date_Time']
data_drop1.index.name='Date'
data_drop2 = data_drop1.drop(['Date_Time'],axis=1)  #去掉原字段
data_drop2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>Transaction</th>
      <th>Item</th>
      <th>Day</th>
      <th>Month</th>
      <th>Month_name</th>
      <th>Year</th>
      <th>Year_Month</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2016-10-30 09:58:11</td>
      <td>09:58:11</td>
      <td>1</td>
      <td>Bread</td>
      <td>Sunday</td>
      <td>10</td>
      <td>October</td>
      <td>2016</td>
      <td>2016 October</td>
    </tr>
    <tr>
      <td>2016-10-30 10:05:34</td>
      <td>10:05:34</td>
      <td>2</td>
      <td>Scandinavian</td>
      <td>Sunday</td>
      <td>10</td>
      <td>October</td>
      <td>2016</td>
      <td>2016 October</td>
    </tr>
    <tr>
      <td>2016-10-30 10:05:34</td>
      <td>10:05:34</td>
      <td>2</td>
      <td>Scandinavian</td>
      <td>Sunday</td>
      <td>10</td>
      <td>October</td>
      <td>2016</td>
      <td>2016 October</td>
    </tr>
    <tr>
      <td>2016-10-30 10:07:57</td>
      <td>10:07:57</td>
      <td>3</td>
      <td>Hot chocolate</td>
      <td>Sunday</td>
      <td>10</td>
      <td>October</td>
      <td>2016</td>
      <td>2016 October</td>
    </tr>
    <tr>
      <td>2016-10-30 10:07:57</td>
      <td>10:07:57</td>
      <td>3</td>
      <td>Jam</td>
      <td>Sunday</td>
      <td>10</td>
      <td>October</td>
      <td>2016</td>
      <td>2016 October</td>
    </tr>
  </tbody>
</table>
</div>



相对于月份日期，我们还关注每天各时段的销售情况，比如早上、中午、下午、傍晚、晚上卖什么比较多。所以要提取时间并计算时间段，转为新特征



```python
data_drop2.loc[(data_drop2['Time']<'12:00:00'),'Daytime']='Morning'
data_drop2.loc[(data_drop2['Time']>='12:00:00')&(data_drop2['Time']<'17:00:00'),'Daytime']='Afternoon'
data_drop2.loc[(data_drop2['Time']>='17:00:00')&(data_drop2['Time']<'21:00:00'),'Daytime']='Evening'
data_drop2.loc[(data_drop2['Time']>='21:00:00')&(data_drop2['Time']<'23:50:00'),'Daytime']='Night'
```


```python
fig, ax=plt.subplots(figsize=(6,4))
sns.set_style('darkgrid')
data_drop2.groupby('Daytime')['Item'].count().sort_values().plot(kind='bar')
plt.ylabel('Number of transactions')
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.title('Business during the day')
```




    Text(0.5, 1.0, 'Business during the day')




![png](output_25_1.png)



```python
data_drop2.groupby('Daytime')['Item'].count().sort_values(ascending=False)

```




    Daytime
    Afternoon    11569
    Morning       8404
    Evening        520
    Night           14
    Name: Item, dtype: int64




```python
data_drop3 = data_drop2.drop(['Time'],axis=1)  #删除time特征
```

接下来找日销售最高的商品。 早上、中午和晚上，咖啡的销量是最高的；理由很显然，晚上咖啡不是太受大家欢迎。Vegan feast在晚上是最畅销的。


```python
data3=data_drop3.pivot_table(index='Daytime',columns='Item', aggfunc={'Item':'count'}).fillna(0)
data3['Max']=data3.idxmax(axis=1)
data3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="20" halign="left">Item</th>
      <th>Max</th>
    </tr>
    <tr>
      <th>Item</th>
      <th>Adjustment</th>
      <th>Afternoon with the baker</th>
      <th>Alfajores</th>
      <th>Argentina Night</th>
      <th>Art Tray</th>
      <th>Bacon</th>
      <th>Baguette</th>
      <th>Bakewell</th>
      <th>Bare Popcorn</th>
      <th>Basket</th>
      <th>...</th>
      <th>The Nomad</th>
      <th>Tiffin</th>
      <th>Toast</th>
      <th>Truffles</th>
      <th>Tshirt</th>
      <th>Valentine's card</th>
      <th>Vegan Feast</th>
      <th>Vegan mincepie</th>
      <th>Victorian Sponge</th>
      <th></th>
    </tr>
    <tr>
      <th>Daytime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Afternoon</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>245.0</td>
      <td>3.0</td>
      <td>31.0</td>
      <td>1.0</td>
      <td>67.0</td>
      <td>30.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>31.0</td>
      <td>93.0</td>
      <td>114.0</td>
      <td>152.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>30.0</td>
      <td>5.0</td>
      <td>(Item, Coffee)</td>
    </tr>
    <tr>
      <td>Evening</td>
      <td>1.0</td>
      <td>15.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>21.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>(Item, Coffee)</td>
    </tr>
    <tr>
      <td>Morning</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>107.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>84.0</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>...</td>
      <td>26.0</td>
      <td>49.0</td>
      <td>204.0</td>
      <td>34.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>22.0</td>
      <td>2.0</td>
      <td>(Item, Coffee)</td>
    </tr>
    <tr>
      <td>Night</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>(Item, Vegan Feast)</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 95 columns</p>
</div>




```python
#咖啡从星期一到星期天都是畅销的。
data4=data_drop3.pivot_table(index='Day',columns='Item', aggfunc={'Item':'count'}).fillna(0)
data4['Max']=data4.idxmax(axis=1)
data4
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="20" halign="left">Item</th>
      <th>Max</th>
    </tr>
    <tr>
      <th>Item</th>
      <th>Adjustment</th>
      <th>Afternoon with the baker</th>
      <th>Alfajores</th>
      <th>Argentina Night</th>
      <th>Art Tray</th>
      <th>Bacon</th>
      <th>Baguette</th>
      <th>Bakewell</th>
      <th>Bare Popcorn</th>
      <th>Basket</th>
      <th>...</th>
      <th>The Nomad</th>
      <th>Tiffin</th>
      <th>Toast</th>
      <th>Truffles</th>
      <th>Tshirt</th>
      <th>Valentine's card</th>
      <th>Vegan Feast</th>
      <th>Vegan mincepie</th>
      <th>Victorian Sponge</th>
      <th></th>
    </tr>
    <tr>
      <th>Day</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Friday</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>59.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>21.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>6.0</td>
      <td>37.0</td>
      <td>63.0</td>
      <td>33.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>(Item, Coffee)</td>
    </tr>
    <tr>
      <td>Monday</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>32.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>38.0</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>(Item, Coffee)</td>
    </tr>
    <tr>
      <td>Saturday</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>67.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>33.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>24.0</td>
      <td>35.0</td>
      <td>53.0</td>
      <td>46.0</td>
      <td>21.0</td>
      <td>7.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>(Item, Coffee)</td>
    </tr>
    <tr>
      <td>Sunday</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>15.0</td>
      <td>28.0</td>
      <td>36.0</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>(Item, Coffee)</td>
    </tr>
    <tr>
      <td>Thursday</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>57.0</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>21.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>25.0</td>
      <td>53.0</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>(Item, Coffee)</td>
    </tr>
    <tr>
      <td>Tuesday</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>43.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>23.0</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>40.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>(Item, Coffee)</td>
    </tr>
    <tr>
      <td>Wednesday</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>41.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>23.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>13.0</td>
      <td>35.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>(Item, Coffee)</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 95 columns</p>
</div>



通过曲线进一步探索。通过观察上面的柱形图，11月的销售最大，其次是2月、3月，而12月、1月有所下降。

pandas的resample重采样方法，是一个对常规时间序列数据重新采样和频率转换的便捷的方法  
降采样：高频数据到低频数据
升采样：低频数据到高频数据


| 参数 | 使用 |
| --- | --- |
| freq | 表示重采样频率，例如‘M’、‘5min’，Second(15) |
| how=’mean’ | 用于产生聚合值的函数名或数组函数，例如‘mean’、‘ohlc’、np.max等，默认是‘mean’，其他常用的值由：‘first’、‘last’、‘median’、‘max’、‘min’ |
| axis=0 | 默认是纵轴，横轴设置axis=1 |
| fill_method = None | 升采样时如何插值，比如‘ffill’、‘bfill’等， 即forward和back |
| closed = ‘right’ |在降采样时，各时间段的哪一段是闭合的，‘right’或‘left’，默认‘right’ |
| label= ‘right’ | 在降采样时，如何设置聚合值的标签，例如，9：30-9：35会被标记成9：30还是9：35,默认9：35 |
| loffset = None	 | 面元标签的时间校正值，比如‘-1s’或Second(-1)用于将聚合标签调早1秒|
| limit=None	 | 在向前或向后填充时，允许填充的最大时期数 |
| kind = None	 | 聚合到时期（‘period’）或时间戳（‘timestamp’），默认聚合到时间序列的索引类型 |
| convention = None | 当重采样时期时，将低频率转换到高频率所采用的约定（start或end）。默认‘end’ |



```python
data_drop3['Item'].resample('M').count().plot()
plt.ylabel('Number of transactions')
plt.title('Business during the past months')
```




    Text(0.5, 1.0, 'Business during the past months')




![png](output_33_1.png)


下钻到周为周期，看下具体销售情况


```python
data_drop3['Item'].resample('W').count().plot()
plt.ylabel('Number of transactions')
plt.title('Weekly business during the past months')
```




    Text(0.5, 1.0, 'Weekly business during the past months')




![png](output_35_1.png)


 查看高频的日期数据，继续升采样


```python
data_drop3['Item'].resample('D').count().plot()
plt.ylabel('Number of transactions')
plt.title('Daily business during the past months')
```




    Text(0.5, 1.0, 'Daily business during the past months')




![png](output_37_1.png)



```python
data_drop3['Item'].resample('D').count().max()  #最多一天的销售量
```




    292



## 关联分析

首先原始数据都是单个商品的条目，也就是每个商品的购买时间，我们要将同时购买的交易根据交易单号（transform）聚合到一条记录，也就是构建我们的项集。以便分析关联的商品


```python
data_drop3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Transaction</th>
      <th>Item</th>
      <th>Day</th>
      <th>Month</th>
      <th>Month_name</th>
      <th>Year</th>
      <th>Year_Month</th>
      <th>Daytime</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2016-10-30 09:58:11</td>
      <td>1</td>
      <td>Bread</td>
      <td>Sunday</td>
      <td>10</td>
      <td>October</td>
      <td>2016</td>
      <td>2016 October</td>
      <td>Morning</td>
    </tr>
    <tr>
      <td>2016-10-30 10:05:34</td>
      <td>2</td>
      <td>Scandinavian</td>
      <td>Sunday</td>
      <td>10</td>
      <td>October</td>
      <td>2016</td>
      <td>2016 October</td>
      <td>Morning</td>
    </tr>
    <tr>
      <td>2016-10-30 10:05:34</td>
      <td>2</td>
      <td>Scandinavian</td>
      <td>Sunday</td>
      <td>10</td>
      <td>October</td>
      <td>2016</td>
      <td>2016 October</td>
      <td>Morning</td>
    </tr>
    <tr>
      <td>2016-10-30 10:07:57</td>
      <td>3</td>
      <td>Hot chocolate</td>
      <td>Sunday</td>
      <td>10</td>
      <td>October</td>
      <td>2016</td>
      <td>2016 October</td>
      <td>Morning</td>
    </tr>
    <tr>
      <td>2016-10-30 10:07:57</td>
      <td>3</td>
      <td>Jam</td>
      <td>Sunday</td>
      <td>10</td>
      <td>October</td>
      <td>2016</td>
      <td>2016 October</td>
      <td>Morning</td>
    </tr>
  </tbody>
</table>
</div>




```python

shopping_list=[]
for item in data_drop3['Transaction'].unique():
    lst2=list(set(data_drop3[data_drop3['Transaction']==item]['Item']))
    if len(lst2)>0:
        shopping_list.append(lst2)
print(shopping_list[0:3])
print(len(shopping_list))
```

    [['Bread'], ['Scandinavian'], ['Hot chocolate', 'Jam', 'Cookies']]
    9465



```python
shopping_list
```




    [['Bread'],
     ['Scandinavian'],
     ['Cookies', 'Hot chocolate', 'Jam'],
     ['Muffin'],
     ['Pastry', 'Coffee', 'Bread'],
     ['Pastry', 'Medialuna', 'Muffin'],
     ['Pastry', 'Medialuna', 'Coffee', 'Tea'],
     ['Pastry', 'Bread'],
     ['Bread', 'Muffin'],
     ['Medialuna', 'Scandinavian'],
     ['Bread', 'NONE', 'Medialuna'],
     ['Pastry', 'Coffee', 'Tartine', 'Jam', 'Tea'],
     ['Bread', 'Coffee', 'Basket'],
     ['Pastry', 'Bread', 'Medialuna'],
     ['NONE', 'Mineral water', 'Scandinavian'],
     ['Bread', 'Coffee', 'Medialuna'],
     ['Hot chocolate'],
     ['Farm House'],
     ['Bread', 'Farm House'],
     ['Bread', 'Medialuna'],
     ['Bread', 'Coffee', 'Medialuna'],
     ['Jam'],
     ['Scandinavian', 'Muffin'],
     ['Bread'],
     ['Scandinavian'],
     ['Fudge'],
     ['Scandinavian'],
     ['Coffee', 'Bread'],
     ['Bread', 'NONE', 'Jam'],
     ['Bread'],
     ['Basket'],
     ['Scandinavian', 'Muffin'],
     ['Coffee'],
     ['Coffee', 'Muffin'],
     ['Scandinavian', 'Muffin'],
     ['Tea', 'Bread'],
     ['Coffee', 'NONE', 'Bread'],
     ['Tea', 'Bread'],
     ['Scandinavian'],
     ['Juice', 'Muffin', 'Coffee', 'Tartine', 'NONE'],
     ['Scandinavian'],
     ['Tea', 'Bread'],
     ['Fudge', 'Scandinavian'],
     ['Coffee', 'Medialuna'],
     ['Medialuna', 'Coffee', 'Hot chocolate'],
     ['Coffee'],
     ['Juice', 'Muffin', 'Jam', "Ella's Kitchen Pouches", 'Bread'],
     ['Coffee'],
     ['Coffee', 'Medialuna'],
     ['Bread', 'Victorian Sponge'],
     ['Bread'],
     ['Scandinavian'],
     ['Bread'],
     ['Tea', 'Frittata', 'Coffee', 'Hearty & Seasonal'],
     ['Coffee', 'Frittata'],
     ['Scandinavian'],
     ['Tea', 'Hot chocolate', 'Victorian Sponge', 'Soup'],
     ['Tea', 'NONE'],
     ['Cookies', 'Coffee', 'Juice'],
     ['Coffee'],
     ['Coffee', 'Smoothies', 'Pick and Mix Bowls', 'Hearty & Seasonal'],
     ['Coffee'],
     ['Cake'],
     ['Mighty Protein', 'Coffee', 'Tartine', 'Tea', 'NONE'],
     ['Frittata', 'Mineral water', 'Hearty & Seasonal'],
     ['Muffin', 'NONE', 'Mineral water', 'Hearty & Seasonal'],
     ['Chicken sand', 'Coffee', 'Scandinavian', 'Tea', 'Frittata'],
     ['Tea', 'Bread', 'Victorian Sponge'],
     ['Fudge'],
     ['Muffin'],
     ['Coffee', 'Bread'],
     ['Bread'],
     ['Coffee', 'Bread'],
     ['Frittata', 'NONE', 'Jam'],
     ['Coffee'],
     ['Scandinavian'],
     ['Fudge'],
     ['Tea', 'Coffee', 'Fudge', 'Muffin'],
     ['Frittata', 'Coffee', 'Bread'],
     ['Coffee', 'Cake'],
     ['Bread', 'NONE', 'Tartine'],
     ['Coffee', 'Bread'],
     ['Bread'],
     ['Pastry', 'Coffee', 'Medialuna'],
     ['Juice'],
     ['Bread'],
     ['Coffee', 'Jam'],
     ['Bread'],
     ['Bread', 'Coffee'],
     ['Tea', 'NONE'],
     ['Coffee'],
     ['Coke', 'Bread'],
     ['Coffee'],
     ['Tea', 'Farm House', 'Pastry'],
     ['Pastry', 'Coffee', 'Juice'],
     ['Pastry', 'Coffee', 'Juice'],
     ['Farm House'],
     ['Pastry', 'Coffee', 'Bread'],
     ['Bread'],
     ['Pastry', 'Coffee'],
     ['Farm House'],
     ['Bread', 'NONE'],
     ['Coffee', 'Medialuna', 'Muffin'],
     ['Tea', 'Coffee', 'NONE', 'Cake'],
     ['Pastry', 'Coffee'],
     ['Tea', 'Coffee'],
     ['Farm House', 'Muffin'],
     ['Pastry', 'Bread', 'Muffin'],
     ['Pastry'],
     ['Coffee'],
     ['Coffee'],
     ['Coffee', 'Cake'],
     ['Farm House'],
     ['Bread'],
     ['Coffee'],
     ['Cookies', 'Coffee', 'Cake', 'My-5 Fruit Shoot'],
     ['Cookies', 'Tea', 'Pastry'],
     ['Coffee', 'Muffin'],
     ['Farm House'],
     ['Coffee', 'Bread'],
     ['Cookies'],
     ['Juice', 'Coffee', 'NONE', 'Jam'],
     ['Coffee', 'Soup'],
     ['Tea'],
     ['Coffee', 'Jam'],
     ['Farm House'],
     ['Tea', 'Coffee', 'NONE', 'Tartine'],
     ['Coffee', 'Cake'],
     ['Coffee', 'Muffin'],
     ['Bread', 'Coffee'],
     ['Tea', 'Juice', 'Cookies', 'Pick and Mix Bowls'],
     ['Hearty & Seasonal', 'Soup'],
     ['Tea', 'Farm House', 'Hearty & Seasonal'],
     ['Bread'],
     ['Tea', 'Soup'],
     ['Farm House'],
     ['Coffee', 'Hearty & Seasonal', 'Soup'],
     ['Bread', 'Hearty & Seasonal', 'Muffin'],
     ['Cookies', 'Coffee'],
     ['Bread'],
     ['Tea', 'Muffin'],
     ['Tea'],
     ['Hearty & Seasonal', 'Soup'],
     ['Tea', 'Coffee', 'Focaccia'],
     ['Sandwich'],
     ['Coffee', 'Tartine'],
     ['Coffee', 'Fudge'],
     ['Coffee'],
     ['Bread'],
     ['Coke', 'Soup'],
     ['Muffin'],
     ['Tea', 'Coffee', 'Cookies', 'Muffin'],
     ['Coke', 'Coffee', 'Mighty Protein', 'Hearty & Seasonal'],
     ['Tea', 'Sandwich', 'Muffin'],
     ['Coffee', 'Muffin'],
     ['Coffee', 'Bread'],
     ['Coffee', 'Muffin'],
     ['Cookies', 'Bread', 'Hot chocolate'],
     ['Mineral water', 'Muffin'],
     ['Tea'],
     ['Muffin'],
     ['Juice', 'Coffee'],
     ['Coffee', 'Muffin'],
     ['Tea'],
     ['Coffee', 'Muffin'],
     ['Bread', 'Sandwich', 'Mineral water', 'Hearty & Seasonal'],
     ['Tea', 'Coffee'],
     ['Mineral water'],
     ['Coffee'],
     ['Tea', 'Soup'],
     ['Bread', 'Coffee'],
     ['Bread', 'Muffin'],
     ['Coffee'],
     ['Bread'],
     ['Cookies', 'Coffee'],
     ['Pastry', 'Coffee'],
     ['Pastry', 'Coffee'],
     ['Tea', 'Jam'],
     ['Tea'],
     ['Pastry', "Ella's Kitchen Pouches"],
     ['Pastry'],
     ['Tea', 'NONE', 'Farm House'],
     ['Tea', 'Victorian Sponge'],
     ['Tea', 'Pastry'],
     ['Bread'],
     ['Coffee'],
     ['Bread', 'Coffee'],
     ['Pastry', 'Coffee'],
     ['Tea', 'Soup'],
     ['Bread', 'NONE'],
     ['Coffee'],
     ['Bread'],
     ['Bread', 'NONE', 'Scandinavian'],
     ['Cookies', 'Bread', 'Cake'],
     ['Tea', 'Coffee'],
     ['Cookies', 'Coffee'],
     ['Coffee'],
     ['Bread'],
     ['Pastry', 'Coffee', 'NONE'],
     ['Coffee'],
     ['Coffee'],
     ['Bread', 'Scandinavian'],
     ['Pastry', 'Coffee'],
     ['Pastry', 'Coffee', 'Scandinavian'],
     ['Muffin', 'Coffee', 'Hearty & Seasonal'],
     ['Bread'],
     ['Coffee'],
     ['Scandinavian'],
     ['Coffee', 'Soup'],
     ['Cookies', 'Coffee', 'Tartine'],
     ['Bread'],
     ['Tea', 'Bread'],
     ['Pastry', 'Coffee'],
     ['Coffee', 'Tartine'],
     ['Cookies', 'Bread'],
     ['Sandwich'],
     ['Tartine'],
     ['Pastry', 'Bread'],
     ['Bread', 'Scandinavian'],
     ['Scandinavian'],
     ['Coffee', 'Alfajores'],
     ['Coffee'],
     ['Coke', 'Coffee', 'Soup'],
     ['Cake'],
     ['Tea', 'Sandwich', 'Soup'],
     ['Coffee'],
     ['Bread', 'Hearty & Seasonal'],
     ['Alfajores'],
     ['Cake'],
     ['Tea', 'Bread'],
     ['Tea', 'Coffee', 'Bread', 'Soup'],
     ['Coffee'],
     ['Scandinavian'],
     ['Fudge'],
     ['Coffee', 'Sandwich'],
     ['Tea', 'Bread', 'Cookies', 'Pastry'],
     ['Muffin'],
     ['Scandinavian'],
     ['Tea', 'Jam'],
     ['Juice', 'Tea', 'Alfajores'],
     ['Coffee', 'Cake', 'Muffin', 'Soup'],
     ['Hot chocolate', 'Cake', 'Coffee'],
     ['Cookies', 'Cake'],
     ['Tea', 'Bread'],
     ['Juice', 'Coffee', 'Alfajores'],
     ['Tea', 'Coffee', 'Soup'],
     ['Hot chocolate'],
     ['Cookies'],
     ['Coffee', 'Medialuna'],
     ['Bread', 'Eggs'],
     ['Scandinavian'],
     ['Medialuna'],
     ['Pastry', 'Coffee'],
     ['Coffee', 'Bread'],
     ['Pastry', 'Bread', 'Coffee', 'NONE'],
     ['Pastry', 'Coffee'],
     ['Pastry'],
     ['Coffee'],
     ['Coffee'],
     ['Pastry', 'Coffee', 'Hot chocolate'],
     ['Coke'],
     ['Coffee', 'Jam'],
     ['Tea'],
     ['Pastry', 'Bread', 'NONE'],
     ['Coffee', 'Farm House', 'Alfajores'],
     ['Tea', 'Pastry'],
     ['Bread'],
     ['Coffee', 'Jam'],
     ['Tea', 'Coffee', 'Pastry'],
     ['Farm House'],
     ['Pastry', 'Coffee', 'NONE', 'Bread'],
     ['Farm House'],
     ['Pastry', 'Coffee'],
     ['Farm House'],
     ['Pastry', 'Coffee'],
     ['Coffee', 'NONE'],
     ['Pastry', 'Coffee'],
     ['Bread'],
     ['Pastry', 'Bread', 'Coffee'],
     ['Pastry', 'Coffee'],
     ['Coffee', 'Farm House'],
     ['Pastry', 'Farm House'],
     ['Juice', 'Bread', 'Alfajores'],
     ['Pastry', 'Coffee'],
     ['Bread', 'Alfajores'],
     ['Bread'],
     ['Cookies', 'Coffee'],
     ['Coffee'],
     ['Cookies'],
     ['Hearty & Seasonal'],
     ['Coffee'],
     ['Bread', 'Tartine', 'Jam'],
     ['Farm House', 'Scandinavian'],
     ['Muffin', 'Coffee', 'Scandinavian', 'Hearty & Seasonal'],
     ['Coke', 'Pick and Mix Bowls', 'Coffee', 'Tartine', 'NONE'],
     ['Bread'],
     ['Coffee', 'Farm House', 'Scandinavian', 'Alfajores'],
     ['Juice', 'Mineral water', 'Soup'],
     ['Tea', 'Soup'],
     ['Juice', 'Cookies', 'Tea', 'Soup'],
     ['Tea', 'Hearty & Seasonal'],
     ['Fudge'],
     ['Juice', 'Coffee', 'Mineral water', 'Soup'],
     ['Hearty & Seasonal'],
     ['Smoothies', 'Hearty & Seasonal'],
     ['Tea', 'Hearty & Seasonal'],
     ['Coffee', 'Sandwich'],
     ['Cookies', 'Farm House'],
     ['Tea', 'Juice', 'Soup'],
     ['Soup'],
     ['Coffee'],
     ['Bread'],
     ['Cookies', 'Tea'],
     ['Bread', 'Hearty & Seasonal', 'Soup'],
     ['Coffee'],
     ['Coffee'],
     ['Tea', 'Coffee'],
     ['Bread'],
     ['Tea'],
     ['Tea', 'Coffee'],
     ['Coffee'],
     ['Fudge'],
     ['Tea'],
     ['Tea', 'Soup'],
     ['Juice'],
     ['Alfajores'],
     ['Bread'],
     ['Coffee', 'Alfajores'],
     ['Juice', 'Cookies', 'Pick and Mix Bowls', 'Mineral water', 'Alfajores'],
     ['Coffee', 'Alfajores'],
     ['Fudge'],
     ['Juice', 'Mighty Protein'],
     ['Coffee', 'Mineral water', 'Alfajores'],
     ['Coffee', 'Mighty Protein'],
     ['Alfajores'],
     ['Coffee', 'Bread'],
     ['Medialuna'],
     ['Medialuna', 'Bread'],
     ['Farm House'],
     ['Pastry', 'Coffee'],
     ['Farm House'],
     ['Pastry', 'Coffee', 'NONE'],
     ['Medialuna'],
     ['Tea', 'Bread', 'NONE', 'Sandwich'],
     ['Bread'],
     ['Coffee', 'Farm House', 'Jam'],
     ['Focaccia'],
     ['Bread'],
     ['Pastry', 'Bread', 'Tartine', 'My-5 Fruit Shoot'],
     ['Pastry', 'Farm House'],
     ['Pastry', 'Bread'],
     ['Pastry', 'Bread'],
     ['Coffee', 'Jam'],
     ['Farm House'],
     ['Pastry', 'Bread'],
     ['Tea', 'Coffee', 'Bread'],
     ['Jam'],
     ['Pastry', 'Medialuna'],
     ['Coffee'],
     ['Tea'],
     ['Bread'],
     ['Pastry', 'Bread'],
     ['Pastry'],
     ['Farm House'],
     ['Tea'],
     ['Coffee', 'Brownie', 'Fudge', 'Alfajores'],
     ['Cookies', 'Bread', 'Fudge'],
     ['Coffee'],
     ['Bread'],
     ['Pastry', 'Coffee', 'Jam'],
     ['Coffee', 'Soup'],
     ['Pastry', 'Smoothies', 'Hearty & Seasonal', 'Soup'],
     ['Pastry', 'Bread'],
     ['Pastry', 'Medialuna'],
     ['Tea', 'Coffee', 'NONE'],
     ['Cookies', 'Bread'],
     ['Pick and Mix Bowls',
      'Hearty & Seasonal',
      'Coffee',
      'Focaccia',
      'Bread',
      'Sandwich'],
     ['Coffee'],
     ['Coffee', 'Sandwich', 'Brownie'],
     ['Cookies', 'Brownie'],
     ['Coffee'],
     ['Coffee', 'NONE'],
     ['Bread', 'NONE', 'Sandwich', 'Coffee'],
     ['Coffee', 'Hot chocolate'],
     ['Tea', 'NONE'],
     ['Coffee', 'Sandwich', 'Brownie'],
     ['Pastry', 'Brownie', 'Farm House', 'Focaccia'],
     ['Cake', 'Focaccia'],
     ['Mineral water', 'Hearty & Seasonal', 'Soup'],
     ['Hearty & Seasonal'],
     ['Hearty & Seasonal'],
     ['Pastry', 'Coffee'],
     ['Coffee'],
     ['Tea', 'Cake'],
     ['Coffee', 'Soup'],
     ['Bread', 'Focaccia', 'Fudge'],
     ['Hot chocolate', 'Coffee', 'Alfajores'],
     ['Pastry', 'Cake', 'Brownie'],
     ['Medialuna', 'Dulce de Leche', 'Alfajores'],
     ['Dulce de Leche', 'Alfajores'],
     ['Bread', 'Cake'],
     ['Coffee', 'Cake'],
     ['Bread'],
     ['Sandwich'],
     ['Bread', 'Coffee', 'Medialuna', 'Alfajores'],
     ['Coffee'],
     ['Bread', 'Alfajores'],
     ['Tea', 'Coffee'],
     ['Bread'],
     ['Dulce de Leche'],
     ['Farm House'],
     ['Fudge'],
     ['Coffee'],
     ['Juice', 'Coffee', 'Brownie', "Ella's Kitchen Pouches", 'Alfajores'],
     ['Alfajores'],
     ['Cookies', 'Juice', 'Brownie', 'Alfajores', 'Bread'],
     ['Cookies', 'Cake'],
     ['Pick and Mix Bowls'],
     ['Coffee'],
     ['Farm House'],
     ['Tea'],
     ['Coffee'],
     ['Focaccia', 'Farm House'],
     ['Dulce de Leche'],
     ['Juice', 'My-5 Fruit Shoot', 'Cookies', "Ella's Kitchen Pouches"],
     ['Bread'],
     ['Hot chocolate'],
     ['Coffee'],
     ['Coffee'],
     ['Coffee'],
     ['Medialuna'],
     ['Tea', 'Bread', 'Jam'],
     ['Farm House'],
     ['Bread'],
     ['Coffee', 'NONE'],
     ['Medialuna', 'Coffee'],
     ['Bread'],
     ['Pastry', 'Medialuna'],
     ['Coffee', 'Jam'],
     ['Bread', 'Coffee'],
     ['Coffee'],
     ['Bread'],
     ['Hot chocolate'],
     ['Bread', 'Medialuna'],
     ['Bread', 'Jam'],
     ['Bread', 'Jam'],
     ['Coffee', 'Bread'],
     ['Bread', 'Farm House'],
     ['Bread', 'Coffee'],
     ['Tea', 'Pastry'],
     ['Coffee'],
     ['Bread'],
     ['Pastry', 'Brownie'],
     ['Farm House'],
     ['Coffee'],
     ['Tea', 'Coffee', 'Cookies', 'Honey'],
     ['Tea'],
     ['Bread'],
     ['Brownie'],
     ['Bread', 'Coffee'],
     ['Bread'],
     ['Coffee'],
     ['Tea'],
     ['Tea'],
     ['Coffee'],
     ['Bread', 'Coffee'],
     ['Bread'],
     ['Bread', 'The BART'],
     ['Coffee', 'NONE', 'Brownie'],
     ['Coffee'],
     ['Coffee', 'Brownie'],
     ['Coffee', 'NONE', 'Tartine', 'Mineral water'],
     ['Coffee', 'Brownie'],
     ['Juice', 'Brownie'],
     ['Pastry', 'Medialuna', 'Coffee', 'Jam', 'Alfajores', 'Tea', 'Bread'],
     ['Bread'],
     ['Fudge'],
     ['Coffee', 'Pick and Mix Bowls'],
     ['Pastry', 'Bread'],
     ['Coffee', 'Mineral water', 'Scandinavian', 'Soup'],
     ['Juice', 'Sandwich', 'Mineral water', 'Pick and Mix Bowls'],
     ['Pastry', 'Bread', 'NONE'],
     ['Coffee', 'Hearty & Seasonal'],
     ['Bread', 'Coffee', 'Mineral water', 'Hearty & Seasonal'],
     ['Coffee', 'Tartine', 'Sandwich', 'Brownie'],
     ['Coffee', 'Brownie'],
     ['Coffee', 'Tartine', 'Alfajores'],
     ['Coffee', 'Mighty Protein'],
     ['Coffee', 'Hot chocolate'],
     ['Coffee', 'Jam', 'Soup'],
     ['Brownie', 'Mineral water'],
     ['Coffee', 'Scandinavian'],
     ['Coffee', 'Scandinavian'],
     ['Coffee'],
     ['Coffee', 'Brownie', 'Hot chocolate', 'Hearty & Seasonal'],
     ['Tea', 'Cake'],
     ['Medialuna'],
     ['Tea', 'Coffee', 'Alfajores'],
     ['Juice', 'Coffee', 'Brownie', 'Alfajores'],
     ['Coffee', 'Sandwich'],
     ['Bread', 'Soup'],
     ['Bread', 'Soup'],
     ['Bread'],
     ['Tea', 'Brownie', 'Cookies'],
     ['Coffee', 'Brownie', 'Bread'],
     ['Brownie'],
     ['Cookies', 'Bread', 'Jam'],
     ['Coffee'],
     ['Tea', 'Brownie'],
     ['Medialuna', 'Scandinavian', 'Alfajores'],
     ['Cookies', 'Hot chocolate', 'Coffee', 'Alfajores'],
     ['Brownie'],
     ['Fudge'],
     ['Tea', 'Brownie'],
     ['Dulce de Leche', 'Fudge'],
     ['Pastry', 'Fudge', 'Coffee', 'Scandinavian', 'Bread', 'Granola'],
     ['Bread', 'Fudge'],
     ['Hot chocolate', 'Coffee', 'Medialuna', 'NONE'],
     ['Bread'],
     ['Bread', 'Medialuna'],
     ['Bread', 'Coffee', 'Jam', 'Medialuna'],
     ['Coffee', 'Brownie', 'Jam'],
     ['Bread', 'Coffee', 'Jam'],
     ['Scandinavian'],
     ['Cookies', 'Coffee'],
     ['Coffee', 'Scandinavian'],
     ['Bread'],
     ['Bread'],
     ['Coffee', 'NONE', 'Brownie', 'Jam'],
     ['Coffee'],
     ['Hot chocolate', 'Brownie'],
     ['Bread'],
     ['Bread', 'Medialuna'],
     ['Coffee', 'Brownie', 'Hot chocolate'],
     ['Scandinavian'],
     ['Bread', 'NONE', 'Basket'],
     ['Coffee', 'Brownie', 'Bread'],
     ['Cake', 'Fudge'],
     ['Coffee', 'Basket', 'Hot chocolate'],
     ['Scandinavian'],
     ['Scandinavian'],
     ['Medialuna', 'Coffee', 'Cake', 'Bread'],
     ['Coffee', 'Medialuna'],
     ['Pastry', 'Bread'],
     ['Bread', 'Brownie'],
     ['Medialuna', 'Farm House'],
     ['Bread', 'Medialuna'],
     ['Smoothies', 'Medialuna', 'Coffee', 'Brownie', 'Hot chocolate'],
     ['Medialuna', 'Coffee', 'Brownie', 'Tea', 'Bread'],
     ['Cookies', 'Medialuna'],
     ['Coffee', 'Medialuna'],
     ['Bread', 'Fudge'],
     ['Tea', 'Coffee'],
     ['Cookies', 'Coffee'],
     ['Pastry', 'Bread', 'Brownie'],
     ['Bread', 'Coffee'],
     ['Tea', 'Coffee', 'NONE'],
     ['Bread'],
     ['Medialuna', 'Coffee'],
     ['My-5 Fruit Shoot', 'Coffee', 'Brownie'],
     ['Coffee', 'Fudge'],
     ['Coffee'],
     ['Coffee', 'Basket'],
     ['Farm House'],
     ['Farm House'],
     ['Coffee', 'Brownie'],
     ['My-5 Fruit Shoot', 'Brownie', 'Scandinavian'],
     ['Coffee', 'Tartine', 'Brownie', 'Hot chocolate', 'NONE'],
     ['Bread', 'Brownie', 'Hot chocolate'],
     ['Fudge', 'Jam'],
     ['Coffee'],
     ['Fudge'],
     ['Mighty Protein', 'Hearty & Seasonal', 'Coffee', 'Cake', 'Tea'],
     ['Coffee'],
     ['Scandinavian'],
     ['Cake', 'Victorian Sponge', 'Scandinavian'],
     ['Tea', 'Cake'],
     ['Coffee'],
     ['Coffee'],
     ['Bread'],
     ['Frittata'],
     ['My-5 Fruit Shoot', 'Coffee', 'Brownie'],
     ['Fairy Doors', 'Focaccia'],
     ['Tea', 'Coffee', 'NONE'],
     ['Hot chocolate', 'Brownie', 'Pick and Mix Bowls', 'Hearty & Seasonal'],
     ['Frittata', 'Coffee'],
     ['Bread', 'Brownie'],
     ['Frittata'],
     ['Scandinavian'],
     ['Bread'],
     ['Farm House'],
     ['Brownie', 'Jam', 'Pick and Mix Bowls', 'Soup'],
     ['Tea', 'Coffee', 'Hearty & Seasonal'],
     ['Bread'],
     ['Bread', 'Frittata'],
     ['Scandinavian'],
     ['Hot chocolate', 'NONE', 'Tartine', 'Coffee'],
     ['Coffee', 'Mighty Protein'],
     ['Frittata', 'Coffee'],
     ['Scandinavian'],
     ['Coffee', 'Mineral water', 'Hearty & Seasonal'],
     ['Scandinavian'],
     ['Bread', 'Coffee'],
     ['Bread', 'Coffee', 'Hearty & Seasonal', 'Soup'],
     ['Tea', 'Cake', 'Brownie'],
     ['Coffee', 'Mighty Protein'],
     ['Tea', 'Coffee', 'Brownie', 'Soup'],
     ['Cake'],
     ['Hot chocolate', 'Coffee', 'Cake'],
     ['Coffee', 'Hearty & Seasonal'],
     ['Coffee'],
     ['Frittata', 'Coffee', 'Brownie'],
     ['Coke', 'Mineral water', 'Brownie', 'Frittata', 'Sandwich'],
     ['Granola', 'Scandinavian'],
     ['Coke', 'Coffee'],
     ['Tea', 'Cake', 'Brownie'],
     ['Fudge'],
     ['Coffee'],
     ['Brownie'],
     ['Cake'],
     ['Tea'],
     ['Coffee', 'Cake', 'Sandwich', 'Soup'],
     ['Coffee', 'Brownie', 'Bread'],
     ['Bread', 'Coffee', 'Brownie'],
     ['Coffee', 'Brownie', 'Bread'],
     ['Coffee', 'Brownie'],
     ['Coffee', 'Brownie', 'Soup'],
     ['Fudge'],
     ['Bread', 'Focaccia'],
     ['Bread', 'Farm House', 'Jam'],
     ['Hot chocolate', 'Brownie'],
     ['Hot chocolate'],
     ['Coffee', 'Brownie'],
     ['Pastry', 'Medialuna'],
     ['Bread'],
     ['Pastry', 'Coffee', 'Bread'],
     ['Coffee', 'NONE', 'Mineral water'],
     ['Pastry', 'Bread', 'Hot chocolate'],
     ['Bread'],
     ['Bread'],
     ['Coffee', 'Jam', 'Bread'],
     ['Pastry', 'Coffee', 'NONE'],
     ['Medialuna', 'Hearty & Seasonal', 'Coffee', 'NONE', 'Focaccia'],
     ['Bread', 'Coffee', 'Farm House', 'Jam'],
     ['Farm House'],
     ['Tea', 'Coffee', 'Medialuna', 'Pastry'],
     ['Pastry', 'Medialuna', 'Coffee'],
     ['Coffee', 'Farm House', 'Basket', 'Hot chocolate'],
     ['Pastry', 'Farm House'],
     ['My-5 Fruit Shoot', 'Hot chocolate', 'Coffee'],
     ['Pastry', 'Medialuna', 'Muffin'],
     ['Farm House'],
     ['Pastry', 'Focaccia', 'Muffin'],
     ['Tea', 'Coffee'],
     ['Bread'],
     ['Focaccia'],
     ['Pastry', 'Coffee', 'Jam', 'Hot chocolate'],
     ['Bread'],
     ['Pastry', 'Coffee', 'Bread'],
     ['Tea', 'NONE'],
     ['Hot chocolate', 'Coffee', 'Mineral water'],
     ['Coffee'],
     ['Coffee'],
     ['Coffee', 'NONE', 'Jam'],
     ['Muffin'],
     ['Bread'],
     ['Bread'],
     ['Bread'],
     ['Hot chocolate', 'NONE', 'Coffee', 'Muffin'],
     ['Fudge', 'Muffin'],
     ['Bread'],
     ['Bread'],
     ['Farm House'],
     ['Coffee', 'Muffin'],
     ['Focaccia'],
     ['Tea', 'Bread'],
     ['Coffee', 'Hot chocolate'],
     ['Coffee', 'Tartine', 'Jam', 'Pick and Mix Bowls'],
     ['Jam'],
     ['Tartine'],
     ['Bread', 'Coffee'],
     ['Juice', 'Pick and Mix Bowls', 'Mineral water', 'Coffee', 'Tartine'],
     ['Tea', 'Bread', 'Juice'],
     ['Coffee', 'Cake'],
     ['Farm House'],
     ['Farm House'],
     ['Coffee', 'Cake', 'Brownie'],
     ['Coffee', 'Focaccia'],
     ['Tea', 'Coffee', 'Muffin'],
     ['Coffee'],
     ['Tea', 'Coffee'],
     ['Coffee', 'Cake', 'Hearty & Seasonal'],
     ['Tea', 'Bread', 'Cake'],
     ['Coffee', 'Mighty Protein'],
     ['Hot chocolate', 'Muffin'],
     ['Coffee', 'Mighty Protein', 'Soup'],
     ['Jam'],
     ['Coffee', 'Alfajores'],
     ['Hot chocolate', 'Cake', 'Alfajores'],
     ['Tea', 'Hot chocolate', 'Muffin'],
     ['Brownie'],
     ['Coffee', 'Cake'],
     ['Tea', 'Victorian Sponge', 'Alfajores'],
     ['Bread'],
     ['Tea', 'Tartine'],
     ['Tea', 'Mighty Protein'],
     ['Bread', 'Fudge'],
     ['Medialuna', 'Coffee', 'Tartine'],
     ['Pastry', 'Bread'],
     ['Pastry', 'Bread'],
     ['Tea', 'Coffee', 'Alfajores'],
     ['Coffee', 'Alfajores'],
     ['Coffee', 'Alfajores'],
     ['Pastry', 'Alfajores'],
     ['Alfajores'],
     ['Pastry', 'Hearty & Seasonal', 'Tea', 'Alfajores'],
     ['Tea', 'Sandwich'],
     ['Tea', 'Empanadas'],
     ['Tea', 'Alfajores'],
     ['Alfajores'],
     ['Pastry', 'Dulce de Leche'],
     ['Pastry', 'Coffee'],
     ['Coffee', 'Farm House'],
     ['Medialuna', 'Bread', 'Scandinavian'],
     ['Bread'],
     ['Pastry', 'Tea', 'Alfajores'],
     ['Pastry', 'Coffee'],
     ['Pastry', 'Bread', 'Farm House', 'Muffin'],
     ['Tea', 'Medialuna'],
     ['Alfajores'],
     ['Tea', 'Sandwich'],
     ['Tea', 'Bread'],
     ['Pastry', 'Bread', 'Coffee'],
     ['Coffee', 'Empanadas', 'Muffin', 'Alfajores'],
     ['Coffee'],
     ['Bread'],
     ['Coffee'],
     ['Coffee'],
     ['Bread'],
     ['Pastry', 'Bread', 'Medialuna', 'Tea'],
     ['Tea', 'Bread'],
     ['Muffin'],
     ['Coffee', 'Muffin'],
     ['Pastry', 'Coffee', 'Medialuna', 'Bread'],
     ['Coffee', 'Muffin'],
     ['Bread', 'Hot chocolate', 'Scandinavian', 'Muffin'],
     ['Farm House'],
     ['Muffin'],
     ['Coffee'],
     ['Bread'],
     ['Cookies', 'Coffee'],
     ['Coffee'],
     ['Muffin'],
     ['Tea', 'Coffee', 'Keeping It Local'],
     ['Coffee', 'Medialuna'],
     ['Bread'],
     ['Tartine', 'Scandinavian'],
     ['Tea', 'Coffee'],
     ['Bread'],
     ['Coke', 'Coffee'],
     ['Coffee', 'Muffin'],
     ['Muffin', 'Medialuna', 'Coffee', 'Scandinavian', 'Bread'],
     ['Bread'],
     ['Coke'],
     ['Bread', 'Scandinavian'],
     ['Coffee'],
     ['Bread'],
     ['Bread', 'Scandinavian'],
     ['Fudge'],
     ['Coffee', 'Hearty & Seasonal', 'Soup'],
     ['Coke'],
     ['Cookies', 'Coffee', 'NONE', 'Tea'],
     ['Pastry', 'Coffee', 'Tea'],
     ['Bread'],
     ['Coffee', 'Medialuna'],
     ['Coffee', 'Sandwich', 'Muffin', 'Soup'],
     ['Tea', 'Coffee', 'Muffin'],
     ['Alfajores'],
     ['Bread'],
     ['Coffee', 'Medialuna'],
     ['Coffee'],
     ['Coffee'],
     ['Medialuna'],
     ['Bread', 'Empanadas', 'Scandinavian'],
     ['Coffee', 'Tartine'],
     ['Tea'],
     ['Scandinavian'],
     ['Muffin'],
     ['Bread', 'Scandinavian'],
     ['Bread'],
     ['Coffee'],
     ['Bread'],
     ['Bread'],
     ['Coffee'],
     ['Cookies', 'Scandinavian'],
     ['Medialuna', 'Coffee', 'Empanadas', 'Bread'],
     ['Pastry', 'Coffee'],
     ['Bread'],
     ['Pastry', 'Tea'],
     ['Pastry', 'Coffee'],
     ['Coffee', 'Bread'],
     ['Coffee', 'Keeping It Local'],
     ['Fudge'],
     ['Coffee', 'NONE', 'Tartine'],
     ['Pastry', 'Hot chocolate', 'Coffee', 'Medialuna'],
     ['Coffee', 'Alfajores'],
     ['Coffee'],
     ['Coffee', 'Medialuna'],
     ['Coke'],
     ['Coffee', 'Alfajores'],
     ['Muffin'],
     ['Brownie', 'Farm House', 'Fudge'],
     ['Coffee'],
     ['Tea', 'Farm House', 'Muffin'],
     ['Pastry', 'Coffee'],
     ['Pastry', 'Brownie'],
     ['Medialuna'],
     ['Coffee'],
     ['Pastry', 'Coffee', 'Juice'],
     ['Coffee', 'Bread'],
     ['Tea', 'Coffee'],
     ['Tea', 'Alfajores'],
     ['Coffee', 'Alfajores'],
     ['Scandinavian'],
     ['Pastry', 'Medialuna'],
     ['Brownie'],
     ['Farm House'],
     ['Cookies', 'Coffee'],
     ['Soup'],
     ['Coffee', 'Brownie', 'Art Tray'],
     ['Coffee'],
     ['Farm House'],
     ['Alfajores', 'Art Tray', 'Frittata', 'Soup'],
     ['Coffee', 'Sandwich'],
     ['Tea'],
     ['Farm House'],
     ['Scandinavian', 'Muffin'],
     ['Farm House'],
     ['Scandinavian'],
     ['Soup'],
     ['Tea', 'Bread', 'Hearty & Seasonal'],
     ['Hot chocolate'],
     ['Coffee', 'Sandwich', 'Hearty & Seasonal'],
     ['Coffee', 'Brownie', 'Sandwich', 'Hearty & Seasonal'],
     ['Medialuna', 'Coffee'],
     ['Coffee', 'Brownie', 'Soup'],
     ['Sandwich', 'Focaccia', 'Alfajores'],
     ['Farm House', 'Tartine'],
     ['Bread'],
     ['Coffee'],
     ['Tea', 'Hot chocolate', 'Muffin'],
     ['Pastry'],
     ['Bowl Nic Pitt', 'Hearty & Seasonal', 'Soup', 'Coffee', 'Brownie'],
     ['Coffee'],
     ['Medialuna', 'Farm House', 'Scandinavian', 'Alfajores'],
     ['Coffee', 'Soup'],
     ['Tea'],
     ['Tea', 'Coffee', 'Cookies'],
     ['Tea', 'Muffin'],
     ['Coffee', 'Brownie'],
     ['Coffee'],
     ['Coffee'],
     ['Coffee'],
     ['Farm House'],
     ['Coffee', 'Bread Pudding'],
     ['Coffee', 'Medialuna'],
     ['Hot chocolate'],
     ['Coffee', 'Farm House', 'Hot chocolate'],
     ['Pastry', 'Coffee'],
     ['Bread'],
     ['Pastry', 'Coffee'],
     ['Coffee'],
     ['Pastry', 'Coffee'],
     ['Farm House'],
     ['Bread'],
     ['Coffee', 'Hot chocolate'],
     ['Bread'],
     ['Bread', 'Coffee', 'Focaccia', 'Medialuna'],
     ['Coffee'],
     ['Bread', 'Keeping It Local'],
     ['Bread', 'Coffee'],
     ['Bread', 'NONE'],
     ['Bread', 'Medialuna'],
     ['Coffee'],
     ['Coffee', 'Medialuna'],
     ['Cookies', 'Coffee', 'Tartine'],
     ['Tea', 'Bread'],
     ['Bread', 'Coffee', 'Keeping It Local'],
     ['Coffee', 'Medialuna'],
     ['Coffee', 'Medialuna'],
     ['Tea'],
     ['Juice', 'Coke', 'Soup'],
     ['Bread'],
     ['Brownie'],
     ['Bread', 'Coffee', 'Brownie'],
     ['Pastry', 'Coffee', 'Bread', 'Hearty & Seasonal'],
     ['Bread', 'Coffee', 'Brownie'],
     ['Coffee'],
     ['Cookies', 'Bread'],
     ['Fudge', 'Eggs'],
     ['Bread'],
     ['Bread', 'Muffin'],
     ['Soup'],
     ['Coffee', 'Bread Pudding'],
     ['Bread', 'Tartine'],
     ['Fudge', 'Coffee', 'Bread Pudding', 'Jam', 'Sandwich'],
     ['Coffee', 'Bread Pudding', 'Soup'],
     ['Bread', 'Hearty & Seasonal'],
     ['Cookies', 'Coffee'],
     ['Coffee'],
     ['Bread'],
     ['Coffee', 'Bread', 'Soup'],
     ['Tea', 'Focaccia', 'Soup'],
     ['My-5 Fruit Shoot', 'Coffee', 'Cookies'],
     ['Coffee'],
     ['Tea', 'Coffee', 'Sandwich'],
     ['Scandinavian'],
     ['Adjustment'],
     ['Coffee'],
     ['Medialuna', 'Coffee'],
     ['Bread'],
     ['Bread'],
     ['Pastry', 'Coffee'],
     ['Farm House'],
     ['Medialuna', 'Brownie'],
     ['Pastry', 'Bread', 'Brownie'],
     ['Pastry', 'Bread', 'Cookies'],
     ['Bread'],
     ['Cookies', 'Coffee', 'Juice', 'Pastry'],
     ['Pastry', 'Coffee', 'Cookies'],
     ['Coffee', 'Medialuna'],
     ['Bread'],
     ['Coffee', 'Tartine'],
     ['Coffee', 'Medialuna'],
     ['Bread'],
     ['Cookies', 'Bread', 'Coffee', 'Alfajores'],
     ['Bread', 'Coffee'],
     ['Bread'],
     ['Cookies', 'Coffee', 'Jam', 'Keeping It Local', 'Tea', 'Bread'],
     ['Pastry', 'Coffee', 'Juice'],
     ['Medialuna', 'Focaccia', 'Bread'],
     ['Pastry', 'Medialuna', 'Coffee'],
     ['Coffee'],
     ['Coffee'],
     ['Cookies', 'Coffee', 'Brownie', 'Truffles', 'Bread'],
     ['Tea', 'Art Tray'],
     ['Bread'],
     ['Coffee'],
     ['Cookies', 'Coffee'],
     ['Brownie'],
     ['Tea', 'Bread', 'Empanadas'],
     ['Bread', 'Hot chocolate'],
     ['Cookies', 'Coffee'],
     ['Bread'],
     ['Tea', 'Bread', 'Tartine', 'Sandwich'],
     ['Bread', 'Sandwich', 'Soup'],
     ['Bread', 'Soup'],
     ['Fudge'],
     ['Coffee'],
     ['Coffee', 'Empanadas', 'Soup'],
     ['Bread', 'Coffee', 'Brownie'],
     ['Juice', 'Smoothies', 'Hearty & Seasonal', 'Soup', 'Coffee'],
     ['Cookies',
      'Coffee',
      'Scandinavian',
      'Alfajores',
      'Hot chocolate',
      'Sandwich'],
     ['Tea', 'Coffee'],
     ['Cookies', 'Bread'],
     ['Cookies', 'Bread', 'Alfajores'],
     ['Tea', 'Brownie'],
     ['Tea', 'Bread'],
     ['Coffee', 'Alfajores'],
     ['Bread'],
     ['Coffee', 'Empanadas'],
     ['Bread'],
     ['Coffee', 'Hot chocolate'],
     ['Coffee'],
     ['Coffee', 'Brownie'],
     ['Tea', 'Hot chocolate', 'Brownie', 'Truffles'],
     ['Coffee', 'Keeping It Local'],
     ['Coffee'],
     ['Alfajores'],
     ['Chimichurri Oil', 'Scandinavian'],
     ['Bread', 'Truffles'],
     ['Brownie', 'Focaccia'],
     ['Coffee', 'Bread'],
     ['Tea', 'Coffee', 'Cookies', 'Art Tray'],
     ['Coffee'],
     ['Bread'],
     ['Coke', 'Coffee', 'Alfajores'],
     ['Bread'],
     ['Bread'],
     ['Pastry', 'Bread', 'Coffee', 'Medialuna'],
     ['Coffee'],
     ['Bread', 'Keeping It Local'],
     ['Bread'],
     ['Coffee', 'Scandinavian'],
     ['Pastry', 'Bread', 'Farm House', 'Medialuna'],
     ['Medialuna'],
     ['Coffee'],
     ['Bread', 'Farm House'],
     ['Bread'],
     ['Bread', 'Keeping It Local'],
     ...]




```python
#转为DataFrame，以便做清洗和处理
shopping_df = pd.DataFrame(shopping_list)  

```

### 数据转换

mlxtend算法要将数据转为0/1或True/False的one-hot编码，这里需要先做处理


```python
from mlxtend.preprocessing import TransactionEncoder   #类似文本分析库，将数据库项转为数字编码
#转为数组
def deal(data):
    return data.dropna().tolist()
df_arr = shopping_df.apply(deal,axis=1).tolist()


#转为bool值
te = TransactionEncoder()   # 定义模型
df_data = te.fit_transform(df_arr)
dataset = pd.DataFrame(df_data,columns=te.columns_) 
```


```python
dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>Transaction</th>
      <th>Item</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2016-10-30</td>
      <td>09:58:11</td>
      <td>1</td>
      <td>Bread</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2016-10-30</td>
      <td>10:05:34</td>
      <td>2</td>
      <td>Scandinavian</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2016-10-30</td>
      <td>10:05:34</td>
      <td>2</td>
      <td>Scandinavian</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2016-10-30</td>
      <td>10:07:57</td>
      <td>3</td>
      <td>Hot chocolate</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2016-10-30</td>
      <td>10:07:57</td>
      <td>3</td>
      <td>Jam</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>21288</td>
      <td>2017-04-09</td>
      <td>14:32:58</td>
      <td>9682</td>
      <td>Coffee</td>
    </tr>
    <tr>
      <td>21289</td>
      <td>2017-04-09</td>
      <td>14:32:58</td>
      <td>9682</td>
      <td>Tea</td>
    </tr>
    <tr>
      <td>21290</td>
      <td>2017-04-09</td>
      <td>14:57:06</td>
      <td>9683</td>
      <td>Coffee</td>
    </tr>
    <tr>
      <td>21291</td>
      <td>2017-04-09</td>
      <td>14:57:06</td>
      <td>9683</td>
      <td>Pastry</td>
    </tr>
    <tr>
      <td>21292</td>
      <td>2017-04-09</td>
      <td>15:04:24</td>
      <td>9684</td>
      <td>Smoothies</td>
    </tr>
  </tbody>
</table>
<p>21293 rows × 4 columns</p>
</div>



###  apriori算法


```python
#调用mlxtend里的apriori算法
from mlxtend.frequent_patterns import apriori
 
frequent_itemsets = apriori(dataset,min_support=0.03,use_colnames=True)    # use_colnames=True表示使用元素名字，默认的False使用列名代表元素
#frequent_itemsets = apriori(df,min_support=0.05)
frequent_itemsets.sort_values(by='support',ascending=False,inplace=True)# 频繁项集可以按支持度排序
```


```python
#输出关联规则
from mlxtend.frequent_patterns import association_rules
 
association_rule = association_rules(frequent_itemsets,metric='lift',min_threshold=1)   # metric可以有很多的度量选项，返回的表列名都可以作为参数
association_rule.sort_values(by='leverage',ascending=False,inplace=True)    #关联规则可以按leverage排序
print(association_rule)  #有提升的
```

       antecedents  consequents  antecedent support  consequent support   support  \
    2     (Coffee)     (Pastry)            0.478394            0.086107  0.047544   
    3     (Pastry)     (Coffee)            0.086107            0.478394  0.047544   
    6  (Medialuna)     (Coffee)            0.061807            0.478394  0.035182   
    7     (Coffee)  (Medialuna)            0.478394            0.061807  0.035182   
    0       (Cake)     (Coffee)            0.103856            0.478394  0.054728   
    1     (Coffee)       (Cake)            0.478394            0.103856  0.054728   
    4   (Sandwich)     (Coffee)            0.071844            0.478394  0.038246   
    5     (Coffee)   (Sandwich)            0.478394            0.071844  0.038246   
    
       confidence      lift  leverage  conviction  
    2    0.099382  1.154168  0.006351    1.014740  
    3    0.552147  1.154168  0.006351    1.164682  
    6    0.569231  1.189878  0.005614    1.210871  
    7    0.073542  1.189878  0.005614    1.012667  
    0    0.526958  1.101515  0.005044    1.102664  
    1    0.114399  1.101515  0.005044    1.011905  
    4    0.532353  1.112792  0.003877    1.115384  
    5    0.079947  1.112792  0.003877    1.008807  


### fpgrowth算法
mlxtend里也加了fpgrowth的实现，这里也试试


```python
from mlxtend.frequent_patterns import fpgrowth
 
frequent_itemsets = fpgrowth(dataset,min_support=0.03,use_colnames=True)    # use_colnames=True表示使用元素名字，默认的False使用列名代表元素
#frequent_itemsets = apriori(df,min_support=0.05)
frequent_itemsets.sort_values(by='support',ascending=False,inplace=True)# 频繁项集可以按支持度排序
```


```python
#输出关联规则
from mlxtend.frequent_patterns import association_rules
 
association_rule = association_rules(frequent_itemsets,metric='lift',min_threshold=1)   # metric可以有很多的度量选项，返回的表列名都可以作为参数
association_rule.sort_values(by='leverage',ascending=False,inplace=True)    #关联规则可以按leverage排序
print(association_rule)  #有提升的
```

       antecedents  consequents  antecedent support  consequent support   support  \
    2     (Coffee)     (Pastry)            0.478394            0.086107  0.047544   
    3     (Pastry)     (Coffee)            0.086107            0.478394  0.047544   
    6  (Medialuna)     (Coffee)            0.061807            0.478394  0.035182   
    7     (Coffee)  (Medialuna)            0.478394            0.061807  0.035182   
    0       (Cake)     (Coffee)            0.103856            0.478394  0.054728   
    1     (Coffee)       (Cake)            0.478394            0.103856  0.054728   
    4   (Sandwich)     (Coffee)            0.071844            0.478394  0.038246   
    5     (Coffee)   (Sandwich)            0.478394            0.071844  0.038246   
    
       confidence      lift  leverage  conviction  
    2    0.099382  1.154168  0.006351    1.014740  
    3    0.552147  1.154168  0.006351    1.164682  
    6    0.569231  1.189878  0.005614    1.210871  
    7    0.073542  1.189878  0.005614    1.012667  
    0    0.526958  1.101515  0.005044    1.102664  
    1    0.114399  1.101515  0.005044    1.011905  
    4    0.532353  1.112792  0.003877    1.115384  
    5    0.079947  1.112792  0.003877    1.008807  


###  输出关系图
这里使用NetworkX去构建网络图，以直观表示商品间的关系


```python
import networkx as nx 
fig, ax=plt.subplots(figsize=(10,4))
GA = nx.from_pandas_edgelist(association_rule,source='antecedents',target='consequents')
nx.draw(GA,with_labels=True)
plt.show()
```


![png](output_56_0.png)


### 商业分析

coffee是面包店最畅销的商品，并且与 4种商品- pastry, cake, medialuna and sandwich有关联关系。考虑到coffee与4种商品之间的关系，建议面包店可以采取一些策略来增加销售额。

4种商品中的任何一个促销折扣或是优惠券都可以吸引顾客购买咖啡。  
将这4种商品放在咖啡订购台附近可能会吸引顾客购买。  
适当得指定一些食谱，如：咖啡蛋糕或咖啡糕点。  
可以增加套餐或是会员储值优惠活动


```python

```
