

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
df_data=pd.read_csv('data.csv',index_col=0)   # 将之前的csv改成 csv utf-8读取，不然报错
```


```python
df_data.drop_duplicates(inplace=True)
```


```python
df_data.head()
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
      <th>custid</th>
      <th>trade_no</th>
      <th>bank_card_no</th>
      <th>low_volume_percent</th>
      <th>middle_volume_percent</th>
      <th>take_amount_in_later_12_month_highest</th>
      <th>trans_amount_increase_rate_lately</th>
      <th>trans_activity_month</th>
      <th>trans_activity_day</th>
      <th>transd_mcc</th>
      <th>...</th>
      <th>loans_max_limit</th>
      <th>loans_avg_limit</th>
      <th>consfin_credit_limit</th>
      <th>consfin_credibility</th>
      <th>consfin_org_count_current</th>
      <th>consfin_product_count</th>
      <th>consfin_max_limit</th>
      <th>consfin_avg_limit</th>
      <th>latest_query_day</th>
      <th>loans_latest_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>2791858</td>
      <td>20180507115231274000000023057383</td>
      <td>卡号1</td>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.90</td>
      <td>0.55</td>
      <td>0.313</td>
      <td>17.0</td>
      <td>...</td>
      <td>2900.0</td>
      <td>1688.0</td>
      <td>1200.0</td>
      <td>75.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1200.0</td>
      <td>1200.0</td>
      <td>12.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>534047</td>
      <td>20180507121002192000000023073000</td>
      <td>卡号1</td>
      <td>0.02</td>
      <td>0.94</td>
      <td>2000</td>
      <td>1.28</td>
      <td>1.00</td>
      <td>0.458</td>
      <td>19.0</td>
      <td>...</td>
      <td>3500.0</td>
      <td>1758.0</td>
      <td>15100.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>22800.0</td>
      <td>9360.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2849787</td>
      <td>20180507125159718000000023114911</td>
      <td>卡号1</td>
      <td>0.04</td>
      <td>0.96</td>
      <td>0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.114</td>
      <td>13.0</td>
      <td>...</td>
      <td>1600.0</td>
      <td>1250.0</td>
      <td>4200.0</td>
      <td>87.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4200.0</td>
      <td>4200.0</td>
      <td>2.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1809708</td>
      <td>20180507121358683000000388283484</td>
      <td>卡号1</td>
      <td>0.00</td>
      <td>0.96</td>
      <td>2000</td>
      <td>0.13</td>
      <td>0.57</td>
      <td>0.777</td>
      <td>22.0</td>
      <td>...</td>
      <td>3200.0</td>
      <td>1541.0</td>
      <td>16300.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>30000.0</td>
      <td>12180.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2499829</td>
      <td>20180507115448545000000388205844</td>
      <td>卡号1</td>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.46</td>
      <td>1.00</td>
      <td>0.175</td>
      <td>13.0</td>
      <td>...</td>
      <td>2300.0</td>
      <td>1630.0</td>
      <td>8300.0</td>
      <td>79.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>8400.0</td>
      <td>8250.0</td>
      <td>22.0</td>
      <td>120.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 89 columns</p>
</div>




```python
df_data.shape  #有4754个样本，89个特征
```




    (4754, 89)



# 1.对数据类型进行分析


```python
# 查看object，int,float 类型有哪些，后期需要做处理
def find_type(df_data):
    object_columns=[]
    for i in df_data.columns:
        if str(df_data[i].dtypes)=='object':
            object_columns.append(i)

    int_columns=[]
    for i in df_data.columns:
        if str(df_data[i].dtypes)=='int64':
            int_columns.append(i)

    float_columns=[]
    for i in df_data.columns:
        if str(df_data[i].dtypes)=='float64':
            float_columns.append(i)

    print(object_columns)   
    print('-----------')
    print(int_columns)
    print('-----------')
    print(float_columns)
    return object_columns,int_columns,float_columns
object_columns,int_columns,float_columns=find_type(df_data)
```

    ['trade_no', 'bank_card_no', 'reg_preference_for_trad', 'source', 'id_name', 'latest_query_time', 'loans_latest_time']
    -----------
    ['custid', 'take_amount_in_later_12_month_highest', 'repayment_capability', 'is_high_user', 'historical_trans_amount', 'trans_amount_3_month', 'abs', 'avg_price_last_12_month', 'max_cumulative_consume_later_1_month', 'pawns_auctions_trusts_consume_last_1_month', 'pawns_auctions_trusts_consume_last_6_month', 'status']
    -----------
    ['low_volume_percent', 'middle_volume_percent', 'trans_amount_increase_rate_lately', 'trans_activity_month', 'trans_activity_day', 'transd_mcc', 'trans_days_interval_filter', 'trans_days_interval', 'regional_mobility', 'student_feature', 'number_of_trans_from_2011', 'first_transaction_time', 'historical_trans_day', 'rank_trad_1_month', 'avg_consume_less_12_valid_month', 'top_trans_count_last_1_month', 'avg_price_top_last_12_valid_month', 'trans_top_time_last_1_month', 'trans_top_time_last_6_month', 'consume_top_time_last_1_month', 'consume_top_time_last_6_month', 'cross_consume_count_last_1_month', 'trans_fail_top_count_enum_last_1_month', 'trans_fail_top_count_enum_last_6_month', 'trans_fail_top_count_enum_last_12_month', 'consume_mini_time_last_1_month', 'max_consume_count_later_6_month', 'railway_consume_count_last_12_month', 'jewelry_consume_count_last_6_month', 'first_transaction_day', 'trans_day_last_12_month', 'apply_score', 'apply_credibility', 'query_org_count', 'query_finance_count', 'query_cash_count', 'query_sum_count', 'latest_one_month_apply', 'latest_three_month_apply', 'latest_six_month_apply', 'loans_score', 'loans_credibility_behavior', 'loans_count', 'loans_settle_count', 'loans_overdue_count', 'loans_org_count_behavior', 'consfin_org_count_behavior', 'loans_cash_count', 'latest_one_month_loan', 'latest_three_month_loan', 'latest_six_month_loan', 'history_suc_fee', 'history_fail_fee', 'latest_one_month_suc', 'latest_one_month_fail', 'loans_long_time', 'loans_credit_limit', 'loans_credibility_limit', 'loans_org_count_current', 'loans_product_count', 'loans_max_limit', 'loans_avg_limit', 'consfin_credit_limit', 'consfin_credibility', 'consfin_org_count_current', 'consfin_product_count', 'consfin_max_limit', 'consfin_avg_limit', 'latest_query_day', 'loans_latest_day']
    


```python
cancel_columns=[]  # 创建list 存储 需要删除的列
```


```python
df_data[object_columns].head()  #查看object 类型大概样子
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
      <th>trade_no</th>
      <th>bank_card_no</th>
      <th>reg_preference_for_trad</th>
      <th>source</th>
      <th>id_name</th>
      <th>latest_query_time</th>
      <th>loans_latest_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>20180507115231274000000023057383</td>
      <td>卡号1</td>
      <td>一线城市</td>
      <td>xs</td>
      <td>蒋红</td>
      <td>2018-04-25</td>
      <td>2018-04-19</td>
    </tr>
    <tr>
      <th>10</th>
      <td>20180507121002192000000023073000</td>
      <td>卡号1</td>
      <td>一线城市</td>
      <td>xs</td>
      <td>崔向朝</td>
      <td>2018-05-03</td>
      <td>2018-05-05</td>
    </tr>
    <tr>
      <th>12</th>
      <td>20180507125159718000000023114911</td>
      <td>卡号1</td>
      <td>一线城市</td>
      <td>xs</td>
      <td>王中云</td>
      <td>2018-05-05</td>
      <td>2018-05-01</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20180507121358683000000388283484</td>
      <td>卡号1</td>
      <td>三线城市</td>
      <td>xs</td>
      <td>何洋洋</td>
      <td>2018-05-05</td>
      <td>2018-05-03</td>
    </tr>
    <tr>
      <th>14</th>
      <td>20180507115448545000000388205844</td>
      <td>卡号1</td>
      <td>一线城市</td>
      <td>xs</td>
      <td>赵洋</td>
      <td>2018-04-15</td>
      <td>2018-01-07</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_data['trade_no'].value_counts().count()  # 应该删除该列，没有任何意义
```




    4754




```python
cancel_columns.append('trade_no')
```


```python
df_data['bank_card_no'].value_counts()  # 应该删除该列，没有任何意义
```




    卡号1    4754
    Name: bank_card_no, dtype: int64




```python
cancel_columns.append('bank_card_no')
```


```python
df_data['reg_preference_for_trad'].value_counts()
```




    一线城市    3403
    三线城市    1064
    境外       150
    二线城市     131
    其他城市       4
    Name: reg_preference_for_trad, dtype: int64




```python
df_data['source'].value_counts() # 应该删除该列，都一样
```




    xs    4754
    Name: source, dtype: int64




```python
cancel_columns.append('source')
cancel_columns.append('id_name')
```


```python
df_data[int_columns].head()
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
      <th>custid</th>
      <th>take_amount_in_later_12_month_highest</th>
      <th>repayment_capability</th>
      <th>is_high_user</th>
      <th>historical_trans_amount</th>
      <th>trans_amount_3_month</th>
      <th>abs</th>
      <th>avg_price_last_12_month</th>
      <th>max_cumulative_consume_later_1_month</th>
      <th>pawns_auctions_trusts_consume_last_1_month</th>
      <th>pawns_auctions_trusts_consume_last_6_month</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>2791858</td>
      <td>0</td>
      <td>19890</td>
      <td>0</td>
      <td>149050</td>
      <td>34030</td>
      <td>3920</td>
      <td>1020</td>
      <td>2170</td>
      <td>1970</td>
      <td>18040</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>534047</td>
      <td>2000</td>
      <td>16970</td>
      <td>0</td>
      <td>302910</td>
      <td>10590</td>
      <td>6950</td>
      <td>1210</td>
      <td>2100</td>
      <td>1820</td>
      <td>15680</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2849787</td>
      <td>0</td>
      <td>9710</td>
      <td>0</td>
      <td>11520</td>
      <td>5710</td>
      <td>840</td>
      <td>570</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1809708</td>
      <td>2000</td>
      <td>6210</td>
      <td>0</td>
      <td>491130</td>
      <td>91690</td>
      <td>46850</td>
      <td>1290</td>
      <td>8140</td>
      <td>2700</td>
      <td>27970</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2499829</td>
      <td>0</td>
      <td>11150</td>
      <td>0</td>
      <td>61470</td>
      <td>9770</td>
      <td>760</td>
      <td>1110</td>
      <td>1000</td>
      <td>0</td>
      <td>6410</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
cancel_columns.append('custid')
```


```python
df_data[float_columns].head()
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
      <th>low_volume_percent</th>
      <th>middle_volume_percent</th>
      <th>trans_amount_increase_rate_lately</th>
      <th>trans_activity_month</th>
      <th>trans_activity_day</th>
      <th>transd_mcc</th>
      <th>trans_days_interval_filter</th>
      <th>trans_days_interval</th>
      <th>regional_mobility</th>
      <th>student_feature</th>
      <th>...</th>
      <th>loans_max_limit</th>
      <th>loans_avg_limit</th>
      <th>consfin_credit_limit</th>
      <th>consfin_credibility</th>
      <th>consfin_org_count_current</th>
      <th>consfin_product_count</th>
      <th>consfin_max_limit</th>
      <th>consfin_avg_limit</th>
      <th>latest_query_day</th>
      <th>loans_latest_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0.90</td>
      <td>0.55</td>
      <td>0.313</td>
      <td>17.0</td>
      <td>27.0</td>
      <td>26.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>2900.0</td>
      <td>1688.0</td>
      <td>1200.0</td>
      <td>75.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1200.0</td>
      <td>1200.0</td>
      <td>12.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.02</td>
      <td>0.94</td>
      <td>1.28</td>
      <td>1.00</td>
      <td>0.458</td>
      <td>19.0</td>
      <td>30.0</td>
      <td>14.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>3500.0</td>
      <td>1758.0</td>
      <td>15100.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>22800.0</td>
      <td>9360.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.04</td>
      <td>0.96</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.114</td>
      <td>13.0</td>
      <td>68.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>1600.0</td>
      <td>1250.0</td>
      <td>4200.0</td>
      <td>87.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4200.0</td>
      <td>4200.0</td>
      <td>2.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.00</td>
      <td>0.96</td>
      <td>0.13</td>
      <td>0.57</td>
      <td>0.777</td>
      <td>22.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>3200.0</td>
      <td>1541.0</td>
      <td>16300.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>30000.0</td>
      <td>12180.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0.46</td>
      <td>1.00</td>
      <td>0.175</td>
      <td>13.0</td>
      <td>66.0</td>
      <td>42.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>2300.0</td>
      <td>1630.0</td>
      <td>8300.0</td>
      <td>79.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>8400.0</td>
      <td>8250.0</td>
      <td>22.0</td>
      <td>120.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 70 columns</p>
</div>



# 2.删除无关的列


```python
cancel_columns
```




    ['trade_no', 'bank_card_no', 'source', 'id_name', 'custid']




```python
df=df_data
df=df.drop(cancel_columns,axis=1)
```


```python
df.head()
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
      <th>low_volume_percent</th>
      <th>middle_volume_percent</th>
      <th>take_amount_in_later_12_month_highest</th>
      <th>trans_amount_increase_rate_lately</th>
      <th>trans_activity_month</th>
      <th>trans_activity_day</th>
      <th>transd_mcc</th>
      <th>trans_days_interval_filter</th>
      <th>trans_days_interval</th>
      <th>regional_mobility</th>
      <th>...</th>
      <th>loans_max_limit</th>
      <th>loans_avg_limit</th>
      <th>consfin_credit_limit</th>
      <th>consfin_credibility</th>
      <th>consfin_org_count_current</th>
      <th>consfin_product_count</th>
      <th>consfin_max_limit</th>
      <th>consfin_avg_limit</th>
      <th>latest_query_day</th>
      <th>loans_latest_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.90</td>
      <td>0.55</td>
      <td>0.313</td>
      <td>17.0</td>
      <td>27.0</td>
      <td>26.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>2900.0</td>
      <td>1688.0</td>
      <td>1200.0</td>
      <td>75.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1200.0</td>
      <td>1200.0</td>
      <td>12.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.02</td>
      <td>0.94</td>
      <td>2000</td>
      <td>1.28</td>
      <td>1.00</td>
      <td>0.458</td>
      <td>19.0</td>
      <td>30.0</td>
      <td>14.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>3500.0</td>
      <td>1758.0</td>
      <td>15100.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>22800.0</td>
      <td>9360.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.04</td>
      <td>0.96</td>
      <td>0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.114</td>
      <td>13.0</td>
      <td>68.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1600.0</td>
      <td>1250.0</td>
      <td>4200.0</td>
      <td>87.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4200.0</td>
      <td>4200.0</td>
      <td>2.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.00</td>
      <td>0.96</td>
      <td>2000</td>
      <td>0.13</td>
      <td>0.57</td>
      <td>0.777</td>
      <td>22.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>3200.0</td>
      <td>1541.0</td>
      <td>16300.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>30000.0</td>
      <td>12180.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.46</td>
      <td>1.00</td>
      <td>0.175</td>
      <td>13.0</td>
      <td>66.0</td>
      <td>42.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>2300.0</td>
      <td>1630.0</td>
      <td>8300.0</td>
      <td>79.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>8400.0</td>
      <td>8250.0</td>
      <td>22.0</td>
      <td>120.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 84 columns</p>
</div>



# 3.数据类型转换+填补缺失值


```python
object_columns,int_columns,float_columns=find_type(df)
```

    ['reg_preference_for_trad', 'latest_query_time', 'loans_latest_time']
    -----------
    ['take_amount_in_later_12_month_highest', 'repayment_capability', 'is_high_user', 'historical_trans_amount', 'trans_amount_3_month', 'abs', 'avg_price_last_12_month', 'max_cumulative_consume_later_1_month', 'pawns_auctions_trusts_consume_last_1_month', 'pawns_auctions_trusts_consume_last_6_month', 'status']
    -----------
    ['low_volume_percent', 'middle_volume_percent', 'trans_amount_increase_rate_lately', 'trans_activity_month', 'trans_activity_day', 'transd_mcc', 'trans_days_interval_filter', 'trans_days_interval', 'regional_mobility', 'student_feature', 'number_of_trans_from_2011', 'first_transaction_time', 'historical_trans_day', 'rank_trad_1_month', 'avg_consume_less_12_valid_month', 'top_trans_count_last_1_month', 'avg_price_top_last_12_valid_month', 'trans_top_time_last_1_month', 'trans_top_time_last_6_month', 'consume_top_time_last_1_month', 'consume_top_time_last_6_month', 'cross_consume_count_last_1_month', 'trans_fail_top_count_enum_last_1_month', 'trans_fail_top_count_enum_last_6_month', 'trans_fail_top_count_enum_last_12_month', 'consume_mini_time_last_1_month', 'max_consume_count_later_6_month', 'railway_consume_count_last_12_month', 'jewelry_consume_count_last_6_month', 'first_transaction_day', 'trans_day_last_12_month', 'apply_score', 'apply_credibility', 'query_org_count', 'query_finance_count', 'query_cash_count', 'query_sum_count', 'latest_one_month_apply', 'latest_three_month_apply', 'latest_six_month_apply', 'loans_score', 'loans_credibility_behavior', 'loans_count', 'loans_settle_count', 'loans_overdue_count', 'loans_org_count_behavior', 'consfin_org_count_behavior', 'loans_cash_count', 'latest_one_month_loan', 'latest_three_month_loan', 'latest_six_month_loan', 'history_suc_fee', 'history_fail_fee', 'latest_one_month_suc', 'latest_one_month_fail', 'loans_long_time', 'loans_credit_limit', 'loans_credibility_limit', 'loans_org_count_current', 'loans_product_count', 'loans_max_limit', 'loans_avg_limit', 'consfin_credit_limit', 'consfin_credibility', 'consfin_org_count_current', 'consfin_product_count', 'consfin_max_limit', 'consfin_avg_limit', 'latest_query_day', 'loans_latest_day']
    


```python
df[object_columns].head()
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
      <th>reg_preference_for_trad</th>
      <th>latest_query_time</th>
      <th>loans_latest_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>一线城市</td>
      <td>2018-04-25</td>
      <td>2018-04-19</td>
    </tr>
    <tr>
      <th>10</th>
      <td>一线城市</td>
      <td>2018-05-03</td>
      <td>2018-05-05</td>
    </tr>
    <tr>
      <th>12</th>
      <td>一线城市</td>
      <td>2018-05-05</td>
      <td>2018-05-01</td>
    </tr>
    <tr>
      <th>13</th>
      <td>三线城市</td>
      <td>2018-05-05</td>
      <td>2018-05-03</td>
    </tr>
    <tr>
      <th>14</th>
      <td>一线城市</td>
      <td>2018-04-15</td>
      <td>2018-01-07</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[object_columns].isnull().sum()
```




    reg_preference_for_trad      2
    latest_query_time          304
    loans_latest_time          297
    dtype: int64




```python
df['reg_preference_for_trad'].value_counts()
```




    一线城市    3403
    三线城市    1064
    境外       150
    二线城市     131
    其他城市       4
    Name: reg_preference_for_trad, dtype: int64




```python
df['reg_preference_for_trad']=df['reg_preference_for_trad'].fillna('其他城市')  #用其他城市代替缺失值
```


```python
# 时间序列不知道如何处理并且填补缺失值
```


```python
#对 reg_preference_for_trad 做 one_hot_encoder
df=pd.get_dummies(df,columns=['reg_preference_for_trad'])
```


```python
object_columns,int_columns,float_columns=find_type(df)
```

    ['latest_query_time', 'loans_latest_time']
    -----------
    ['take_amount_in_later_12_month_highest', 'repayment_capability', 'is_high_user', 'historical_trans_amount', 'trans_amount_3_month', 'abs', 'avg_price_last_12_month', 'max_cumulative_consume_later_1_month', 'pawns_auctions_trusts_consume_last_1_month', 'pawns_auctions_trusts_consume_last_6_month', 'status']
    -----------
    ['low_volume_percent', 'middle_volume_percent', 'trans_amount_increase_rate_lately', 'trans_activity_month', 'trans_activity_day', 'transd_mcc', 'trans_days_interval_filter', 'trans_days_interval', 'regional_mobility', 'student_feature', 'number_of_trans_from_2011', 'first_transaction_time', 'historical_trans_day', 'rank_trad_1_month', 'avg_consume_less_12_valid_month', 'top_trans_count_last_1_month', 'avg_price_top_last_12_valid_month', 'trans_top_time_last_1_month', 'trans_top_time_last_6_month', 'consume_top_time_last_1_month', 'consume_top_time_last_6_month', 'cross_consume_count_last_1_month', 'trans_fail_top_count_enum_last_1_month', 'trans_fail_top_count_enum_last_6_month', 'trans_fail_top_count_enum_last_12_month', 'consume_mini_time_last_1_month', 'max_consume_count_later_6_month', 'railway_consume_count_last_12_month', 'jewelry_consume_count_last_6_month', 'first_transaction_day', 'trans_day_last_12_month', 'apply_score', 'apply_credibility', 'query_org_count', 'query_finance_count', 'query_cash_count', 'query_sum_count', 'latest_one_month_apply', 'latest_three_month_apply', 'latest_six_month_apply', 'loans_score', 'loans_credibility_behavior', 'loans_count', 'loans_settle_count', 'loans_overdue_count', 'loans_org_count_behavior', 'consfin_org_count_behavior', 'loans_cash_count', 'latest_one_month_loan', 'latest_three_month_loan', 'latest_six_month_loan', 'history_suc_fee', 'history_fail_fee', 'latest_one_month_suc', 'latest_one_month_fail', 'loans_long_time', 'loans_credit_limit', 'loans_credibility_limit', 'loans_org_count_current', 'loans_product_count', 'loans_max_limit', 'loans_avg_limit', 'consfin_credit_limit', 'consfin_credibility', 'consfin_org_count_current', 'consfin_product_count', 'consfin_max_limit', 'consfin_avg_limit', 'latest_query_day', 'loans_latest_day']
    


```python
df.head()
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
      <th>low_volume_percent</th>
      <th>middle_volume_percent</th>
      <th>take_amount_in_later_12_month_highest</th>
      <th>trans_amount_increase_rate_lately</th>
      <th>trans_activity_month</th>
      <th>trans_activity_day</th>
      <th>transd_mcc</th>
      <th>trans_days_interval_filter</th>
      <th>trans_days_interval</th>
      <th>regional_mobility</th>
      <th>...</th>
      <th>consfin_product_count</th>
      <th>consfin_max_limit</th>
      <th>consfin_avg_limit</th>
      <th>latest_query_day</th>
      <th>loans_latest_day</th>
      <th>reg_preference_for_trad_一线城市</th>
      <th>reg_preference_for_trad_三线城市</th>
      <th>reg_preference_for_trad_二线城市</th>
      <th>reg_preference_for_trad_其他城市</th>
      <th>reg_preference_for_trad_境外</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.90</td>
      <td>0.55</td>
      <td>0.313</td>
      <td>17.0</td>
      <td>27.0</td>
      <td>26.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1200.0</td>
      <td>1200.0</td>
      <td>12.0</td>
      <td>18.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.02</td>
      <td>0.94</td>
      <td>2000</td>
      <td>1.28</td>
      <td>1.00</td>
      <td>0.458</td>
      <td>19.0</td>
      <td>30.0</td>
      <td>14.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>6.0</td>
      <td>22800.0</td>
      <td>9360.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.04</td>
      <td>0.96</td>
      <td>0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.114</td>
      <td>13.0</td>
      <td>68.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>4200.0</td>
      <td>4200.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.00</td>
      <td>0.96</td>
      <td>2000</td>
      <td>0.13</td>
      <td>0.57</td>
      <td>0.777</td>
      <td>22.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>5.0</td>
      <td>30000.0</td>
      <td>12180.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.46</td>
      <td>1.00</td>
      <td>0.175</td>
      <td>13.0</td>
      <td>66.0</td>
      <td>42.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>8400.0</td>
      <td>8250.0</td>
      <td>22.0</td>
      <td>120.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 88 columns</p>
</div>




```python
df[int_columns].head()
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
      <th>take_amount_in_later_12_month_highest</th>
      <th>repayment_capability</th>
      <th>is_high_user</th>
      <th>historical_trans_amount</th>
      <th>trans_amount_3_month</th>
      <th>abs</th>
      <th>avg_price_last_12_month</th>
      <th>max_cumulative_consume_later_1_month</th>
      <th>pawns_auctions_trusts_consume_last_1_month</th>
      <th>pawns_auctions_trusts_consume_last_6_month</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>19890</td>
      <td>0</td>
      <td>149050</td>
      <td>34030</td>
      <td>3920</td>
      <td>1020</td>
      <td>2170</td>
      <td>1970</td>
      <td>18040</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2000</td>
      <td>16970</td>
      <td>0</td>
      <td>302910</td>
      <td>10590</td>
      <td>6950</td>
      <td>1210</td>
      <td>2100</td>
      <td>1820</td>
      <td>15680</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>9710</td>
      <td>0</td>
      <td>11520</td>
      <td>5710</td>
      <td>840</td>
      <td>570</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2000</td>
      <td>6210</td>
      <td>0</td>
      <td>491130</td>
      <td>91690</td>
      <td>46850</td>
      <td>1290</td>
      <td>8140</td>
      <td>2700</td>
      <td>27970</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>11150</td>
      <td>0</td>
      <td>61470</td>
      <td>9770</td>
      <td>760</td>
      <td>1110</td>
      <td>1000</td>
      <td>0</td>
      <td>6410</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[int_columns].isnull().sum()  # int类型没有缺失值
```




    take_amount_in_later_12_month_highest         0
    repayment_capability                          0
    is_high_user                                  0
    historical_trans_amount                       0
    trans_amount_3_month                          0
    abs                                           0
    avg_price_last_12_month                       0
    max_cumulative_consume_later_1_month          0
    pawns_auctions_trusts_consume_last_1_month    0
    pawns_auctions_trusts_consume_last_6_month    0
    status                                        0
    dtype: int64




```python
df.isnull().sum().sort_values(ascending=False)[:10]
```




    student_feature                     2998
    cross_consume_count_last_1_month     426
    apply_credibility                    304
    query_cash_count                     304
    latest_six_month_apply               304
    latest_one_month_apply               304
    latest_query_time                    304
    query_sum_count                      304
    latest_three_month_apply             304
    query_finance_count                  304
    dtype: int64




```python
df['student_feature'].value_counts()
```




    1.0    1754
    2.0       2
    Name: student_feature, dtype: int64




```python
df['student_feature'].fillna(0,inplace=True)  #用0填充
```


```python
# 对于float类型的特征，填补缺失值
for i in float_columns:
    df[i].fillna(df[i].median(),inplace=True)
```


```python
df.head()
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
      <th>low_volume_percent</th>
      <th>middle_volume_percent</th>
      <th>take_amount_in_later_12_month_highest</th>
      <th>trans_amount_increase_rate_lately</th>
      <th>trans_activity_month</th>
      <th>trans_activity_day</th>
      <th>transd_mcc</th>
      <th>trans_days_interval_filter</th>
      <th>trans_days_interval</th>
      <th>regional_mobility</th>
      <th>...</th>
      <th>consfin_product_count</th>
      <th>consfin_max_limit</th>
      <th>consfin_avg_limit</th>
      <th>latest_query_day</th>
      <th>loans_latest_day</th>
      <th>reg_preference_for_trad_一线城市</th>
      <th>reg_preference_for_trad_三线城市</th>
      <th>reg_preference_for_trad_二线城市</th>
      <th>reg_preference_for_trad_其他城市</th>
      <th>reg_preference_for_trad_境外</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.90</td>
      <td>0.55</td>
      <td>0.313</td>
      <td>17.0</td>
      <td>27.0</td>
      <td>26.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>1200.0</td>
      <td>1200.0</td>
      <td>12.0</td>
      <td>18.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.02</td>
      <td>0.94</td>
      <td>2000</td>
      <td>1.28</td>
      <td>1.00</td>
      <td>0.458</td>
      <td>19.0</td>
      <td>30.0</td>
      <td>14.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>6.0</td>
      <td>22800.0</td>
      <td>9360.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.04</td>
      <td>0.96</td>
      <td>0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.114</td>
      <td>13.0</td>
      <td>68.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>4200.0</td>
      <td>4200.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.00</td>
      <td>0.96</td>
      <td>2000</td>
      <td>0.13</td>
      <td>0.57</td>
      <td>0.777</td>
      <td>22.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>5.0</td>
      <td>30000.0</td>
      <td>12180.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.46</td>
      <td>1.00</td>
      <td>0.175</td>
      <td>13.0</td>
      <td>66.0</td>
      <td>42.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>8400.0</td>
      <td>8250.0</td>
      <td>22.0</td>
      <td>120.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 88 columns</p>
</div>




```python
df[object_columns].head(3)
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
      <th>latest_query_time</th>
      <th>loans_latest_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>2018-04-25</td>
      <td>2018-04-19</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018-05-03</td>
      <td>2018-05-05</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2018-05-05</td>
      <td>2018-05-01</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['latest_query_time']=df['latest_query_time'].apply(lambda x:float(str(x)[:4]+str(x)[5:7]+str(x)[8:]))
df['loans_latest_time']=df['loans_latest_time'].apply(lambda x:float(str(x)[:4]+str(x)[5:7]+str(x)[8:]))
```


```python
# 对于转换的float类型的特征，填补缺失值
for i in object_columns:
    df[i].fillna(df[i].median(),inplace=True)
```


```python
df['reg_preference_for_trad_一线城市']=df['reg_preference_for_trad_一线城市'].apply(lambda x:float(x))
df['reg_preference_for_trad_二线城市']=df['reg_preference_for_trad_二线城市'].apply(lambda x:float(x))
df['reg_preference_for_trad_三线城市']=df['reg_preference_for_trad_三线城市'].apply(lambda x:float(x))
df['reg_preference_for_trad_其他城市']=df['reg_preference_for_trad_其他城市'].apply(lambda x:float(x))
df['reg_preference_for_trad_境外']=df['reg_preference_for_trad_境外'].apply(lambda x:float(x))
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4754 entries, 5 to 11992
    Data columns (total 88 columns):
    low_volume_percent                            4754 non-null float64
    middle_volume_percent                         4754 non-null float64
    take_amount_in_later_12_month_highest         4754 non-null int64
    trans_amount_increase_rate_lately             4754 non-null float64
    trans_activity_month                          4754 non-null float64
    trans_activity_day                            4754 non-null float64
    transd_mcc                                    4754 non-null float64
    trans_days_interval_filter                    4754 non-null float64
    trans_days_interval                           4754 non-null float64
    regional_mobility                             4754 non-null float64
    student_feature                               4754 non-null float64
    repayment_capability                          4754 non-null int64
    is_high_user                                  4754 non-null int64
    number_of_trans_from_2011                     4754 non-null float64
    first_transaction_time                        4754 non-null float64
    historical_trans_amount                       4754 non-null int64
    historical_trans_day                          4754 non-null float64
    rank_trad_1_month                             4754 non-null float64
    trans_amount_3_month                          4754 non-null int64
    avg_consume_less_12_valid_month               4754 non-null float64
    abs                                           4754 non-null int64
    top_trans_count_last_1_month                  4754 non-null float64
    avg_price_last_12_month                       4754 non-null int64
    avg_price_top_last_12_valid_month             4754 non-null float64
    trans_top_time_last_1_month                   4754 non-null float64
    trans_top_time_last_6_month                   4754 non-null float64
    consume_top_time_last_1_month                 4754 non-null float64
    consume_top_time_last_6_month                 4754 non-null float64
    cross_consume_count_last_1_month              4754 non-null float64
    trans_fail_top_count_enum_last_1_month        4754 non-null float64
    trans_fail_top_count_enum_last_6_month        4754 non-null float64
    trans_fail_top_count_enum_last_12_month       4754 non-null float64
    consume_mini_time_last_1_month                4754 non-null float64
    max_cumulative_consume_later_1_month          4754 non-null int64
    max_consume_count_later_6_month               4754 non-null float64
    railway_consume_count_last_12_month           4754 non-null float64
    pawns_auctions_trusts_consume_last_1_month    4754 non-null int64
    pawns_auctions_trusts_consume_last_6_month    4754 non-null int64
    jewelry_consume_count_last_6_month            4754 non-null float64
    status                                        4754 non-null int64
    first_transaction_day                         4754 non-null float64
    trans_day_last_12_month                       4754 non-null float64
    apply_score                                   4754 non-null float64
    apply_credibility                             4754 non-null float64
    query_org_count                               4754 non-null float64
    query_finance_count                           4754 non-null float64
    query_cash_count                              4754 non-null float64
    query_sum_count                               4754 non-null float64
    latest_query_time                             4754 non-null float64
    latest_one_month_apply                        4754 non-null float64
    latest_three_month_apply                      4754 non-null float64
    latest_six_month_apply                        4754 non-null float64
    loans_score                                   4754 non-null float64
    loans_credibility_behavior                    4754 non-null float64
    loans_count                                   4754 non-null float64
    loans_settle_count                            4754 non-null float64
    loans_overdue_count                           4754 non-null float64
    loans_org_count_behavior                      4754 non-null float64
    consfin_org_count_behavior                    4754 non-null float64
    loans_cash_count                              4754 non-null float64
    latest_one_month_loan                         4754 non-null float64
    latest_three_month_loan                       4754 non-null float64
    latest_six_month_loan                         4754 non-null float64
    history_suc_fee                               4754 non-null float64
    history_fail_fee                              4754 non-null float64
    latest_one_month_suc                          4754 non-null float64
    latest_one_month_fail                         4754 non-null float64
    loans_long_time                               4754 non-null float64
    loans_latest_time                             4754 non-null float64
    loans_credit_limit                            4754 non-null float64
    loans_credibility_limit                       4754 non-null float64
    loans_org_count_current                       4754 non-null float64
    loans_product_count                           4754 non-null float64
    loans_max_limit                               4754 non-null float64
    loans_avg_limit                               4754 non-null float64
    consfin_credit_limit                          4754 non-null float64
    consfin_credibility                           4754 non-null float64
    consfin_org_count_current                     4754 non-null float64
    consfin_product_count                         4754 non-null float64
    consfin_max_limit                             4754 non-null float64
    consfin_avg_limit                             4754 non-null float64
    latest_query_day                              4754 non-null float64
    loans_latest_day                              4754 non-null float64
    reg_preference_for_trad_一线城市                  4754 non-null float64
    reg_preference_for_trad_三线城市                  4754 non-null float64
    reg_preference_for_trad_二线城市                  4754 non-null float64
    reg_preference_for_trad_其他城市                  4754 non-null float64
    reg_preference_for_trad_境外                    4754 non-null float64
    dtypes: float64(77), int64(11)
    memory usage: 3.2 MB
    

# 划分训练集+测试集


```python
from sklearn.model_selection import train_test_split
df_train,df_test = train_test_split(df,test_size = 0.3,random_state = 2018)
df_train.to_csv('df_train.csv')
df_test.to_csv('df_test.csv')
```


```python

```


```python

```
