#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


prices = pd.read_excel('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/commodity_prices.xlsx')
prices.set_index('Date', inplace=True)
prices = prices.bfill()


# In[3]:


def convert_prices_to_returns(prices):
    returns = prices.pct_change(1)
    returns.iloc[0,:] = 0
    return returns

def daily_monthly_rollup(daily_returns):
    monthly_returns = (1 + daily_returns).resample('M').prod() - 1
    return monthly_returns

def macd(prices,shorter,longer):
    short_ma = prices.ewm(span=shorter, min_periods=shorter).mean()
    long_ma = prices.ewm(span=longer, min_periods=longer).mean()
    ewmstd_60 = prices.ewm(span=60).std()
    macd = short_ma - long_ma
    macd_flag = (long_ma > short_ma).astype(int)
    q = macd / ewmstd_60
    z = q / q.ewm(span=300, min_periods=300).std()
    return z, macd_flag

def week52_high_low_ratio(prices):
    high52_ratio = prices/prices.rolling(window=52*7).max()
    low52_ratio = prices/prices.rolling(window=52*7).min()
    return high52_ratio, low52_ratio


# In[4]:


daily_returns = convert_prices_to_returns(prices)
monthly_returns = daily_monthly_rollup(daily_returns)

std_daily_60 = daily_returns.rolling(window=60).std()
std_monthly_6 = (daily_returns.rolling(window=150).std())*(6**0.5)

daily_returns_20_cum = ((1+daily_returns).rolling(window=20).apply(lambda x: x.prod(), raw=True) - 1)
daily_returns_20_cum_mean = daily_returns_20_cum.mean(axis=1)
z_daily_returns_20_cum = daily_returns_20_cum.div(daily_returns_20_cum_mean, axis='index')

monthly_returns_12_cum = ((1+monthly_returns).rolling(window=12).apply(lambda x: x.prod(), raw=True) - 1)
monthly_returns_12_cum_mean = monthly_returns_12_cum.mean(axis=1)
z_monthly_returns_12_cum = monthly_returns_12_cum.div(monthly_returns_12_cum_mean, axis='index')

macd_8_24, macdf_8_24 = macd(prices,8,24)
macd_16_48, macdf_16_48 = macd(prices,16,48)
macd_32_96, macdf_32_96 = macd(prices,32,96)

week52_high,_ = week52_high_low_ratio(prices)


# In[5]:


df = pd.DataFrame(columns = ['Asset','ret_d_t0','ret_d_t1','ret_d_t2','ret_d_t3','ret_d_t4','ret_d_t5','ret_d_t6',
                             'ret_d_t7','ret_d_t8','ret_d_t9','ret_d_t10','ret_d_t11','ret_d_t12','ret_d_t13',
                             'ret_d_t14','ret_d_t15','ret_d_t16','ret_d_t17','ret_d_t18','ret_d_t19','ret_d_t20',
                             'ret_m_t0','ret_m_t1','ret_m_t2','ret_m_t3','ret_m_t4','ret_m_t5','ret_m_t6','ret_m_t7',
                             'ret_m_t8','ret_m_t9','ret_m_t10','ret_m_t11','ret_m_t12', 
                             'retn_d_t0','retn_d_t1','retn_d_t2','retn_d_t3','retn_d_t4',
                             'retn_d_t5','retn_d_t6','retn_d_t7','retn_d_t8','retn_d_t9','retn_d_t10',
                             'retn_d_t11','retn_d_t12','retn_d_t13','retn_d_t14','retn_d_t15','retn_d_t16',
                             'retn_d_t17','retn_d_t18','retn_d_t19','retn_d_t20','retn_m_t0','retn_m_t1','retn_m_t2',
                             'retn_m_t3','retn_m_t4','retn_m_t5','retn_m_t6','retcc_d_t0','retcc_d_t1','retcc_d_t2',
                             'retcc_d_t3','retcc_d_t4','retcc_d_t5','retcc_d_t6','retcc_d_t7','retcc_d_t8','retcc_d_t9',
                             'retcc_d_t10','retcc_d_t11','retcc_d_t12','retcc_d_t13','retcc_d_t14','retcc_d_t15',
                             'retcc_d_t16','retcc_d_t17','retcc_d_t18','retcc_d_t19','retcc_d_t20',
                             'retcc_m_t0','retcc_m_t1','retcc_m_t2','retcc_m_t3','retcc_m_t4','retcc_m_t5','retcc_m_t6',
                             'retcc_m_t7','retcc_m_t8','retcc_m_t9','retcc_m_t10','retcc_m_t11','retcc_m_t12',
                             'macd_8_24','macd_16_48','macd_32_96','macdf_8_24','macdf_16_48','macdf_32_96','week52_high',
                             'return_next_period'])


# In[6]:


# row_num = 0
# for i in range(12,monthly_returns.shape[0]-1):
#     daily_index = daily_returns.index.get_loc(monthly_returns.index[i])
#     for j in monthly_returns.columns:
#         df['Asset'].loc[row_num] = j
        
#         df['ret_d_t0'].loc[row_num] = daily_returns[j].iloc[daily_index-0]
#         df['ret_d_t1'].loc[row_num] = daily_returns[j].iloc[daily_index-1]
#         df['ret_d_t2'].loc[row_num] = daily_returns[j].iloc[daily_index-2]
#         df['ret_d_t3'].loc[row_num] = daily_returns[j].iloc[daily_index-3]
#         df['ret_d_t4'].loc[row_num] = daily_returns[j].iloc[daily_index-4]
#         df['ret_d_t5'].loc[row_num] = daily_returns[j].iloc[daily_index-5]
#         df['ret_d_t6'].loc[row_num] = daily_returns[j].iloc[daily_index-6]
#         df['ret_d_t7'].loc[row_num] = daily_returns[j].iloc[daily_index-7]
#         df['ret_d_t8'].loc[row_num] = daily_returns[j].iloc[daily_index-8]
#         df['ret_d_t9'].loc[row_num] = daily_returns[j].iloc[daily_index-9]
#         df['ret_d_t10'].loc[row_num] = daily_returns[j].iloc[daily_index-10]
#         df['ret_d_t11'].loc[row_num] = daily_returns[j].iloc[daily_index-11]
#         df['ret_d_t12'].loc[row_num] = daily_returns[j].iloc[daily_index-12]
#         df['ret_d_t13'].loc[row_num] = daily_returns[j].iloc[daily_index-13]
#         df['ret_d_t14'].loc[row_num] = daily_returns[j].iloc[daily_index-14]
#         df['ret_d_t15'].loc[row_num] = daily_returns[j].iloc[daily_index-15]
#         df['ret_d_t16'].loc[row_num] = daily_returns[j].iloc[daily_index-16]
#         df['ret_d_t17'].loc[row_num] = daily_returns[j].iloc[daily_index-17]
#         df['ret_d_t18'].loc[row_num] = daily_returns[j].iloc[daily_index-18]
#         df['ret_d_t19'].loc[row_num] = daily_returns[j].iloc[daily_index-19]
#         df['ret_d_t20'].loc[row_num] = daily_returns[j].iloc[daily_index-20]
        
#         df['ret_m_t0'].loc[row_num] = monthly_returns[j].iloc[i-0]
#         df['ret_m_t1'].loc[row_num] = monthly_returns[j].iloc[i-1]
#         df['ret_m_t2'].loc[row_num] = monthly_returns[j].iloc[i-2]
#         df['ret_m_t3'].loc[row_num] = monthly_returns[j].iloc[i-3]
#         df['ret_m_t4'].loc[row_num] = monthly_returns[j].iloc[i-4]
#         df['ret_m_t5'].loc[row_num] = monthly_returns[j].iloc[i-5]
#         df['ret_m_t6'].loc[row_num] = monthly_returns[j].iloc[i-6]
#         df['ret_m_t7'].loc[row_num] = monthly_returns[j].iloc[i-7]
#         df['ret_m_t8'].loc[row_num] = monthly_returns[j].iloc[i-8]
#         df['ret_m_t9'].loc[row_num] = monthly_returns[j].iloc[i-9]
#         df['ret_m_t10'].loc[row_num] = monthly_returns[j].iloc[i-10]
#         df['ret_m_t11'].loc[row_num] = monthly_returns[j].iloc[i-11]
#         df['ret_m_t12'].loc[row_num] = monthly_returns[j].iloc[i-12]
        
#         df['retn_d_t0'].loc[row_num] = daily_returns[j].iloc[daily_index-0]/ std_daily_60[j].iloc[daily_index-0]
#         df['retn_d_t1'].loc[row_num] = daily_returns[j].iloc[daily_index-1]/ std_daily_60[j].iloc[daily_index-1]
#         df['retn_d_t2'].loc[row_num] = daily_returns[j].iloc[daily_index-2]/ std_daily_60[j].iloc[daily_index-2]
#         df['retn_d_t3'].loc[row_num] = daily_returns[j].iloc[daily_index-3]/ std_daily_60[j].iloc[daily_index-3]
#         df['retn_d_t4'].loc[row_num] = daily_returns[j].iloc[daily_index-4]/ std_daily_60[j].iloc[daily_index-4]
#         df['retn_d_t5'].loc[row_num] = daily_returns[j].iloc[daily_index-5]/ std_daily_60[j].iloc[daily_index-5]
#         df['retn_d_t6'].loc[row_num] = daily_returns[j].iloc[daily_index-6]/ std_daily_60[j].iloc[daily_index-6]
#         df['retn_d_t7'].loc[row_num] = daily_returns[j].iloc[daily_index-7]/ std_daily_60[j].iloc[daily_index-7]
#         df['retn_d_t8'].loc[row_num] = daily_returns[j].iloc[daily_index-8]/ std_daily_60[j].iloc[daily_index-8]
#         df['retn_d_t9'].loc[row_num] = daily_returns[j].iloc[daily_index-9]/ std_daily_60[j].iloc[daily_index-9]
#         df['retn_d_t10'].loc[row_num] = daily_returns[j].iloc[daily_index-10]/ std_daily_60[j].iloc[daily_index-10]
#         df['retn_d_t11'].loc[row_num] = daily_returns[j].iloc[daily_index-11]/ std_daily_60[j].iloc[daily_index-11]
#         df['retn_d_t12'].loc[row_num] = daily_returns[j].iloc[daily_index-12]/ std_daily_60[j].iloc[daily_index-12]
#         df['retn_d_t13'].loc[row_num] = daily_returns[j].iloc[daily_index-13]/ std_daily_60[j].iloc[daily_index-13]
#         df['retn_d_t14'].loc[row_num] = daily_returns[j].iloc[daily_index-14]/ std_daily_60[j].iloc[daily_index-14]
#         df['retn_d_t15'].loc[row_num] = daily_returns[j].iloc[daily_index-15]/ std_daily_60[j].iloc[daily_index-15]
#         df['retn_d_t16'].loc[row_num] = daily_returns[j].iloc[daily_index-16]/ std_daily_60[j].iloc[daily_index-16]
#         df['retn_d_t17'].loc[row_num] = daily_returns[j].iloc[daily_index-17]/ std_daily_60[j].iloc[daily_index-17]
#         df['retn_d_t18'].loc[row_num] = daily_returns[j].iloc[daily_index-18]/ std_daily_60[j].iloc[daily_index-18]
#         df['retn_d_t19'].loc[row_num] = daily_returns[j].iloc[daily_index-19]/ std_daily_60[j].iloc[daily_index-19]
#         df['retn_d_t20'].loc[row_num] = daily_returns[j].iloc[daily_index-20]/ std_daily_60[j].iloc[daily_index-20]

        
#         df['retn_m_t0'].loc[row_num] = monthly_returns[j].iloc[i-0]/ std_monthly_6[j].iloc[daily_index-0]
#         df['retn_m_t1'].loc[row_num] = monthly_returns[j].iloc[i-1]/ std_monthly_6[j].iloc[daily_index-1]
#         df['retn_m_t2'].loc[row_num] = monthly_returns[j].iloc[i-2]/ std_monthly_6[j].iloc[daily_index-2]
#         df['retn_m_t3'].loc[row_num] = monthly_returns[j].iloc[i-3]/ std_monthly_6[j].iloc[daily_index-3]
#         df['retn_m_t4'].loc[row_num] = monthly_returns[j].iloc[i-4]/ std_monthly_6[j].iloc[daily_index-4]
#         df['retn_m_t5'].loc[row_num] = monthly_returns[j].iloc[i-5]/ std_monthly_6[j].iloc[daily_index-5]
#         df['retn_m_t6'].loc[row_num] = monthly_returns[j].iloc[i-6]/ std_monthly_6[j].iloc[daily_index-6]
        
#         df['retcc_d_t0'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-0]
#         df['retcc_d_t1'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-1]
#         df['retcc_d_t2'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-2]
#         df['retcc_d_t3'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-3]
#         df['retcc_d_t4'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-4]
#         df['retcc_d_t5'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-5]
#         df['retcc_d_t6'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-6]
#         df['retcc_d_t7'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-7]
#         df['retcc_d_t8'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-8]
#         df['retcc_d_t9'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-9]
#         df['retcc_d_t10'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-10]
#         df['retcc_d_t11'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-11]
#         df['retcc_d_t12'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-12]
#         df['retcc_d_t13'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-13]
#         df['retcc_d_t14'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-14]
#         df['retcc_d_t15'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-15]
#         df['retcc_d_t16'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-16]
#         df['retcc_d_t17'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-17]
#         df['retcc_d_t18'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-18]
#         df['retcc_d_t19'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-19]
#         df['retcc_d_t20'].loc[row_num] = z_daily_returns_20_cum[j].iloc[daily_index-20]
        
#         df['retcc_m_t0'].loc[row_num] = z_monthly_returns_12_cum[j].iloc[i-0]
#         df['retcc_m_t1'].loc[row_num] = z_monthly_returns_12_cum[j].iloc[i-1]
#         df['retcc_m_t2'].loc[row_num] = z_monthly_returns_12_cum[j].iloc[i-2]
#         df['retcc_m_t3'].loc[row_num] = z_monthly_returns_12_cum[j].iloc[i-3]
#         df['retcc_m_t4'].loc[row_num] = z_monthly_returns_12_cum[j].iloc[i-4]
#         df['retcc_m_t5'].loc[row_num] = z_monthly_returns_12_cum[j].iloc[i-5]
#         df['retcc_m_t6'].loc[row_num] = z_monthly_returns_12_cum[j].iloc[i-6]
#         df['retcc_m_t7'].loc[row_num] = z_monthly_returns_12_cum[j].iloc[i-7]
#         df['retcc_m_t8'].loc[row_num] = z_monthly_returns_12_cum[j].iloc[i-8]
#         df['retcc_m_t9'].loc[row_num] = z_monthly_returns_12_cum[j].iloc[i-9]
#         df['retcc_m_t10'].loc[row_num] = z_monthly_returns_12_cum[j].iloc[i-10]
#         df['retcc_m_t11'].loc[row_num] = z_monthly_returns_12_cum[j].iloc[i-11]
#         df['retcc_m_t12'].loc[row_num] = z_monthly_returns_12_cum[j].iloc[i-12]
        
#         df['macd_8_24'].loc[row_num] = macd_8_24[j].iloc[i]
#         df['macd_8_24'].loc[row_num] = macd_8_24[j].iloc[i]
#         df['macd_8_24'].loc[row_num] = macd_8_24[j].iloc[i]
#         df['macdf_8_24'].loc[row_num] = macdf_8_24[j].iloc[i]
#         df['macdf_8_24'].loc[row_num] = macdf_8_24[j].iloc[i]
#         df['macdf_8_24'].loc[row_num] = macdf_8_24[j].iloc[i]
        
#         df['return_next_period'].loc[row_num] = monthly_returns[j].iloc[i+1]
        
#         row_num+=1


# In[7]:


row_num = 0
for i in range(24,monthly_returns.shape[0]-1):
    daily_index = daily_returns.index.get_loc(daily_returns[(daily_returns.index.month==monthly_returns.index[i].month) & (daily_returns.index.year==monthly_returns.index[i].year)].index[-1])
    for j in monthly_returns.columns:
        Asset =  j

        ret_d_t0 =  daily_returns[j].iloc[daily_index-0]
        ret_d_t1 =  daily_returns[j].iloc[daily_index-1]
        ret_d_t2 =  daily_returns[j].iloc[daily_index-2]
        ret_d_t3 =  daily_returns[j].iloc[daily_index-3]
        ret_d_t4 =  daily_returns[j].iloc[daily_index-4]
        ret_d_t5 =  daily_returns[j].iloc[daily_index-5]
        ret_d_t6 =  daily_returns[j].iloc[daily_index-6]
        ret_d_t7 =  daily_returns[j].iloc[daily_index-7]
        ret_d_t8 =  daily_returns[j].iloc[daily_index-8]
        ret_d_t9 =  daily_returns[j].iloc[daily_index-9]
        ret_d_t10 =  daily_returns[j].iloc[daily_index-10]
        ret_d_t11 =  daily_returns[j].iloc[daily_index-11]
        ret_d_t12 =  daily_returns[j].iloc[daily_index-12]
        ret_d_t13 =  daily_returns[j].iloc[daily_index-13]
        ret_d_t14 =  daily_returns[j].iloc[daily_index-14]
        ret_d_t15 =  daily_returns[j].iloc[daily_index-15]
        ret_d_t16 =  daily_returns[j].iloc[daily_index-16]
        ret_d_t17 =  daily_returns[j].iloc[daily_index-17]
        ret_d_t18 =  daily_returns[j].iloc[daily_index-18]
        ret_d_t19 =  daily_returns[j].iloc[daily_index-19]
        ret_d_t20 =  daily_returns[j].iloc[daily_index-20]

        ret_m_t0 =  monthly_returns[j].iloc[i-0]
        ret_m_t1 =  monthly_returns[j].iloc[i-1]
        ret_m_t2 =  monthly_returns[j].iloc[i-2]
        ret_m_t3 =  monthly_returns[j].iloc[i-3]
        ret_m_t4 =  monthly_returns[j].iloc[i-4]
        ret_m_t5 =  monthly_returns[j].iloc[i-5]
        ret_m_t6 =  monthly_returns[j].iloc[i-6]
        ret_m_t7 =  monthly_returns[j].iloc[i-7]
        ret_m_t8 =  monthly_returns[j].iloc[i-8]
        ret_m_t9 =  monthly_returns[j].iloc[i-9]
        ret_m_t10 =  monthly_returns[j].iloc[i-10]
        ret_m_t11 =  monthly_returns[j].iloc[i-11]
        ret_m_t12 =  monthly_returns[j].iloc[i-12]

        retn_d_t0 =  daily_returns[j].iloc[daily_index-0]/ std_daily_60[j].iloc[daily_index-0]
        retn_d_t1 =  daily_returns[j].iloc[daily_index-1]/ std_daily_60[j].iloc[daily_index-1]
        retn_d_t2 =  daily_returns[j].iloc[daily_index-2]/ std_daily_60[j].iloc[daily_index-2]
        retn_d_t3 =  daily_returns[j].iloc[daily_index-3]/ std_daily_60[j].iloc[daily_index-3]
        retn_d_t4 =  daily_returns[j].iloc[daily_index-4]/ std_daily_60[j].iloc[daily_index-4]
        retn_d_t5 =  daily_returns[j].iloc[daily_index-5]/ std_daily_60[j].iloc[daily_index-5]
        retn_d_t6 =  daily_returns[j].iloc[daily_index-6]/ std_daily_60[j].iloc[daily_index-6]
        retn_d_t7 =  daily_returns[j].iloc[daily_index-7]/ std_daily_60[j].iloc[daily_index-7]
        retn_d_t8 =  daily_returns[j].iloc[daily_index-8]/ std_daily_60[j].iloc[daily_index-8]
        retn_d_t9 =  daily_returns[j].iloc[daily_index-9]/ std_daily_60[j].iloc[daily_index-9]
        retn_d_t10 =  daily_returns[j].iloc[daily_index-10]/ std_daily_60[j].iloc[daily_index-10]
        retn_d_t11 =  daily_returns[j].iloc[daily_index-11]/ std_daily_60[j].iloc[daily_index-11]
        retn_d_t12 =  daily_returns[j].iloc[daily_index-12]/ std_daily_60[j].iloc[daily_index-12]
        retn_d_t13 =  daily_returns[j].iloc[daily_index-13]/ std_daily_60[j].iloc[daily_index-13]
        retn_d_t14 =  daily_returns[j].iloc[daily_index-14]/ std_daily_60[j].iloc[daily_index-14]
        retn_d_t15 =  daily_returns[j].iloc[daily_index-15]/ std_daily_60[j].iloc[daily_index-15]
        retn_d_t16 =  daily_returns[j].iloc[daily_index-16]/ std_daily_60[j].iloc[daily_index-16]
        retn_d_t17 =  daily_returns[j].iloc[daily_index-17]/ std_daily_60[j].iloc[daily_index-17]
        retn_d_t18 =  daily_returns[j].iloc[daily_index-18]/ std_daily_60[j].iloc[daily_index-18]
        retn_d_t19 =  daily_returns[j].iloc[daily_index-19]/ std_daily_60[j].iloc[daily_index-19]
        retn_d_t20 =  daily_returns[j].iloc[daily_index-20]/ std_daily_60[j].iloc[daily_index-20]


        retn_m_t0 =  monthly_returns[j].iloc[i-0]/ std_monthly_6[j].iloc[daily_index-0]
        retn_m_t1 =  monthly_returns[j].iloc[i-1]/ std_monthly_6[j].iloc[daily_index-1]
        retn_m_t2 =  monthly_returns[j].iloc[i-2]/ std_monthly_6[j].iloc[daily_index-2]
        retn_m_t3 =  monthly_returns[j].iloc[i-3]/ std_monthly_6[j].iloc[daily_index-3]
        retn_m_t4 =  monthly_returns[j].iloc[i-4]/ std_monthly_6[j].iloc[daily_index-4]
        retn_m_t5 =  monthly_returns[j].iloc[i-5]/ std_monthly_6[j].iloc[daily_index-5]
        retn_m_t6 =  monthly_returns[j].iloc[i-6]/ std_monthly_6[j].iloc[daily_index-6]

        retcc_d_t0 =  z_daily_returns_20_cum[j].iloc[daily_index-0]
        retcc_d_t1 =  z_daily_returns_20_cum[j].iloc[daily_index-1]
        retcc_d_t2 =  z_daily_returns_20_cum[j].iloc[daily_index-2]
        retcc_d_t3 =  z_daily_returns_20_cum[j].iloc[daily_index-3]
        retcc_d_t4 =  z_daily_returns_20_cum[j].iloc[daily_index-4]
        retcc_d_t5 =  z_daily_returns_20_cum[j].iloc[daily_index-5]
        retcc_d_t6 =  z_daily_returns_20_cum[j].iloc[daily_index-6]
        retcc_d_t7 =  z_daily_returns_20_cum[j].iloc[daily_index-7]
        retcc_d_t8 =  z_daily_returns_20_cum[j].iloc[daily_index-8]
        retcc_d_t9 =  z_daily_returns_20_cum[j].iloc[daily_index-9]
        retcc_d_t10 =  z_daily_returns_20_cum[j].iloc[daily_index-10]
        retcc_d_t11 =  z_daily_returns_20_cum[j].iloc[daily_index-11]
        retcc_d_t12 =  z_daily_returns_20_cum[j].iloc[daily_index-12]
        retcc_d_t13 =  z_daily_returns_20_cum[j].iloc[daily_index-13]
        retcc_d_t14 =  z_daily_returns_20_cum[j].iloc[daily_index-14]
        retcc_d_t15 =  z_daily_returns_20_cum[j].iloc[daily_index-15]
        retcc_d_t16 =  z_daily_returns_20_cum[j].iloc[daily_index-16]
        retcc_d_t17 =  z_daily_returns_20_cum[j].iloc[daily_index-17]
        retcc_d_t18 =  z_daily_returns_20_cum[j].iloc[daily_index-18]
        retcc_d_t19 =  z_daily_returns_20_cum[j].iloc[daily_index-19]
        retcc_d_t20 =  z_daily_returns_20_cum[j].iloc[daily_index-20]

        retcc_m_t0 =  z_monthly_returns_12_cum[j].iloc[i-0]
        retcc_m_t1 =  z_monthly_returns_12_cum[j].iloc[i-1]
        retcc_m_t2 =  z_monthly_returns_12_cum[j].iloc[i-2]
        retcc_m_t3 =  z_monthly_returns_12_cum[j].iloc[i-3]
        retcc_m_t4 =  z_monthly_returns_12_cum[j].iloc[i-4]
        retcc_m_t5 =  z_monthly_returns_12_cum[j].iloc[i-5]
        retcc_m_t6 =  z_monthly_returns_12_cum[j].iloc[i-6]
        retcc_m_t7 =  z_monthly_returns_12_cum[j].iloc[i-7]
        retcc_m_t8 =  z_monthly_returns_12_cum[j].iloc[i-8]
        retcc_m_t9 =  z_monthly_returns_12_cum[j].iloc[i-9]
        retcc_m_t10 =  z_monthly_returns_12_cum[j].iloc[i-10]
        retcc_m_t11 =  z_monthly_returns_12_cum[j].iloc[i-11]
        retcc_m_t12 =  z_monthly_returns_12_cum[j].iloc[i-12]

#         print(macd_8_24[j].iloc[daily_index])
        macd_8_24t =  macd_8_24[j].iloc[daily_index]
        macd_16_48t =  macd_16_48[j].iloc[daily_index]
        macd_32_96t =  macd_32_96[j].iloc[daily_index]
        macdf_8_24t =  macdf_8_24[j].iloc[daily_index]
        macdf_16_48t =  macdf_16_48[j].iloc[daily_index]
        macdf_32_96t =  macdf_32_96[j].iloc[daily_index]
        week52_h = week52_high[j].iloc[daily_index]

        return_next_period =  monthly_returns[j].iloc[i+1]
        
        new_row = [{'Asset':Asset,'ret_d_t0':ret_d_t0,'ret_d_t1':ret_d_t1,'ret_d_t2':ret_d_t2,
        'ret_d_t3':ret_d_t3,
        'ret_d_t4':ret_d_t4,
        'ret_d_t5':ret_d_t5,
        'ret_d_t6':ret_d_t6,
        'ret_d_t7':ret_d_t7,
        'ret_d_t8':ret_d_t8,
        'ret_d_t9':ret_d_t9,
        'ret_d_t10':ret_d_t10,
        'ret_d_t11':ret_d_t11,
        'ret_d_t12':ret_d_t12,
        'ret_d_t13':ret_d_t13,
        'ret_d_t14':ret_d_t14,
        'ret_d_t15':ret_d_t15,
        'ret_d_t16':ret_d_t16,
        'ret_d_t17':ret_d_t17,
        'ret_d_t18':ret_d_t18,
        'ret_d_t19':ret_d_t19,
        'ret_d_t20':ret_d_t20,

        'ret_m_t0':ret_m_t0,
        'ret_m_t1':ret_m_t1,
        'ret_m_t2':ret_m_t2,
        'ret_m_t3':ret_m_t3,
        'ret_m_t4':ret_m_t4,
        'ret_m_t5':ret_m_t5,
        'ret_m_t6':ret_m_t6,
        'ret_m_t7':ret_m_t7,
        'ret_m_t8':ret_m_t8,
        'ret_m_t9':ret_m_t9,
        'ret_m_t10':ret_m_t10,
        'ret_m_t11':ret_m_t11,
        'ret_m_t12':ret_m_t12,

        'retn_d_t0':retn_d_t0,
        'retn_d_t1':retn_d_t1,
        'retn_d_t2':retn_d_t2,
        'retn_d_t3':retn_d_t3,
        'retn_d_t4':retn_d_t4,
        'retn_d_t5':retn_d_t5,
        'retn_d_t6':retn_d_t6,
        'retn_d_t7':retn_d_t7,
        'retn_d_t8':retn_d_t8,
        'retn_d_t9':retn_d_t9,
        'retn_d_t10':retn_d_t10,
        'retn_d_t11':retn_d_t11,
        'retn_d_t12':retn_d_t12,
        'retn_d_t13':retn_d_t13,
        'retn_d_t14':retn_d_t14,
        'retn_d_t15':retn_d_t15,
        'retn_d_t16':retn_d_t16,
        'retn_d_t17':retn_d_t17,
        'retn_d_t18':retn_d_t18,
        'retn_d_t19':retn_d_t19,
        'retn_d_t20':retn_d_t20,


        'retn_m_t0':retn_m_t0,
        'retn_m_t1':retn_m_t1,
        'retn_m_t2':retn_m_t2,
        'retn_m_t3':retn_m_t3,
        'retn_m_t4':retn_m_t4,
        'retn_m_t5':retn_m_t5,
        'retn_m_t6':retn_m_t6,

        'retcc_d_t0':retcc_d_t0,
        'retcc_d_t1':retcc_d_t1,
        'retcc_d_t2':retcc_d_t2,
        'retcc_d_t3':retcc_d_t3,
        'retcc_d_t4':retcc_d_t4,
        'retcc_d_t5':retcc_d_t5,
        'retcc_d_t6':retcc_d_t6,
        'retcc_d_t7':retcc_d_t7,
        'retcc_d_t8':retcc_d_t8,
        'retcc_d_t9':retcc_d_t9,
        'retcc_d_t10':retcc_d_t10,
        'retcc_d_t11':retcc_d_t11,
        'retcc_d_t12':retcc_d_t12,
        'retcc_d_t13':retcc_d_t13,
        'retcc_d_t14':retcc_d_t14,
        'retcc_d_t15':retcc_d_t15,
        'retcc_d_t16':retcc_d_t16,
        'retcc_d_t17':retcc_d_t17,
        'retcc_d_t18':retcc_d_t18,
        'retcc_d_t19':retcc_d_t19,
        'retcc_d_t20':retcc_d_t20,

        'retcc_m_t0':retcc_m_t0,
        'retcc_m_t1':retcc_m_t1,
        'retcc_m_t2':retcc_m_t2,
        'retcc_m_t3':retcc_m_t3,
        'retcc_m_t4':retcc_m_t4,
        'retcc_m_t5':retcc_m_t5,
        'retcc_m_t6':retcc_m_t6,
        'retcc_m_t7':retcc_m_t7,
        'retcc_m_t8':retcc_m_t8,
        'retcc_m_t9':retcc_m_t9,
        'retcc_m_t10':retcc_m_t10,
        'retcc_m_t11':retcc_m_t11,
        'retcc_m_t12':retcc_m_t12,

        'macd_8_24':macd_8_24t,
        'macd_16_48':macd_16_48t,
        'macd_32_96':macd_32_96t,
        'macdf_8_24':macdf_8_24t,
        'macdf_16_48':macdf_16_48t,
        'macdf_32_96':macdf_32_96t,
        'week52_high':week52_h,

        'return_next_period':return_next_period}]
        
        df = df.append(new_row, ignore_index=True)
        
        row_num+=1


# In[8]:


df_index = [val for val in monthly_returns.index[24:(len(monthly_returns.index)-1)] for _ in range(27)]
df.index = df_index

df1 = df.copy()
df1 = df1.dropna()
df1.isna().sum()[25:50]
df1_index =df1.index 
df1 = df1.reset_index(drop = True)


# In[ ]:





# In[9]:


import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


# In[10]:


X = df1.drop(['Asset','return_next_period'], axis=1)
y = df1['return_next_period']
Asset_Name = df1['Asset']


# In[11]:


X_train, X_test = X.iloc[:4500], X[4500:]
y_train, y_test = y[:4500], y[4500:]

# np.random.seed(0)
# num_samples = 1000
# num_variables = 100

# X = np.random.rand(num_samples, num_variables)
# y = np.sum(X, axis=1) + np.random.rand(num_samples)

# # Split the data into training and testing sets
# X_train, X_test = X[:800], X[800:]
# y_train, y_test = y[:800], y[800:]


# In[12]:


def r2_metric(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())


# In[13]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[14]:


validation_split = 0.2
num_validation_samples = int(len(X_train_scaled) * validation_split)
indices = np.arange(len(X_train_scaled))
np.random.shuffle(indices)
X_validation = X_train_scaled[indices[:num_validation_samples]]
y_validation = y_train[indices[:num_validation_samples]]
X_train_scaled_v = X_train_scaled[indices[num_validation_samples:]]
y_train_v = y_train[indices[num_validation_samples:]]


# In[15]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(103,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])


# In[16]:


model.summary()


# In[179]:


model.compile(optimizer='adam', loss='mse', metrics=[r2_metric])

model.fit(X_train_scaled, y_train, epochs=100, batch_size=16,
          validation_data=(X_validation, y_validation), validation_batch_size=32,verbose=2)


# In[180]:


loss = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Test Loss: {loss}')

predictions = model.predict(X_test_scaled)


# In[181]:


X_scaled = scaler.transform(X)
loss = model.evaluate(X_scaled, y, verbose=0)
print(f'Test Loss: {loss}')


# In[182]:


y_pred = model.predict(X_scaled)


# In[199]:


df2 = pd.DataFrame({'Asset':np.array(Asset_Name), 'Actual_Returns':np.array(y), 'Predicted_Returns':y_pred.flatten()},
                  index = df1_index)


# In[200]:


ML_df = pd.DataFrame(columns = monthly_returns.columns)
row_num = 0
ML_df.loc[0] = -1
for i in range(df2.shape[0]):
# for i in range(2):
#     print(df2['Asset'].iloc[i])
    ML_df[df2['Asset'].iloc[i]].loc[row_num] = df2['Predicted_Returns'].iloc[i]
    if (i==(df2.shape[0]-1)):
        continue
    if (df2.index[i] != df2.index[i+1]):
        row_num += 1
        ML_df.loc[row_num] = -1
        
ML_df.index = df2.index.unique()


# In[218]:


x = 2
rank_df = ML_df.rank(axis=1, ascending = False)
top_x = rank_df.apply(lambda row: row.nsmallest(x).index.tolist(), axis=1)

cols = []
for i in range(x):
    cols.append(f'Long{i+1}')
return_l_s = pd.DataFrame(index=ML_df.index, columns=cols)
for i in range(return_l_s.shape[0]-1):
    for j in range(x):
            return_l_s.iloc[i,j] = monthly_returns.iloc[24+i][top_x[i][j]]
            
returns_alloc = return_l_s.mean(axis=1)


# In[219]:


import matplotlib.pyplot as plt
print('22 year CAGR : ',(((1+returns_alloc[:-1]).prod())**(1/22)-1),"%")
cumulative_returns = (1 + returns_alloc).cumprod()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cumulative_returns.index[:], cumulative_returns[:], label='Cumulative Returns')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Returns')
ax.set_title('Cumulative Returns Plot')


# In[230]:


print('3.75 year CAGR : ',(((1+returns_alloc[216:-1]).prod())**(1/3.75)-1),"%")
cumulative_returns = (1 + returns_alloc[216:]).cumprod()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cumulative_returns.index[:], cumulative_returns[:], label='Cumulative Returns')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Returns')
ax.set_title('Cumulative Returns Plot')


# In[244]:


x = 2
rank_df = ML_df.rank(axis=1, ascending = False)
top_x = rank_df.apply(lambda row: row.nsmallest(x).index.tolist(), axis=1)
bottom_x = rank_df.apply(lambda row: row.nlargest(x).index.tolist(), axis=1)

cols = []
for i in range(x):
    cols.append(f'Long{i+1}')
for i in range(x):
    cols.append(f'Short{i+1}')
return_l_s = pd.DataFrame(index=ML_df.index, columns=cols)
for i in range(return_l_s.shape[0]-1):
    for j in range(x):
            return_l_s.iloc[i,j] = monthly_returns.iloc[24+i][top_x[i][j]]
    for j in range(x):
            return_l_s.iloc[i,j+x] = monthly_returns.iloc[24+i][bottom_x[i][j]]
            
returns_alloc = return_l_s.mean(axis=1)
print('3.5 year CAGR : ',(((1+returns_alloc[216:-1]).prod())**(1/3.5)-1)*100,"%")


# In[247]:


ML = returns_alloc[216:-1]
ML.to_excel('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/ML_returns.xlsx')


# In[248]:


def calc_max_drawdown(return_series):
    comp_ret = pd.Series((return_series+1).cumprod())
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret/peak)-1
    return dd.min()

#Analytics
def strategy_analytics(returns_alloc, rf = 0.05):
    mean_return = returns_alloc['returns'].mean()*12
    st_dev = returns_alloc['returns'].std()*(12**0.5)
    sharpe_ratio = (mean_return - rf)/st_dev
    sortino_ratio = (mean_return - rf)/(returns_alloc['returns'].loc[returns_alloc['returns']<0].std()*(12**0.5))
    max_drawd = - calc_max_drawdown(np.array(returns_alloc['returns']))
    calmar_ratio = mean_return/ max_drawd
    success_rate = sum(returns_alloc['returns']>0)/len(returns_alloc['returns'])
    average_up = returns_alloc['returns'].loc[returns_alloc['returns']>0].mean()
    average_down = returns_alloc['returns'].loc[returns_alloc['returns']<0].mean()
    return mean_return,st_dev,sharpe_ratio,sortino_ratio,max_drawd,calmar_ratio,success_rate,average_up,average_down


# In[254]:


returns_alloc = pd.DataFrame(returns_alloc)
returns_alloc.columns = ['returns']
strategy_analytics(returns_alloc[216:-1], rf = 0.05)


# In[ ]:




