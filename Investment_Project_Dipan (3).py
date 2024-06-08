#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import linregress
import scipy.stats
from scipy.optimize import minimize


# In[2]:


# prices = pd.read_excel('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/Dummy_Data.xlsx')
prices = pd.read_excel('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/commodity_prices.xlsx')
prices.set_index('Date', inplace=True)
prices = prices.bfill()

# prices_renew = pd.read_excel('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/commodity_prices_renewables.xlsx')
# prices_renew.set_index('Date', inplace=True)
# prices_renew = prices_renew.bfill()


# In[3]:


# prices.isna().sum(axis=0)


# In[4]:


def convert_prices_to_returns(prices):
    returns = prices.pct_change(1)
    returns.iloc[0,:] = 0
    return returns


# In[ ]:





# In[5]:


def daily_weekly_rollup(daily_returns):
    weekly_returns = (1 + daily_returns).resample('W').prod() - 1
    return weekly_returns

def daily_monthly_rollup(daily_returns):
    monthly_returns = (1 + daily_returns).resample('M').prod() - 1
    return monthly_returns

def weekly_monthly_rollup(weekly_returns):
    monthly_returns = (1 + weekly_returns).resample('M').prod() - 1
    return monthly_returns


# In[6]:


#Trend Indentification Functions
def return_lookback(returns,lookback_period):
    lookback_returns = (1 + returns).rolling(window=lookback_period).apply(lambda x: x.prod()) - 1
    return lookback_returns


# In[7]:


def week52_high_low_ratio(prices):
    high52_ratio = prices/prices.rolling(window=52*7).max()
    low52_ratio = prices/prices.rolling(window=52*7).min()
    return high52_ratio, low52_ratio


# In[8]:


def RSI_slope(prices, lookback_period):
    price_change = prices.diff()
    gain = price_change.applymap(lambda x: x if x > 0 else 0)
    loss = price_change.applymap(lambda x: -x if x < 0 else 0)
    avg_gain = gain.rolling(window=lookback_period).mean()
    avg_loss = loss.rolling(window=lookback_period).mean()
    relative_strength = avg_gain/avg_loss
    RSI = 100 - (100 / (1 + relative_strength))
    RSI_Slope = RSI.diff()/RSI
    return RSI_Slope
    
def moving_average_slope(prices,lookback_period):
    MA = prices.rolling(window=lookback_period).mean()
    MA_Slope = MA.diff()/MA
    return MA_Slope
    
def SMA5_SMA20(prices):
    SMA5 = prices.rolling(window=5).mean()
    SMA20 = prices.rolling(window=20).mean()
    SMA5_20 = (SMA5-SMA20)/SMA20
    return SMA5_20

def SMA20_SMA50(prices):
    SMA20 = prices.rolling(window=20).mean()
    SMA50 = prices.rolling(window=50).mean()
    SMA20_50 = (SMA20-SMA50)/SMA50
    return SMA20_50

def Reg_p_value(prices,lookback_period):
    
    Lp = np.log(prices)
    sequence = list(range(1, lookback_period + 1))
    pval_df = pd.DataFrame(index=Lp.index, columns=Lp.columns)
    for i in range(Lp.shape[0]):
        if i<(lookback_period-1):
            pval_df.iloc[i] = 0
            continue
        else:
            temp_df = Lp.iloc[(i-lookback_period+1):(i+1)]    
            for cols in temp_dF
            _, _, _, p_value, _ = linregress(sequence, temp_df[cols])
                pval_df.iloc[i][cols] = p_value  
    return pval_df

def Periods_Up_Down(returns,lookback_period):
    Up_Down = pd.DataFrame(index=returns.index, columns=returns.columns)
    for i in range(returns.shape[0]):
#         print(i)
        if i<(lookback_period-1):
            Up_Down.iloc[i] = 0
            continue
        else:
            temp_df = returns.iloc[(i-lookback_period+1):(i+1)]    
            for cols in temp_df.columns:
                pos = 0
                neg = 0
                for j in range(temp_df.shape[0]):
                    if temp_df[cols].iloc[j]>0:
                        pos+=1
                    else:
                        neg+=1
#                 print(cols)
                Up_Down.iloc[i][cols] = pos - neg  
    return Up_Down


# In[9]:


#Selection Functions
def month_end_values(df):
    month_end_df = df.resample('M').last()
    return month_end_df

def rank_fn(indicator,order=True):
    asc_flag = True if order=='Ascending' else False
    rank_df = indicator.rank(axis=1, ascending = asc_flag)
    return rank_df

def select_top_x_bottom_x(rank,x):
    top_x = rank.apply(lambda row: row.nsmallest(x).index.tolist(), axis=1)
    bottom_x = rank.apply(lambda row: row.nlargest(x).index.tolist(), axis=1)   
    return top_x, bottom_x


# In[ ]:





# In[10]:


#Allocation Functions
def std_dev(returns, lookback):
    std = returns.rolling(window=lookback).std()
    return std

def corr_mat(returns):
    cor_mat = returns.corr()
    return cor_mat

def cov_mat(returns):
    covr_mat = returns.cov()
    return covr_mat

def lookback_returns_long_short(returns,top_x,bottom_x,x):
    cols = []
    for i in range(x):
        cols.append(f'Long{i+1}')
    for i in range(x):
        cols.append(f'Short{i+1}')
    lookback_return_l_s = pd.DataFrame(index=returns.index, columns=cols)
    for i in range(returns.shape[0]-1):
        for j in range(x):
                lookback_return_l_s.iloc[i,j] = returns.iloc[i][top_x[i][j]]
        for j in range(x):
                lookback_return_l_s.iloc[i,j+x] = returns.iloc[i][bottom_x[i][j]]     
#             return_l_s.iloc[i,:x] = [returns.iloc[i+1][top_x[i][0]],returns.iloc[i+1][top_x[i][1]],returns.iloc[i+1][top_x[i][2]]]
#             return_l_s.iloc[i,x:] = [returns.iloc[i+1][bottom_x[i][0]],returns.iloc[i+1][bottom_x[i][1]],returns.iloc[i+1][bottom_x[i][2]]]
    return lookback_return_l_s

def return_series_long_short(returns,top_x,bottom_x,x,next_per=1):
    cols = []
    for i in range(x):
        cols.append(f'Long{i+1}')
    for i in range(x):
        cols.append(f'Short{i+1}')
    return_l_s = pd.DataFrame(index=returns.index, columns=cols)
    for i in range(return_l_s.shape[0]-next_per):
        for j in range(x):
                return_l_s.iloc[i,j] = returns.iloc[i+next_per][top_x[i][j]]
        for j in range(x):
                return_l_s.iloc[i,j] = returns.iloc[i+next_per][bottom_x[i][j]]     
#             return_l_s.iloc[i,:x] = [returns.iloc[i+1][top_x[i][0]],returns.iloc[i+1][top_x[i][1]],returns.iloc[i+1][top_x[i][2]]]
#             return_l_s.iloc[i,x:] = [returns.iloc[i+1][bottom_x[i][0]],returns.iloc[i+1][bottom_x[i][1]],returns.iloc[i+1][bottom_x[i][2]]]
    return return_l_s

def equally_wtd(returns_l_s,long_only):
    if long_only==True:
        return_next_period = returns_l_s.iloc[:, :int((returns_l_s.shape[1]/2))].mean(axis=1)
    else:
        return_next_period = returns_l_s.iloc[:, :int(returns_l_s.shape[1]/2)].mean(axis=1) + returns_l_s.iloc[:,int((returns_l_s.shape[1]/2-1)):].mean(axis=1) 
    return_next_period = pd.DataFrame(return_next_period)  
    return_next_period.columns = ['returns']
    return return_next_period

# def beta_neutral()



# In[11]:


def risk_target_obj(w,cov_mat,target_risk):
    vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    risk_contributions = w * (np.dot(cov_mat, w) / vol)
    return np.sum((risk_contributions - target_risk) ** 2)

def risk_parity_obj(w,cov_mat):
    vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    return np.sum((w - (vol**2)/(np.dot(cov_mat, w)*len(w))) ** 2)

# def risk_parity(returns_l_s,long_only,daily_returns,top_x,bottom_x,lookback=60):
#     returns_next_period = pd.DataFrame(index=returns_l_s.index, columns=['returns'])
#     for i in range(returns_l_s.shape[0]-1):
#         index_loc_daily_end = daily_returns.index.get_loc(daily_returns[(daily_returns.index.month==monthly_returns.index[i].month) & (daily_returns.index.year==monthly_returns.index[i].year)].index[-1])
#         index_loc_daily_start = max(0,index_loc_daily_end-lookback)

#         if long_only == False:
#             returns_temp = daily_returns.iloc[index_loc_daily_start:index_loc_daily_end][top_x[i]+bottom_x[i]]
#             num_assets = len(top_x[i]+bottom_x[i])
#             bounds = tuple((0, 2) for _ in range(int(num_assets/2))) + tuple((-2, 0) for _ in range(int(num_assets/2)))
#         else:
#             returns_temp = daily_returns.iloc[index_loc_daily_start:index_loc_daily_end][top_x[i]]
#             num_assets = len(top_x[i])
#             bounds = tuple((0, 2) for _ in range(int(num_assets/2)))
        
#         covmat = cov_mat(returns_temp)
#         constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
#         initial_w = np.ones(num_assets) / num_assets
#         w = minimize(risk_parity_obj, initial_w, args=(covmat,), method='SLSQP', bounds=bounds, constraints=constraints).x
#         returns_next_period.iloc[i] = np.sum(w * returns_l_s.iloc[i])
    
#     return returns_next_period

def risk_parity(returns_l_s,long_only,daily_returns,top_x,bottom_x,lookback=60):
    returns_next_period = pd.DataFrame(index=returns_l_s.index, columns=['returns'])
    for i in range(returns_l_s.shape[0]-1):
        index_loc_daily_end = daily_returns.index.get_loc(daily_returns[(daily_returns.index.month==monthly_returns.index[i].month) & (daily_returns.index.year==monthly_returns.index[i].year)].index[-1])
        index_loc_daily_start = max(0,index_loc_daily_end-lookback)

        if long_only == False:
            returns_temp = daily_returns.iloc[index_loc_daily_start:index_loc_daily_end][top_x[i]+bottom_x[i]]
            num_assets = len(top_x[i]+bottom_x[i])
            bounds = tuple((0, 2) for _ in range(int(num_assets/2))) + tuple((-2, 0) for _ in range(int(num_assets/2)))
        else:
            returns_temp = daily_returns.iloc[index_loc_daily_start:index_loc_daily_end][top_x[i]]
            num_assets = len(top_x[i])
            bounds = tuple((0, 2) for _ in range(int(num_assets/2)))
        
        covmat = cov_mat(returns_temp)
        w = [1/covmat.iloc[i,i] for i in range(covmat.shape[0])]
        w = w/sum(w)
        if any(math.isnan(x) for x in w) == True:
            w = [1/num_assets for _ in w]
#         print(w)
            
        returns_next_period.iloc[i] = np.sum(w * returns_l_s.iloc[i,:num_assets])
    
    return returns_next_period


def min_vol(returns_l_s,long_only,daily_returns,top_x,bottom_x,lookback=60):
    
    returns_next_period = pd.DataFrame(index=returns_l_s.index, columns=['returns'])
    
    for i in range(returns_l_s.shape[0]-1):
        index_loc_daily_end = daily_returns.index.get_loc(daily_returns[(daily_returns.index.month==monthly_returns.index[i].month) & (daily_returns.index.year==monthly_returns.index[i].year)].index[-1])
        index_loc_daily_start = max(0,index_loc_daily_end-lookback)

        if long_only == False:
            returns_temp = daily_returns.iloc[index_loc_daily_start:index_loc_daily_end][top_x[i]+bottom_x[i]]
            num_assets = len(top_x[i]+bottom_x[i])
            bounds = tuple((0, 2) for _ in range(int(num_assets/2))) + tuple((-2, 0) for _ in range(int(num_assets/2)))
        else:
            returns_temp = daily_returns.iloc[index_loc_daily_start:index_loc_daily_end][top_x[i]]
            num_assets = len(top_x[i])
            bounds = tuple((0, 2) for _ in range(int(num_assets)))
        
        covmat = cov_mat(returns_temp)
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        initial_w = np.ones(num_assets) / num_assets
        w = minimize(risk_target_obj, initial_w, args=(covmat, 0,), method='SLSQP', bounds=bounds, constraints=constraints).x
        if any(math.isnan(x) for x in w) == True:
            w = [1/num_assets for _ in w]
        returns_next_period.iloc[i] = np.sum(w * returns_l_s.iloc[i,:num_assets])
    
    return returns_next_period

def risk_budget(returns_l_s,long_only,daily_returns,top_x,bottom_x,target_risk,lookback=60):
    returns_next_period = pd.DataFrame(index=returns_l_s.index, columns=['returns'])
    
    for i in range(returns_l_s.shape[0]-1):
        index_loc_daily_end = daily_returns.index.get_loc(daily_returns[(daily_returns.index.month==monthly_returns.index[i].month) & (daily_returns.index.year==monthly_returns.index[i].year)].index[-1])
        index_loc_daily_start = max(0,index_loc_daily_end-lookback)

        if long_only == False:
            returns_temp = daily_returns.iloc[index_loc_daily_start:index_loc_daily_end][top_x[i]+bottom_x[i]]
            num_assets = len(top_x[i]+bottom_x[i])
            bounds = tuple((0, 2) for _ in range(int(num_assets/2))) + tuple((-2, 0) for _ in range(int(num_assets/2)))
        else:
            returns_temp = daily_returns.iloc[index_loc_daily_start:index_loc_daily_end][top_x[i]]
            num_assets = len(top_x[i])
            bounds = tuple((0, 2) for _ in range(int(num_assets)))
        
        covmat = cov_mat(returns_temp)
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        initial_w = np.ones(num_assets) / num_assets
        w = minimize(risk_target_obj, initial_w, args=(covmat, target_risk,), method='SLSQP', bounds=bounds, constraints=constraints).x
        if any(math.isnan(x) for x in w) == True:
            w = [1/num_assets for _ in w]
        returns_next_period.iloc[i] = np.sum(w * returns_l_s.iloc[i,:num_assets])
    
    return returns_next_period


# In[12]:


#Invest vs Liquidate Functions
def invest_liquidate(flag_fn,returns_next_period,safe_asset_returns):
    for i in range(returns_next_period.shape[0]-1):
        if flag_fn.iloc[i]==True:
            returns_next_period.iloc[i] = safe_asset_returns.iloc[i]
    return returns_next_period

# def stop_loss()


# In[13]:


#Hedging Functions
# def create_hedge()


# In[14]:


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


# In[15]:


# prices = prices.iloc[2625:]
# prices = prices_renew.iloc[2625:]
daily_returns = convert_prices_to_returns(prices)
weekly_returns = daily_weekly_rollup(daily_returns)
monthly_returns = daily_monthly_rollup(daily_returns)


# In[16]:


# SNP500 = pd.read_excel('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/SNP500.xlsx')
# SNP500.set_index('Date', inplace=True)
# SNP500 = SNP500.bfill()
# SNP500_daily_returns = convert_prices_to_returns(SNP500)
# SNP500_monthly_returns = daily_monthly_rollup(SNP500_daily_returns)


# In[17]:


# #Weekly, 3 Long
# lookback_returns = return_lookback(weekly_returns,30)
# rank_df = rank_fn(lookback_returns,"Descending")
# top_3, bottom_3 = select_top_x_bottom_x(rank_df,3)
# returns_l_s = return_series_long_short(weekly_returns,top_3, bottom_3,3)   
# returns_alloc = equally_wtd(returns_l_s,True)

# print('22 year CAGR : ',(((1+returns_alloc[:-1]).prod())**(1/22)-1),"%")
# cumulative_returns = (1 + returns_alloc).cumprod()
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(cumulative_returns.index[:], cumulative_returns[:], label='Cumulative Returns')
# ax.set_xlabel('Date')
# ax.set_ylabel('Cumulative Returns')
# ax.set_title('Cumulative Returns Plot')


# In[ ]:





# In[18]:


lookback_returns = return_lookback(monthly_returns,12)

periods_ud = Periods_Up_Down(monthly_returns,12)

monthly_prices = prices.resample('M').last()
p_values = Reg_p_value(monthly_prices,12)
SMA_5_20 = SMA5_SMA20(monthly_prices)
SMA_20_50 = SMA20_SMA50(monthly_prices)

Week52_H,Week52_L  = week52_high_low_ratio(prices)
Week52_H = Week52_H.resample('M').last().bfill()
Week52_L = Week52_L.resample('M').last().bfill()         

rank_df = rank_fn(lookback_returns,"Descending")
rank_df_2 = rank_fn(Week52_H,"Descending")
x = 2
top_x, bottom_x = select_top_x_bottom_x(rank_df_2,x)
# top_x, bottom_x = select_top_x_bottom_x(rank_df+rank_df_2,x)

lookback_returns_l_s = lookback_returns_long_short(monthly_returns,top_x,bottom_x,x)
returns_l_s = return_series_long_short(monthly_returns,top_x, bottom_x,x,1)
returns_alloc = equally_wtd(returns_l_s,False)
# returns_alloc = risk_parity(returns_l_s,False,daily_returns,top_x,bottom_x)
# returns_alloc = min_vol(returns_l_s,False,daily_returns,top_x,bottom_x)
# returns_alloc = risk_budget(returns_l_s,False,daily_returns,top_x,bottom_x,0.2)
# returns_alloc

print('10 year CAGR : ',np.round((((1+returns_alloc[60:-1]).prod())**(1/17)-1)[0]*100,2),"%")
cumulative_returns = (1 + returns_alloc[60:]).cumprod()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Returns')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Returns')
ax.set_title('Selection - 52 weeks high; #Commodities - 2; Allocation - Equally weighted long only')
ax.grid(True)


# In[19]:


winners = pd.DataFrame(0,columns = monthly_returns.columns, index = monthly_returns.index.year.unique())
for i in range(len(top_x)):
    winners.loc[top_x.index[i].year,top_x[i][0]]+=1
    winners.loc[top_x.index[i].year,top_x[i][1]]+=1
# winners.to_excel('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/winners.xlsx')


# In[ ]:


# returns_alloc = returns_alloc[returns_alloc.index.isin(SNP500_monthly_returns.index)]
# SNP500_monthly_returns = SNP500_monthly_returns[SNP500_monthly_returns.index.isin(returns_alloc.index)]

# print('20 year CAGR : ',np.round((((1+returns_alloc[:-1]).prod())**(1/20)-1),2)[0]*100,"%")
# cumulative_returns = (1 + returns_alloc[:]).cumprod()
# cumulative_return_SNP500 = (1+SNP500_monthly_returns).cumprod()
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(cumulative_returns.index, cumulative_returns, label='Strategy', color = 'blue', linestyle = '-')
# ax.plot(cumulative_returns.index, cumulative_return_SNP500, label='S&P 500', color='red', linestyle='--')
# ax.set_xlabel('Date')
# ax.set_ylabel('Cumulative Returns')
# ax.set_title('Cumulative Returns Plot')


# In[20]:


_, p_value = scipy.stats.ttest_1samp(returns_alloc['returns'].iloc[:-1], 0)
print(np.round(p_value,2))

plt.hist(returns_alloc, bins=10, edgecolor='black', alpha=0.7, color='blue')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram Example')
plt.savefig('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/Results/Histogram.jpeg', format='jpeg')
plt.show()


# In[21]:


selection = ['12 months returns','52 weeks high','SMA5 on SMA20', 'SMA 20 on SMA 50','Regression p-value','Periods Up Down']
allocation = ['Equally weighted Long Only','Equally weighted Long Short','Risk parity','Risk budget 10%','Minimum Volatility']
xs = [2,3,4]
long_only = [True,False]


# In[22]:


import warnings
warnings.filterwarnings("ignore")

strategy_df = pd.DataFrame(columns = ['Selection','Allocation','X',
                                     'CAGR','Mean Returns','Standard Deviation','Sharpe','Sortino',
                                     'Max Drawdown','Calmar','Success Rate','Average Up','Average Down', 'p value'])

for s in selection:
    for a in allocation:
        for x in xs:
                
                print(s,a,x)
                
                if s == '12 months returns':
                    rank_df = rank_fn(lookback_returns,"Descending")
                if s == '52 weeks high':
                    rank_df = rank_fn(Week52_H,"Descending")
                if s == 'SMA5 on SMA20':
                    rank_df = rank_fn(SMA_5_20,"Descending")
                if s == 'SMA 20 on SMA 50':
                    rank_df = rank_fn(SMA_20_50,"Descending")
                if s == 'Regression p-value':
                    rank_df = rank_fn(p_values,"Ascending")
                if s == 'Periods Up Down':
                    rank_df = rank_fn(periods_ud,"Descending")
                    
                top_x, bottom_x = select_top_x_bottom_x(rank_df,x)

                returns_l_s = return_series_long_short(monthly_returns,top_x, bottom_x,x,1)
                
                if a == 'Equally weighted Long Only':
                    returns_alloc = equally_wtd(returns_l_s,True)
                if a == 'Equally weighted Long Short':
                    returns_alloc = equally_wtd(returns_l_s,False)
                if a == 'Risk parity':
                    returns_alloc = risk_parity(returns_l_s,True,daily_returns,top_x,bottom_x)
                if a == 'Minimum Volatility':
                    returns_alloc = min_vol(returns_l_s,True,daily_returns,top_x,bottom_x)
                if a == 'Risk budget 10%':
                    returns_alloc = risk_budget(returns_l_s,True,daily_returns,top_x,bottom_x,0.1)     
                
                returns_alloc = returns_alloc[60:-1]
                
                mean_return,st_dev,sharpe_ratio,sortino_ratio,max_drawdown,calmar_ratio,success_rate,average_up,average_down = strategy_analytics(returns_alloc, rf = 0.05)
                
        
                strategy_df = strategy_df.append([{'Selection' : s,
                                                   'Allocation' : a,
                                                   'X' : x,
                                                   'CAGR' : (((1+returns_alloc).prod())**(1/17)-1)[0],
                                                   'Mean Returns' : mean_return,
                                                   'Standard Deviation' : st_dev,
                                                   'Sharpe' : sharpe_ratio,
                                                   'Sortino' : sortino_ratio,
                                                   'Max Drawdown' : max_drawdown,
                                                   'Calmar' : calmar_ratio,
                                                   'Success Rate' : success_rate,
                                                   'Average Up' : average_up,
                                                   'Average Down' : average_down,
                                                   'p value' : scipy.stats.ttest_1samp([x for x in returns_alloc['returns'] if not np.isnan(x)], 0)}])
            
                
                plt.hist(returns_alloc, bins=10, edgecolor='black', alpha=0.7, color='blue')
                plt.xlabel('Values')
                plt.ylabel('Frequency')
                plt.title('Histogram Example')
                name_string = f'Histogram : Selection - {s};#Commodities - {x} ; Allocation - {a}'
                plt.savefig(f'C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/{name_string}.jpeg',
                            format='jpeg')
                plt.show()
                
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 cumulative_returns = (1 + returns_alloc).cumprod()
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 ax.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Returns')
#                 ax.set_xlabel('Date')
#                 ax.set_ylabel('Cumulative Returns')
#                 ax.set_title(f'Selection - {s};#Commodities - {x} ; Allocation - {a}')
#                 ax.grid(True)
#                 name_string = f'Selection - {s};#Commodities - {x} ; Allocation - {a}'
#                 fig.savefig(f'C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/Results/{name_string}.jpeg',
#                                format='jpeg')


# In[ ]:


strategy_df.to_excel('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/Strategies Analysis v2.xlsx')


# In[ ]:





# In[30]:


lookback_returns = return_lookback(monthly_returns,12)
periods_ud = Periods_Up_Down(monthly_returns,12)
monthly_prices = prices.resample('M').last()
p_values = Reg_p_value(monthly_prices,12)
SMA_5_20 = SMA5_SMA20(monthly_prices)
SMA_20_50 = SMA20_SMA50(monthly_prices)
Week52_H,Week52_L  = week52_high_low_ratio(prices)
Week52_H = Week52_H.resample('M').last().bfill()
Week52_L = Week52_L.resample('M').last().bfill()         

rank_df = rank_fn(Week52_H,"Descending")
x = 2
top_x, bottom_x = select_top_x_bottom_x(rank_df,x)
returns_l_s = return_series_long_short(monthly_returns,top_x, bottom_x,x,1)

# returns_alloc = equally_wtd(returns_l_s,False)
# returns_alloc = risk_parity(returns_l_s,True,daily_returns,top_x,bottom_x)
returns_alloc = min_vol(returns_l_s,True,daily_returns,top_x,bottom_x)
# returns_alloc = risk_budget(returns_l_s,False,daily_returns,top_x,bottom_x,0.2)

print('17 year CAGR : ',np.round((((1+returns_alloc[60:-1]).prod())**(1/17)-1)[0]*100,2),"%")


# In[32]:


# SMA2050LS = returns_alloc
# Week52LS = returns_alloc
# Week52RP  = returns_alloc
# Week52MV  = returns_alloc
# ML = pd.read_excel('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/ML_returns.xlsx', index_col = 0)
# ML.columns = ['returns']


# In[33]:


SMA2050LS = SMA2050LS[240:-2]
Week52RP = Week52RP[240:-2]
Week52LS = Week52LS[240:-2]
Week52MV = Week52MV[240:-2]
# SMA2050RP = SMA2050RP[240:-2]


# In[36]:


SMA2050LS_cum = (1 + SMA2050LS).cumprod()
Week52MV_cum = (1 + Week52MV).cumprod()
Week52RP_cum = (1 + Week52RP).cumprod()
Week52LS_cum = (1 + Week52LS).cumprod()
ML_cum = (1 + ML).cumprod()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(SMA2050LS_cum.index, SMA2050LS_cum, label='SMA 20 on SMA 50 Equally Wtd Long Short', color = 'blue')
ax.plot(Week52MV_cum.index, Week52MV_cum, label='52 Week High Min Vol Long', color = 'red')
ax.plot(Week52RP_cum.index, Week52RP_cum, label='52 Week High Risk Parity Long', color = 'magenta')
ax.plot(Week52LS_cum.index, Week52LS_cum, label='S52 Week High Equally Wtd Long Short', color = 'black')
ax.plot(ML_cum.index, ML_cum, label='Deep Learning', color = 'yellow')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Returns')
ax.set_title('Strategies')
ax.legend()
ax.grid(True)
# fig.savefig(f'C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/Charts/Strategies Combo.jpeg',
#                            format='jpeg')


# In[ ]:


# SMA2050LS = SMA2050LS[240:-2]
# Week52MV = Week52MV[240:-2]
# Week52LS = Week52LS[240:-2]
# Month12LS = Month12LS[240:-2]
# SMA2050RP = SMA2050RP[240:-2]
# strategy_df = pd.DataFrame(columns = ['Selection','Allocation','X',
#                                      'CAGR','Mean Returns','Standard Deviation','Sharpe','Sortino',
#                                      'Max Drawdown','Calmar','Success Rate','Average Up','Average Down'])
returns_alloc = ML
mean_return,st_dev,sharpe_ratio,sortino_ratio,max_drawdown,calmar_ratio,success_rate,average_up,average_down = strategy_analytics(returns_alloc, rf = 0.05)
strategy_df = strategy_df.append([{'Selection' : 'Deep Learning',
                                                   'Allocation' : 'Equally Wtd Long Short',
                                                   'X' : 2,
                                                   'CAGR' : (((1+returns_alloc).prod())**(1/3.5)-1)[0],
                                                   'Mean Returns' : mean_return,
                                                   'Standard Deviation' : st_dev,
                                                   'Sharpe' : sharpe_ratio,
                                                   'Sortino' : sortino_ratio,
                                                   'Max Drawdown' : max_drawdown,
                                                   'Calmar' : calmar_ratio,
                                                   'Success Rate' : success_rate,
                                                   'Average Up' : average_up,
                                                   'Average Down' : average_down}])



# In[ ]:


strategy_df.to_excel('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/Strategies Analysis ML vs others.xlsx')


# In[ ]:


annual_std.columns


# In[ ]:


annual_std = daily_returns.rolling(window=252).std()*(252**0.5)
annual_std = annual_std.iloc[2625:]


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(annual_std.index, annual_std['HRC_STEEL'], label='Steel', color = 'blue')
ax.plot(annual_std.index, annual_std['GOLD'], label='Gold', color = 'red')
ax.plot(annual_std.index, annual_std['ZINC_SPOT'], label='Zinc', color = 'green')
ax.plot(annual_std.index, annual_std['NICKEL'], label='Nickel', color = 'black')
ax.plot(annual_std.index, annual_std['VANADIUM'], label='Vanadium', color = 'yellow')
ax.plot(annual_std.index, annual_std['IRIDIUM'], label='Iridium', color = 'yellow')
ax.set_xlabel('Date')
ax.set_ylabel('Annualized Standard Deviation')
ax.set_title('Annualized Standard Deviation')
ax.legend()
# ax.grid(True)
# fig.savefig(f'C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/Charts/Strategies Combo.jpeg',
#                            format='jpeg')


# In[ ]:


annual_std.to_excel('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Investments Project/Annualized Standard Deviation.xlsx')


# In[ ]:




