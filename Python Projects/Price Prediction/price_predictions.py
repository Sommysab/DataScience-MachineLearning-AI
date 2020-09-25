import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns 
from fbprophet import Prophet as P


avocado_df = pd.read_csv('avocado.csv')
avacado_df = avocado_df.sort_values("Date")


# Visual

plt.figure(figsize=(10,10))

plt.plot(avocado_df['Date'], avocado_df['AveragePrice'])

plt.figure(figsize=(25,12))

sns.countplot(x='region', data= avocado_df)

sns.countplot(x='year', data=avocado_df)



# LOGIC 1

avocado_p_df = avocado_df [['Date', 'AveragePrice']]

avocado_p_df.rename(columns={'Date': 'ds', 'AveragePrice': 'y'})

m = P()

m.fit(avocado_p_df)

future = m.make_future_dataframe(periods=365)

forecast = m.predict(future)

figure = m.plot(forecast, xlabel='Date', ylabel='Price')

figure = m.plot_components(forecast)



# LOGIC 2

avocado_df_sample = avocado_df[avocado_df['region']== 'West']

avocado_df_sample = avocado_df_sample.sort_values('Date')

plt.figure(figsize=(10,10))



plt.plot(avocado_df_sample['Date'], avocado_df_sample['AveragePrice'])

avocado_df_sample.rename(columns={'Date': 'ds', 'AvergePrice':'y'})

m = P()

m.fit(avocado_df_sample)

future = m.make_future_dataframe(periods=365)

forecast = m.predict(future)

figure = plt.plot(forecast, xlabel='Date', ylabel='Price')

figure = m.plot_components(forecast)


