import numpy as np
import pandas as pd

from hmmlearn import hmm
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(15, 10)})

np.random.seed(42069)

import warnings; warnings.simplefilter('ignore')

DATA_PATH = "D:/Tick Data/GLD.csv"

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
df.sort_index()

df.drop([' Volume', ' Open', ' High', ' Low'],axis=1, inplace=True)
df.columns = ['Close']

nullvaluecheck = pd.DataFrame(df.isna().sum().sort_values(ascending=False)*100/df.shape[0],columns=['missing %']).head(60)
nullvaluecheck.style.background_gradient(cmap='PuBu')

returns = np.log(df['Close']).diff()
returns.dropna(inplace=True)

returns.plot(kind='hist',bins=150)
plt.title(label='Distribution of GOLD Returns', size=15)
plt.show()

split = int(0.2*len(returns))
X = returns[:-split]
X_test = returns[-split:]

pd.DataFrame(X).plot()
plt.title(label='GOLD Training Set', size=15)
plt.show()

pd.DataFrame(X_test).plot()
plt.title(label='GOLD Testing Set', size=15)
plt.show()

X = X.to_numpy().reshape(-1, 1)
X_test = X_test.to_numpy().reshape(-1, 1)

model = hmm.GaussianHMM(n_components=2, covariance_type="diag", verbose=True)

get_ipython().run_cell_magic('time', '', 'model.fit(X)')

model.transmat_ = np.array([
                            [0.8, 0.2],
                            [0.2, 0.8]
                           ])

Z = model.predict(X_test)
Z_train = model.predict(X)

# Compute State Changes
returns_train0 = np.empty(len(Z_train))
returns_train1 = np.empty(len(Z_train))
returns_train0[:] = np.nan
returns_train1[:] = np.nan

# Create series for each state change
returns_train0[Z_train == 0] = returns[:-split][Z_train == 0]
returns_train1[Z_train == 1] = returns[:-split][Z_train == 1]

# Plot the Volatility Regime and the states
fig, ax = plt.subplots(figsize=(15,10))

plt.subplot(211)
plt.plot(Z)
plt.title(label='GOLD Training Volatility Regime', size=15)

plt.subplot(212)
plt.plot(returns_train0, label='State_0 (High Volatility)', color='r')
plt.plot(returns_train1, label='State_1 (Low Volatility)', color='b', )
plt.title(label='GOLD Training Volatility Clusters', size=15)
plt.legend()
plt.tight_layout()


# Compute State Changes
returns0 = np.empty(len(Z))
returns1 = np.empty(len(Z))
returns0[:] = np.nan
returns1[:] = np.nan

# Create series for each state change
returns0[Z == 0] = returns[-split:][Z == 0]
returns1[Z == 1] = returns[-split:][Z == 1]

# Plot the Volatility Regime and the states
fig, ax = plt.subplots(figsize=(15,10))

plt.subplot(211)
plt.plot(Z)
plt.title(label='GOLD Volatility Regime', size=15)

plt.subplot(212)
plt.plot(returns0, label='State_0 (High Volatility)', color='r')
plt.plot(returns1, label='State_1 (Low Volatility)', color='b')
plt.title(label='GOLD Volatility Clusters', size=15)

plt.legend()
plt.tight_layout()




