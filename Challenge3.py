import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import statsmodels.api as sm


data = pd.read_csv('loan.csv', low_memory=False)

"""plt.hist(data.int_rate,normed=1)
plt.ylabel('Percentages of Loans')
plt.xlabel('Interest Rates(%)')
plt.savefig('ir.png')
plt.clf()

plt.scatter(data.loan_amnt,data.int_rate)
plt.savefig('AmtvIntRate.png')
plt.clf()"""

data2 = data[['int_rate','loan_amnt','dti']]
reg = data2.groupby('int_rate',as_index = False).mean()

y = reg.int_rate
X = reg['loan_amnt','dti']
X["constant"] = 1

model = sm.OLS(y, X)
results = model.fit()
print results.summary()

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(results, 0, y_true=None, ax=ax)
ax.set_ylabel("Interest Rate")
ax.set_xlabel("Average Loan Amount($)")
ax.set_title("Linear Regression")

plt.savefig('Regression_LA.png')
plt.clf()

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(results, 1, y_true=None, ax=ax)
ax.set_ylabel("Interest Rate")
ax.set_xlabel("Average Debt to Income Ratio($)")
ax.set_title("Linear Regression")

plt.savefig('Regression_DTI.png')
plt.clf()