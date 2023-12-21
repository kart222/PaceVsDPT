import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


url = "https://dunksandthrees.com/"
req = requests.get(url)

soup = BeautifulSoup(req.content, 'html.parser')
tabela = soup.find(name='table')





df = pd.read_html(str(tabela))[0]
drtg = []
dpl = []

for i in range(len(df)-2):
    drtg.append(float(df.iloc[i][4].split()[0]))
    dpl.append(float(df.iloc[i][10].split()[0]))



z = [x for x in zip(drtg, dpl)]

z.sort(key = lambda x: x[0], reverse=True)


drtg = np.array([x[0] for x in z]).reshape(-1, 1)
dpl = np.array([x[1] for x in z])

reg = LinearRegression().fit(drtg, dpl)
X, Y = reg.coef_[0], reg.intercept_

plt.scatter(drtg, dpl)
plt.plot(drtg, X * drtg + Y)
plt.xlabel('Defensive Rating')
plt.title(f"y = {X} * x + {Y}".format(np.round(X, 2), np.round(Y, 2)))
plt.ylabel('Avg DPL')
plt.savefig('/Users/Karthik/code/PaceVsDPT/dRTGvsDPL.jpg')
plt.show()
