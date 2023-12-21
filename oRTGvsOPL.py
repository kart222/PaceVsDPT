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
ortg = []
opl = []

for i in range(len(df)-2):
    ortg.append(float(df.iloc[i][3].split()[0]))
    opl.append(float(df.iloc[i][9].split()[0]))



z = [x for x in zip(ortg, opl)]

z.sort(key = lambda x: x[0], reverse=True)


ortg = np.array([x[0] for x in z]).reshape(-1, 1)
opl = np.array([x[1] for x in z])

reg = LinearRegression().fit(ortg, opl)
X, Y = reg.coef_[0], reg.intercept_

plt.scatter(ortg, opl)
plt.plot(ortg, X * ortg + Y)
plt.xlabel('Offensive Rating')
plt.title(f"y = {X} * x + {Y}".format(np.round(X, 2), np.round(Y, 2)))
plt.ylabel('Avg OPL')
plt.savefig('/Users/Karthik/code/PaceVsDPT/oRTGvsOPL.jpg')
plt.show()
