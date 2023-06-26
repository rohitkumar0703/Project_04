import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#read data
data = pd.read_csv('Unemployment.csv')
print(data.head())
#to get information of data
data.info
print(data.describe)
#to check the null value
print (data.isnull().sum())
data.columns = ("state","date","Frequency","Estimated unemployment Rate","Estimated Employed",
                "Estimated labour participation Rate","Longitude","Latitude","Region")
print(data)

data.columns = ("state","date","Frequency","Estimated unemployment Rate","Estimated Employed",
                "Estimated labour participation Rate","Longitude","Latitude","Region")
plt.title("Employmrnt rate in India")
sns.histplot(x="Estimated Employed",hue="Region", data=data)
plt.show()

plt.figure(figsize=(12,10))
plt.title("Unemployment Rate iof India")
