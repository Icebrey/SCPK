import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dataset/heart.csv')

sns.set(rc = {'figure.figsize':(12,10)})
sns.set_context("talk",font_scale=0.5)
sns.heatmap(df.corr(), cmap='Blues', annot=True)
plt.savefig('heatmap.png')

sns.pairplot(data = df, hue = 'target', palette = ['Red', 'Blue'])
plt.savefig('pairplot.png')