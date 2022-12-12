from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import sys

from app.objects.wine_manager import FileManager

#sys.path.append('../objects')
# sys.path.append('../')
# from objects.wine import Wine, FileManager
# from app.objects.wine import FileManager
pd.set_option('display.max_column',13)

# No missing values
# Min quality value = 3
# Max quality value = 8
# Seulement 6 vins ont obtenus une note de 3 !!

fm = FileManager("./data/Wines.csv")
df = fm.read_data()
print(df.columns)
print(df.shape)
print(df.head())
print(df.describe())
df['quality'].value_counts().hist()
plt.show()

