import pandas as pd

df = pd.read_csv("dataset.csv")
drop = ['type', 'nameOrig', 'nameDest']
df = df.drop(columns=drop)

df.fillna(df.mean(), inplace=True)

df.to_csv("file_clean.csv", index=False)

#Plusieurs iterations du fichier clean ont eu lieu pour les differents tests, en voici un a titre d'exemple