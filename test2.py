import pandas as pd
df=pd.read_excel("pre_data.xls")

df =  df.drop(index = df[(df['hour'] == 0)].index.tolist())
df =  df.drop(index = df[(df['hour'] == 1)].index.tolist())
df =  df.drop(index = df[(df['hour'] == 2)].index.tolist())
df =  df.drop(index = df[(df['hour'] == 3)].index.tolist())
df =  df.drop(index = df[(df['hour'] == 4)].index.tolist())
df =  df.drop(index = df[(df['hour'] == 5)].index.tolist())
df =  df.drop(index = df[(df['hour'] == 22)].index.tolist())
df =  df.drop(index = df[(df['hour'] == 23)].index.tolist())
writer = pd.ExcelWriter('pre_data__.xls')
df.to_excel(writer)
writer.save()