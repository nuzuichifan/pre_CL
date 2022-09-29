start_hour=2880#5/1
end_hour=7295#10/31
list_=[ '地面气压(hPa)', '气温2m(℃)', '地表温度(℃)', '露点温度(℃)', '相对湿度(%)', '北向风速(V,m/s)', '东向风速(U,m/s)',
 '总太阳辐射度(down,J/m2)', '净太阳辐射度(net,J/m2)', '紫外强度(J/m2)', '北京时(UTC+8)', '冷负荷（kW）']
import pandas as pd

df2017=pd.read_excel("data2.xls",sheet_name='2017',usecols=list_)
df2018=pd.read_excel("data2.xls",sheet_name='2018',usecols=list_)
df2019=pd.read_excel("data2.xls",sheet_name='2019',usecols=list_)
df2020=pd.read_excel("data2.xls",sheet_name='2020',usecols=list_)
#df2021=pd.read_excel("data2.xls",sheet_name='2021',usecols=list_)
df2017=df2017.loc[start_hour:end_hour,:]
df2018=df2018.loc[start_hour:end_hour,:]
df2019=df2019.loc[start_hour:end_hour,:]
df2020=df2020.loc[start_hour:end_hour,:]
#df2021=df2021.loc[start_hour:end_hour,:]
df=pd.concat([df2017,df2018,df2019,df2020])
df['冷负荷（kW）'].fillna(0, inplace = True)
df['北京时(UTC+8)']=pd.to_datetime(df['北京时(UTC+8)'])
df['year']=df['北京时(UTC+8)'].dt.strftime('%Y')
df['mouth']=df['北京时(UTC+8)'].dt.strftime('%m')
df['day']=df['北京时(UTC+8)'].dt.strftime('%d')
df['hour']=df['北京时(UTC+8)'].dt.strftime('%H')

# print(df)
writer = pd.ExcelWriter('pre_data.xls')
df.to_excel(writer)
writer.save()