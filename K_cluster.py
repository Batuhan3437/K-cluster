import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


df=pd.read_csv("reading_data/Avm_Musterileri.csv")
#df.head()

#plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'])
#plt.xlabel('Annual Income (k$)')
#plt.ylabel('Spending Score (1-100)')
#plt.show()

df.rename(columns={'Annual Income (k$)':"income"},inplace=True)
df.rename(columns={'Spending Score (1-100)':"score"},inplace=True)

#scale etme normalize etme
scaler=MinMaxScaler()

scaler.fit(df[['income']])
df['income']=scaler.transform(df[['income']])
scaler.fit(df[['score']])
df['score']=scaler.transform(df[['score']])
#df.head()


k_range=range(1,11)
list_dist=[]

for k in k_range:
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(df[['income', 'score']])
    list_dist.append(kmeans_model.inertia_)

# Plotting burda k nın 5 olacağını belirledik
#plt.xlabel('K')
#plt.ylabel('Distortion değeri (inertia)')
#plt.plot(k_range, list_dist)
#plt.show()

#k=5 için kmeans modeli 
kmeans_modelim=KMeans(n_clusters=5)
y_predicted=kmeans_modelim.fit_predict(df[['income','score']])
#print(y_predicted)

df['cluster']=y_predicted
#df.head()

#centroidler gözükmesi için
#print(kmeans_modelim.cluster_centers_)

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
df4=df[df.cluster==3]
df5=df[df.cluster==4] 

#tablo oluşturma işlemleri
plt.xlabel('income')
plt.ylabel('score')

plt.scatter(df1['income'],df1['score'],color='green')
plt.scatter(df2['income'],df2['score'],color='red')
plt.scatter(df3['income'],df3['score'],color='black')
plt.scatter(df4['income'],df4['score'],color='orange') 
plt.scatter(df5['income'],df5['score'],color='purple') 

plt.scatter(kmeans_modelim.cluster_centers_[:,0],kmeans_modelim.cluster_centers_[:,1],color='blue',marker='X',label='centroid')
plt.legend()
plt.show()
  