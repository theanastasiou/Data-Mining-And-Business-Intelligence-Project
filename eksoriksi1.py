import pandas as pd
import glob
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import seaborn as sns
import csv

#---------------------PRE PROCESING-----------------------
path = r'ergasia_dm&bi' # use your path
all_files = glob.glob(path + "/*.csv")
li = [] 

#diavasma arxeiwn apo fakelo kai append se mia lista
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

#pernei tin lista poy dimiourgithike pio panw kai vazei ola ta arxeia se ena
frame = pd.concat(li, axis=0, ignore_index=True,sort=True)

frame['ac_pow_current'].replace('unknown',0,inplace=True)  #fill uknown with 0
frame['ac_pow_power'].replace('unknown',0,inplace=True)  
frame['humidity_in'].replace('unknown',0,inplace=True)  #fill uknown with 0
frame['humidity_out'].replace('unknown',0,inplace=True)  

frame['statetime'] =pd.to_datetime(frame['statetime'])
frame.set_index(pd.DatetimeIndex(frame['statetime']),inplace=True)

frame['ac_pow_current'] = frame['ac_pow_current'].astype(float)
frame['ac_pow_power'] = frame['ac_pow_power'].astype(float)
frame['humidity_in'] = frame['humidity_in'].astype(float)
frame['humidity_out'] = frame['humidity_out'].astype(float)
frame['temp_in'] = frame['temp_in'].astype(float)
frame['temp_out'] = frame['temp_out'].astype(float)

frame['motion_detected'].replace('EA674E',1,inplace=True) 

frame = frame.groupby(pd.Grouper(freq='1Min'))['ac_pow_current','ac_pow_power','humidity_in','humidity_out','motion_detected','temp_in','temp_out'].mean()

# frame = frame.dropna(thresh=1) #

frame['week'] = frame.index.week  #find week based on statetime
frame['hour'] = frame.index.hour #find week based on statetime
frame['minute'] = frame.index.minute #find week based on statetime
frame['quarter'] = frame.index.quarter #find week based on statetime
frame['month'] = frame.index.month #find week based on statetime
frame['weekday'] = frame.index.weekday #find weekday based on statetime

frame['day'] = frame.index.day  #find week based on statetime
frame['motion_detected'].fillna(0,inplace=True)
frame["ac_pow_current"].fillna(0.0, inplace=True)
frame["ac_pow_power"].fillna(0.0, inplace=True)
#dimiourgia stilis on_off : an to ac_pow_power+ac_pow_current einai megalitero tu 0 tote on(1) a/c else off(0) 
frame['on_off'] = frame.apply(lambda x: x['ac_pow_current'] + x['ac_pow_power'], axis=1)
frame['on_off'] =frame['on_off'].apply(lambda x: 1 if x > 0 else 0)

#dimiourgia stilis an einai sto grafeio i oxi  an to ac_pow_power+ac_pow_current + to motion_detected einai megalitero i iso tu 1 tote in(1) else out(0)
frame['inoffice'] = frame.apply(lambda x: x['ac_pow_current'] + x['ac_pow_power'] + x['motion_detected'], axis=1)
frame['inoffice'] =frame['inoffice'].apply(lambda x: 1 if x >= 1 else 0)

#gemisma kenwn me mean stin vdomada kai an den iparxei metrisi g olokliri tin vdomada mean se olokliri tin stili
frame["temp_in"].fillna(frame.groupby("day")["temp_in"].transform('mean'), inplace=True)
frame["temp_in"].fillna(frame["temp_in"].mean(), inplace=True)
frame["temp_out"].fillna(frame.groupby("day")["temp_out"].transform('mean'), inplace=True)
frame["temp_out"].fillna(frame["temp_out"].mean(), inplace=True)
frame["humidity_in"].fillna(frame.groupby("day")["humidity_in"].transform('mean'), inplace=True)
frame["humidity_in"].fillna(frame["humidity_in"].mean(), inplace=True)
frame["humidity_out"].fillna(frame.groupby("day")["humidity_out"].transform('mean'), inplace=True)
frame["humidity_out"].fillna(frame["humidity_out"].mean(), inplace=True)

# Reset the index of dataf+rame
frame = frame.reset_index()

#dimiourgia bins me COld hot klp string for visualization
bintemp = [-10, 0, 15, 25, 33, 50]
labels = ['Freezing','Cold','Warm','Hot','Very Hot']
frame['temp'] = pd.cut(frame['temp_out']-0+49 *(frame['temp_out']<0),bins=bintemp,labels=labels,right=False)
#dimiourgia bins me COld hot klp string for training
bintemp = [-10, 0, 15, 25, 33, 50]
labels = [1,2,3,4,5]
frame['tempp'] = pd.cut(frame['temp_out']-0+49 *(frame['temp_out']<0),bins=bintemp,labels=labels,right=False)

#dimiourgia bins me morning-afternoon-night klp number
bins = [0, 5, 13, 17, 25]
labels = ['1','2','3','4']
hours = frame['statetime'].dt.hour
frame['bin'] = pd.cut(hours-5+24 *(hours<5),bins=bins,labels=labels,right=False)

#dimiurgia bins me -morning afternoon string 
labels = ['Morning','Afternoon','Evening', 'Night']
hours = frame['statetime'].dt.hour
frame['bin2'] = pd.cut(hours-5+24 *(hours<5),bins=bins,labels=labels,right=False)

frame['statetime']= frame['statetime'].dt.strftime('%Y-%m-%d %H:%M:%S') #metatropi date se str
frame.to_csv("hello.csv") #metatrepo se csv arxeio


#--------------------END OF PRE PREPROCESING-----------------------

#------------------------ CLASSIFICATION - TRAINING  -------------------------

#imports needed
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

#stiles pou tha xrisimopoithun gia to traoning
data = frame.loc[:,['motion_detected','bin','humidity_in','humidity_out','temp_out','temp_in','tempp']]
X=data
y=frame['on_off'] #target stili

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #20% test set 

from sklearn.metrics import classification_report

target_names = ['class 0', 'class 1']

#DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)  #train
y_pred = clf.predict(X_test)  #predict
decisiontreeac = metrics.accuracy_score(y_test, y_pred) #accuracy
decisiontreepre = metrics.precision_score(y_test, y_pred) #precision
decisiontreerec = metrics.recall_score(y_test, y_pred,average='macro') #recall
decisiontreefm = metrics.f1_score(y_test, y_pred,average='weighted') #f1-score
print("Decision Tree Classification Report")
print(classification_report(y_test, y_pred, target_names=target_names)) #classification report

# for a in range(10):
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
#   clf = clf.fit(X_train,y_train,)
#   y_pred = clf.predict(X_test)
#   print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print("Decision Tree CM")
print(confusion_matrix(y_test, y_pred)) #confusion matrix for desicion tree

#GaussianNB
gnb = GaussianNB()
#train the algorithm on training data and predict using the testing data
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("GuassianNB accuracy : ",precision_score(y_test, y_pred,average='micro'))
NBac = metrics.accuracy_score(y_test, y_pred)
NBpre = metrics.precision_score(y_test, y_pred)
NBrec = metrics.recall_score(y_test, y_pred,average='macro')
NBfm = metrics.f1_score(y_test, y_pred,average='weighted')
print("GuassianNB Classification Report")
print(classification_report(y_test, y_pred, target_names=target_names))
print("Guassian CM")
print(confusion_matrix(y_test, y_pred)) #confusion matrix for GNB
#LinearSVC
svc_model = LinearSVC(random_state=0)
#train the algorithm on training data and predict using the testing data
y_pred = svc_model.fit(X_train, y_train).predict(X_test)
LSVCac = metrics.accuracy_score(y_test, y_pred)
LSVCpre = metrics.precision_score(y_test, y_pred)
LSVCrec = metrics.recall_score(y_test, y_pred,average='macro')
LSVCfm = metrics.f1_score(y_test, y_pred,average='weighted')
print("SVC Classification Report")
print(classification_report(y_test, y_pred, target_names=target_names))
print("SVC CM")
print(confusion_matrix(y_test, y_pred)) #confusion matrix for SVC

n_neighbors=3
#KNeighborsClassifier with 3 neighbors
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
KNNac = metrics.accuracy_score(y_test, y_pred)
KNNpre = metrics.precision_score(y_test, y_pred,average='micro')
KNNrec = metrics.recall_score(y_test, y_pred,average='macro' )
KNNfm = metrics.f1_score(y_test, y_pred,average='weighted')
print("KNN Classification Report")
print(classification_report(y_test, y_pred, target_names=target_names))
print("KNN CM")
print(confusion_matrix(y_test, y_pred)) #confusion matrix for KNN

#------------------------ END  CLASSIFICATION - TRAINING  -------------------------

#-------------------PLOTS-----------------------------------------------
#TA PLOTS EXOUN MPEI SE SXOLIA GIA SKOPOUS DIKIS MAS EUKOLIAS STO TELOS

# dat22 = frame.loc[:,['inoffice','temp_out']]
# # reduced_data = PCA(n_components=2).fit_transform(dat1)
# # results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])
# sns.scatterplot(x="inoffice", y="temp_out", hue=frame['on_off'], data=dat22)
# #plt.title('Before Kmeans Clustering with 2 dimensions')
# plt.show()

# fc = sns.factorplot(x="bin", hue="on_off", col="tempp", 
#                     data=frame, kind="count",
#                     palette=["#FF9999","#FFE888"])
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(X_train['temp_out'], y_train, label="temp_out")
# #ax.plot(frame['inoffice'], frame['on_off'], label="cat")
# ax.legend()
# plt.show()


# X_train.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
#            xlabelsize=8, ylabelsize=8, grid=False)    
# plt.tight_layout(rect=(0, 0, 3, 3))   
# plt.show()


# # Correlation Matrix Heatmap
# f, ax = plt.subplots(figsize=(10, 6))
# corr = frame.corr()
# hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
#                  linewidths=.05)
# f.subplots_adjust(top=0.93)
# t= f.suptitle('Dataset Attributes Correlation Heatmap', fontsize=14)
# plt.show()

# #2dimensions
# cols = ['humidity_out', 'temp_out', 'bin', 'humidity_in','temp_in']
# pp = sns.pairplot(X_train[cols], size=1.8, aspect=1.8,
#                   plot_kws=dict(edgecolor="k", linewidth=0.5),
#                   diag_kind="kde", diag_kws=dict(shade=True))

# fig = pp.fig 
# fig.subplots_adjust(top=0.93, wspace=0.3)
# t = fig.suptitle('Dataset Attributes Pairwise Plots', fontsize=14)  
# plt.show()  

# # import the seaborn module
# import seaborn as sns
# # import the matplotlib module
# import matplotlib.pyplot as plt
# # set the background colour of the plot to white
# sns.set(style="whitegrid", color_codes=True)
# # setting the plot size for all plots
# sns.set(rc={'figure.figsize':(11.7,8.27)})
# # create a countplot
# sns.countplot(y="bin", hue="on_off", data=frame)
# # Remove the top and down margin
# sns.despine(offset=10, trim=True)
# plt.show()

# lp = sns.lmplot(data=frame,
#                 x='temp_out', 
#                 y='hour', 
#                 hue='on_off',
#                 palette=['#FF9999', '#FFE888'],
#                 fit_reg=True,
#                 legend=True,
#                 scatter_kws=dict(edgecolor="k", linewidth=0.5))
# plt.show()

#END PLOTS-----------------------------------------------------------------

from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA #for plotting
from sklearn.cluster import KMeans

#------------------CLUSTERING---------
#Before Kmeans plot
#dat1 = frame.loc[:,['temp_out','hour']]
dat1= frame.loc[:,['motion_detected','bin','humidity_in','humidity_out','temp_out','temp_in','weekday','tempp']]

reduced_data = PCA(n_components=2).fit_transform(dat1)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])
sns.scatterplot(x="pca1", y="pca2", hue=frame['on_off'], data=results)
plt.title('Before Kmeans Clustering with 2 dimensions')
plt.show()

#After Kmeans plot
#dat = frame.loc[:,['temp_out','hour']]
clustering_kmeans = KMeans(n_clusters=2, init='random',n_init=100, n_jobs=-1)
dat1['clusters'] = clustering_kmeans.fit_predict(dat1)
### Run PCA on the data and reduce the dimensions in pca_num_components dimensions
reduced_data = PCA(n_components=2).fit_transform(dat1)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])
sns.scatterplot(x="pca1", y="pca2", hue=dat1['clusters'], data=results)
plt.title('K-means Clustering with 2 dimensions')
plt.show()

#-------------------------Writing Metrics in a file --------------------#
with open("Report.csv","w") as csv_file:
       writer = csv.writer(csv_file,delimiter='\t')
       writer.writerow(['Statistic Measure '] + ['Decision Tree'] + ['Naive Bayes'] + ['LinearSVC'] + ['KNN']) 
       writer.writerow(['Accuracy']  +  [ float("{0:.4f}".format(decisiontreeac))] + [ float("{0:.4f}".format(NBac))]  + [ float("{0:.4f}".format(LSVCac))] +[ float("{0:.4f}".format(KNNac))])
       writer.writerow(['Precision'] +  [float("{0:.4f}".format(decisiontreepre))] + [ float("{0:.4f}".format(NBpre))] + [ float("{0:.4f}".format(LSVCpre))]+[ float("{0:.4f}".format(KNNpre))]) 
       writer.writerow(['Recall']    + 	[ float("{0:.4f}".format(decisiontreerec))]   + [ float("{0:.4f}".format(NBrec))]    + [ float("{0:.4f}".format(LSVCrec))]+[ float("{0:.4f}".format(KNNrec))])
       writer.writerow(['F-Measure'] +  [ float("{0:.4f}".format(decisiontreefm))]+ [ float("{0:.4f}".format(NBfm))] + [ float("{0:.4f}".format(LSVCfm))]+[ float("{0:.4f}".format(KNNfm))]) 
            

# from sklearn.cluster import KMeans
# # Number of clusters
# kmeans = KMeans(n_clusters=2)
# # Fitting the input data
# kmeans = kmeans.fit(X)
# # Getting the cluster labels
# labels = kmeans.predict(X)
# # Centroid values
# centroids = kmeans.cluster_centers_

#----------proetimasia tou test set gia predict me KNN algorithm
frame_test = pd.read_csv('hellotest.csv', index_col=None, header=0)
#print(frame_test.isna().sum())
data_test = frame_test.loc[:,['motion_detected','bin','humidity_in','humidity_out','temperature_out','temperature_in','temp']]

#KNN
n_neighbors=3
#create an object of type KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(data_test)
#print the accuracy score of the model
predictions_file = pd.read_csv('ac_statetime_test.csv', index_col=None, header=0)
predictions_file['predictions'] = y_pred
predictions_file.to_csv("ac_statetime_test.csv") #metatrepo se csv arxeio

