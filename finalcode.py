from pandas import ExcelWriter          #writing to a new excel file
from pandas import ExcelFile            #reading the excel file as an object
import pandas as pd                     #used to make dataframes out of excel files
import numpy as np                      #easy handling of values in the form of an array
import matplotlib.pyplot as plt         #plotting library for graphs and mathematical models.
import scipy                            #easy datascience library
import seaborn as sns                   #seaborn is a package for various graphical functions
import statsmodels.api as sm            #package of various statistical models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings
from sklearn.model_selection import train_test_split
excel_file='fdataset.xlsx'      #object to read the dataset
excel_file1='new.xlsx'
obj=pd.read_excel(excel_file,sheet_name=0,index_col=0)      #object1 creation
gobj=pd.read_excel(excel_file1,sheet_name=0,index_col=0)    #object2 creation
newobj=pd.read_excel(excel_file1,sheet_name=0,index_col=0)  #object3 creation
desired_width = 400                                         #setting the desired width
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 14)                        #displaying only the maxiumum columns
print(obj.head(10))                                             #gives the first 10 lines
print(obj.tail(10))                                             #gives the last 10 lines
print("The number of attributes are: " +str(obj.shape))         #gives the number of rows and columns
print("The number of dimensions are:" +str(obj.ndim))           #gives dimensions
print(" The datatypes are : " + str(obj.dtypes))                #gives the datatypes
print("The number of non null values is : " +str(obj.info()))   #gives number of non null values in attributesThe number of attributes are
print("The number of null values are : " +str(obj.isnull().sum())) #gives number of missing values

des1=obj['CO'].describe(include='all')
#print(des1)

gobj=gobj.drop("Ozone",axis=1)                                  #removes ozone because too many null values
gobj=gobj.drop("Temp",axis=1)                                   #removes temp because too many null values
gobj=gobj.drop("VWS",axis=1)                                    #removes VWS because too many null values
gobj=pd.Series(3*np.random.rand(16), index=["AT","BP","PM10","PM2.5","NOx","SO2","CO","Benzene","Toluene","NH3","NO2","NO","RH","SR","WD","WS"],name='pie chart')
gobj.plot.pie(figsize=(6,6))                                    #plotting pie chart to see contribution of each parameter to air pollution.
plt.show()
plt.scatter(gobj.PM10, gobj.WS,s=10,c='brown')                  #scatter plot functions to see correlation.

plt.show()

des1=obj['AT'].describe(include='all')                          #description functions for all the parameters
print(des1)
des2=obj['BP'].describe(include='all')
print(des2)
des1=obj['PM10'].describe(include='all')
print(des1)
des2=obj['PM2.5'].describe(include='all')
print(des2)
des1=obj['NOx'].describe(include='all')
print(des1)
des1=obj['CO'].describe(include='all')
print(des1)
des2=obj['SO2'].describe(include='all')
print(des2)
des1=obj['CO'].describe(include='all')
print(des1)
des2=obj['Benzene'].describe(include='all')
print(des2)
des1=obj['Toluene'].describe(include='all')
print(des1)
des2=obj['NH3'].describe(include='all')
print(des2)
des1=obj['NO2'].describe(include='all')
print(des1)
des2=obj['NO'].describe(include='all')
print(des2)
des1=obj['RH'].describe(include='all')
print(des1)
des2=obj['SR'].describe(include='all')
print(des2)
des1=obj['WD'].describe(include='all')
print(des1)
des2=obj['WS'].describe(include='all')
print(des2)

gobj.to_excel('new.xlsx')                                         #converting the modified dataframe to excel

ax=gobj['Benzene'].plot.kde()                                     #plotting kernel density function
plt.title("KERNEL DENSITY FUNCTION - Benzene")
plt.show(ax)

var=gobj.boxplot(column=['AT', 'RH', 'BP', 'PM10', 'PM2.5', 'NOx', 'SO2', 'CO', 'Benzene', 'Toluene', 'NH3','NO2','NO','RH','SR','WD','WS'])
plt.title("OUTLIER ANALYSIS")
plt.xlabel("DISTRIBUTION")
plt.ylabel("VALUES")
plt.show(var)                                                       #box plot to identify outliers

f, ax = plt.subplots(figsize=(10, 8))
corr = gobj.corr()                                                  #correlation plot to see dependency
plt.title("CORRELATION ANALYSIS")
plt.show(sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax))

med=obj['WS'].median()                                              #calculating median value for WS
print(med)
mu=obj['WS'].mean()                                                 #caluclating mean value for WS
sigma=obj['WS'].std()
s=np.random.normal(mu,sigma,3247)
count,bins,ignored=plt.hist(s,100,density=True)
plt.plot(bins,1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins-mu)**2/ (2*sigma**2)),linewidth=3)
plt.title("NORMAL DISTRIBUTION CURVE-WS")
plt.show()                                                          #plotting normal distribution curve for WS

#IDENTIFYING DUPLICATE VALUES OF DATASET

dupes=gobj.duplicated()
print(sum(dupes))

#REMOVING DUPLICATE VALUES OF DATASET
dupesremove=gobj.drop_duplicates()
print(dupesremove)
print(gobj.shape)

print(gobj.duplicated())                                            #removes duplicates
#print(gobj.drop_duplicates())

#IQR METHOD TO REMOVE OUTLIERS

def outliers(x):
       return np.abs(x- x.median()) > 1.5*(x.quantile(.75)-x.quantile(0.25))

#GIVES OUTLIERS FOR FIRST COLUMN:

print(gobj.AT[outliers(gobj.AT)])

fobj=obj.drop(['To Date'],axis=1)                                  #removes un-necessary attribute To-Date
print(fobj)
fobj.to_excel("myfdataset.xlsx")                                   #conversion to a new excel file

excel_fi='myfdataset.xlsx'
fi_obj=pd.read_excel(excel_fi,sheet_name=0,index_col=0)
print("After dropping to date:"  +str(fi_obj.shape))               #after dropping to_date

#DROPPING OUTLIERS USING IQR
Q1 = fi_obj.quantile(0.25)
Q3 = fi_obj.quantile(0.75)
IQR = Q3 - Q1

print(IQR)                                                          #printing IQR obatined


fi_obj_cleaned=fi_obj[~((fi_obj < (Q1 - 1.5 * IQR)) |(fi_obj > (Q3 + 1.5 * IQR))).any(axis=1)]
print(fi_obj_cleaned)                                               #removing outliers
fi_obj_cleaned.to_excel("mydataset.xlsx")


excel_f='mydataset.xlsx'                                            #new cleaned dataset
obj=pd.read_excel(excel_f,sheet_name=0,index_col=0)

#BOXPLOT FOR OUTLIER ANALYSIS
var=obj.boxplot(column=['AT', 'RH', 'BP', 'PM10', 'PM2.5', 'NOx', 'SO2', 'CO', 'Benzene', 'Toluene', 'NH3','NO2','NO','RH','SR','WD','WS'])
plt.title("OUTLIER ANALYSIS")
plt.xlabel("DISTRIBUTION")
plt.ylabel("VALUES")
#plt.show(var)

#COUNT OF FILL MISSING VALUES
print("The number of null values are : " +str(obj.isnull().sum())) #gives number of missing values

#TO FILL MISSING VALUES
#WHERE NUMBER OF MISSING VALUES IS LESS-MEAN
#WHERE NUMBER OF MISSING VALUES IS MORE-MEDIAN
obj['AT']=obj['AT'].fillna(value=obj.AT.mean())
obj['BP']=obj['BP'].fillna(value=obj.BP.mean())
obj['PM10']=obj['PM10'].fillna(value=obj.PM10.mean())
obj['PM2.5']=obj['PM2.5'].fillna(value=obj['PM2.5'].mean())
obj['Benzene']=obj['Benzene'].fillna(value=obj.Benzene.mean())
obj['Toluene']=obj['Toluene'].fillna(value=obj.Toluene.mean())
obj['RH']=obj['RH'].fillna(value=obj.RH.mean())
obj['SR']=obj['SR'].fillna(value=obj.SR.mean())
obj['WD']=obj['WD'].fillna(value=obj.WD.mean())
obj['WS']=obj['WS'].fillna(value=obj.WS.mean())
obj['NOx']=obj['NOx'].fillna(value=obj.NOx.median())
obj['SO2']=obj['SO2'].fillna(value=obj.SO2.median())
obj['CO']=obj['CO'].fillna(value=obj.CO.median())
obj['NH3']=obj['NH3'].fillna(value=obj.NH3.median())
obj['NO2']=obj['NO2'].fillna(value=obj.NO2.median())
obj['NO']=obj['NO'].fillna(value=obj.NO.median())
obj.to_excel("d.xlsx")
print("The number of null values are : " +str(obj.isnull().sum()))

excel_f='d.xlsx'
obj=pd.read_excel(excel_f,sheet_name=0,index_col=0)

#TRANSFORMING DATA USING MIN_MAX NORMALIZATION
for columns in obj:
    obj[columns]=pd.DataFrame(((obj[columns]-obj[columns].min())/(obj[columns].max()-obj[columns].min())))
print(obj)
obj.to_excel("d.xlsx")                                              #convert to new excel after modifications

#CHECKING FOR SKEWNESS
for columns in obj:
    print("the skew is : "+columns + "  "+ str(obj[columns].skew()))

excel_f='d.xlsx'
obj=pd.read_excel(excel_f,sheet_name=0,index_col=0)
nobj=pd.read_excel(excel_f,sheet_name=0,index_col=0)
print(obj)

#TRYING TO REMOVE SKEWNESS
for columns in nobj:
   nobj[columns]=pd.DataFrame(nobj[columns].apply(np.sqrt))
print("The skew after sqrt transformation: ")
for columns in nobj:
   print("the skew is : "+columns + "  "+ str(nobj[columns].skew()))

nobj.to_excel('da.xlsx')
excel_f1='da.xlsx'
gobj=pd.read_excel(excel_f1,sheet_name=0,index_col=0)


#PLOTTING THE GRAPH
for columns in obj:
    x=sns.distplot(obj[columns], hist=False, rug=True)
    plt.show(x)
for columns in gobj:
    y=sns.distplot(gobj[columns], hist=False, rug=True)
    plt.show(y)


#SPLITTING THE DATA
train, test = train_test_split(gobj, test_size=0.2)
print(train.shape)
print(test.shape)

#ALGORITHM APPLICATION

df=pd.read_excel("da.xlsx")

#SPLIT THE DATA IN TEST AND TRAIN
X=df[['NO2','NO','NOx','PM2.5','Benzene','Toluene']]
y=df['PM10']
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0) #80% train data and 20% test data
X = sm.add_constant(X)                                                                   #adding regression constant
est = sm.OLS(y, X).fit()                                                                 #fittig the model to what is required

print(est.summary())                                                                     #printing summary
plt.scatter(X['BP'],y)
plt.ylabel("dependent variable")
plt.xlabel("independent variable")
plt.show()
print(X_train)                                                                          #printing Xtrain,Xtest
print(X_test)                                                                           #printing Ytrain,Ytest
print(y_train)
print(y_test)
regressor=LinearRegression()                                                            #calling linear regression
regressor.fit(X_train,y_train)                                                          #fitting it to linear regression
coeff_dr=pd.DataFrame(regressor.coef_,X.columns,columns=['Coefficient'])
print(coeff_dr)
y_pred = regressor.predict(X_test)                                                      #predicting values of y
dff=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
warnings.filterwarnings("ignore")
print(dff)
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))                 #printing Mean Absolute Error
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))                   #printing Mean Square Error
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))    #printing Root Mean Square Error
sns.regplot(y_pred,y_test,color='red')                                                   #final regression plot
plt.show()

#CROSS VALIDATION OF THE PREDICTED MODEL





