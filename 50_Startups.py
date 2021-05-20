'''
Prepare a prediction model for profit of 50_startups data.
Do transformations for getting better predictions of profit and
make a table containing R^2 value for each prepared model.

R&D Spend -- Research and devolop spend in the past few years
Administration -- spend on administration in the past few years
Marketing Spend -- spend on Marketing in the past few years
State -- states from which data is collected
Profit  -- profit of each state in the past few years
'''

# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor

# create a pandas dataframe from a csv file
df = pd.read_csv('50_startups.csv')

# View Data
print(df.head())

# Perform Exploratory Data Analysis
# Identifying the number of features or columns
print("Dataset has {} columns".format(len(df.columns)))

# Identifying the features or columns
print("The columns in our dataset are :",list(df.columns))

# We can use shape to see the size of the dataset
print(df.shape) # has 50 rows and 5 columns

# Identifying the data types of features
print(df.dtypes)

# Checking if the dataset has empty cells 
print(df.isnull().values.any()) # Returns False as there are no null values

# Identifying the number of empty cells by features or columns
print(df.isnull().sum())

# Info can also be used to check the datatypes 
df.info() # shows the datatypes, non-null count of all the columns

# Describe will help to see how numerical data has been spread. 
# We can see some of the measure of central tendency, and percentile values.
print(df.describe())

# Graphical Univariate Analysis:
# Histogram

# Histogram using matplotlib
plt.figure(figsize=(20,10))
plt.subplot(2,4,1)
plt.hist(df['R&D Spend'], histtype='step', color='teal', density=False)
plt.title("Histogram of 'R&D Spend'")
plt.subplot(2,4,2)
plt.hist(df['Administration'], histtype='step', color='teal', density=False)
plt.title("Histogram of 'Administration'")
plt.subplot(2,4,3)
plt.hist(df['Marketing Spend'], histtype='step', color='teal', density=False)
plt.title("Histogram of 'Marketing Spend'")
plt.subplot(2,4,4)
plt.hist(df['Profit'], histtype='step', color='teal', density=False)
plt.title("Histogram of 'Profit'")

# Histogram using seaborn
plt.subplot(2,4,5)
sns.histplot(data=df, x='R&D Spend', hue='State', palette='pastel')
plt.title("Histogram of 'R&D Spend'")
plt.subplot(2,4,6)
sns.histplot(data=df, x='Administration', hue='State', palette='pastel')
plt.title("Histogram of 'Administration'")
plt.subplot(2,4,7)
sns.histplot(data=df, x='Marketing Spend', hue='State', palette='pastel')
plt.title("Histogram of 'Marketing Spend'")
plt.subplot(2,4,8)
sns.histplot(data=df, x='Profit', hue='State', palette='pastel')
plt.title("Histogram of 'Profit'")
# plt.savefig('50_startups_hist.png', bbox_inches='tight')

# Normal Q-Q plot
plt.figure(figsize=(20,10))
plt.subplot(2,4,1)
probplot(df['R&D Spend'], plot=plt)
plt.title("Normal Q-Q plot of 'R&D Spend'")
plt.subplot(2,4,2)
probplot(df['Administration'], plot=plt)
plt.title("Normal Q-Q plot of 'Administration'")
plt.subplot(2,4,3)
probplot(df['Marketing Spend'], plot=plt)
plt.title("Normal Q-Q plot of 'Marketing Spend'")
plt.subplot(2,4,4)
probplot(df['Profit'], plot=plt)
plt.title("Normal Q-Q plot of 'Profit'")

# density plot
plt.subplot(2,4,5)
sns.distplot(df['R&D Spend'], kde=True) # color='#1FFBAA'
plt.title("Density plot of 'R&D Spend'")
plt.subplot(2,4,6)
sns.distplot(df['Administration'], kde=True)
plt.title("Density plot of 'Administration'")
plt.subplot(2,4,7)
sns.distplot(df['Marketing Spend'], kde=True)
plt.title("Density plot of 'Marketing Spend'")
plt.subplot(2,4,8)
sns.distplot(df['Profit'], kde=True)
plt.title("Density plot of 'Profit'")
# plt.savefig('50_startups_qq_density.png', dpi=300, bbox_inches='tight')

# Boxplot 
plt.figure(figsize=(8,8))
df.boxplot() # .hist()
plt.title("Boxplots of all numerical features")
# plt.savefig('50_startups_boxplot.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(8,8))
sns.lineplot(data=df[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']])
plt.title("Lineplots of all numerical features")
# plt.savefig('50_startups_lineplot.png', dpi=300, bbox_inches='tight')

# Checking correlation
correlation = df.corr()
print("Correlation Matrix: \n", correlation)
plt.figure(figsize=(5,5))
sns.heatmap(correlation, annot=True)
# plt.savefig("50_Startups_Corr_heatmap.png", dpi=300, bbox_inches='tight')

# pairplot between variables
import seaborn as sns
sns.pairplot(df)
# plt.savefig("50_Startups_pairplot.png", dpi=300, bbox_inches='tight')
# plt.show() 

# pd.plotting.scatter_matrix(df)
# plt.show()

# Inference from pairplot: 
# Profit has good linear relationship with R&D spend, OK kinda linear relationship with Marketing Spend 
# and there is no relation with administration
# R&D, Marketing does show slight collinearity


# All the numerical columns look normal from the graphs. Let us check using the Shapiro test
Ho='data is normal'
Ha='data is not normal'
alpha = 0.05
def normality_check(df):
	for columnName, columnData in df.iteritems():
		print("Shapiro test for {}".format(columnName))
		res = stats.shapiro(columnData)
		pvalue = round(res[1], 2)
		if pvalue>alpha:
			print("pvalue = {pvalue} > alpha={alpha}. We fail to reject Null Hypothesis. {Ho}".format(pvalue=pvalue, alpha=alpha, Ho=Ho))
		else:
			print("pvalue = {pvalue} <= alpha={alpha}. We fail to reject Null Hypothesis. {Ha}".format(pvalue=pvalue, alpha=alpha, Ha=Ha))

# Driver code
normality_check(df.loc[:, df.columns != 'State']) # From Shapiro test, All numerical columns are normally distributed


# We have one categorical feature in out dataset. Lets create dummy variables.
# One hot encoding
states = pd.get_dummies(df['State'], drop_first=True)
# print(states)

df_new = pd.concat([df.iloc[:, df.columns!='State'], states], axis=1)
print(df_new.tail())

# Building a regression model using statsmodels.formula.api
# defining a function for regression model
def smf_OLS(df):
	target = 'Profit'
	x = df.iloc[:, df.columns!=target]
	y = df.iloc[:, df.columns==target]
	# print(x.columns, y.columns)

	# Instantiating the regressoin model
	model = smf.ols(formula='y ~ x', data=df)
	# print(dir(model))	

	# Training the model
	res = model.fit()
	# print(dir(res))
	print(res.summary())

	# Confidence values 99%
	print(res.conf_int(0.01))

	#estimated coefficients
	print("Estimated Coef: ",res.params)
	#R2
	print("R-squared value: ",res.rsquared)
	#calculating the F-statistic
	F = res.mse_model / res.mse_resid
	print("F-statistic: ",F)
	# F-statistic provided by the res object
	# print(res.fvalue)

	# Predicting values
	y_pred = res.predict(x)

	# For each X, calculate VIF and save in dataframe
	vif = pd.DataFrame()
	vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
	vif["features"] = x.columns
	print(vif) # there isn't any multicollinearity issue

	# # # calculating VIF's values of independent variables
	# x = df[['Marketing Spend', 'Administration', 'Florida', 'New York']]
	# y = df['R&D Spend']
	# rsq_rd = smf.ols('y~x',data=df).fit().rsquared  
	# vif_rd = 1/(1-rsq_rd)
	# print("vif_rd: ", vif_rd) # 2.495

	# x = df[['R&D Spend', 'Administration', 'Florida', 'New York']]
	# y = df['Marketing Spend']
	# rsq_mark = smf.ols('y~x',data=df).fit().rsquared  
	# vif_mark = 1/(1-rsq_mark)
	# print("vif_mark: ", vif_mark) #2.416

	# x = df[['R&D Spend', 'Marketing Spend', 'Florida', 'New York']]
	# y = df['Administration']
	# rsq_admin = smf.ols('y~x',data=df).fit().rsquared  
	# vif_admin = 1/(1-rsq_admin)
	# print("vif_admin: ", vif_admin) #1.177

	# # Storing vif values in a data frame
	# vifs = {'Variables':['R&D Spend','Marketing Spend','Administration'],'VIF':[vif_rd,vif_mark,vif_admin]}
	# Vif_frame = pd.DataFrame(vifs)  
	# print(Vif_frame) # there isn't any multicollinearity issue


	# checking the model assumptions
	# Linearity 
	# Observed values VS Fitted values
	plt.figure(figsize=(8,8))
	plt.scatter(y,y_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

	# Check Normality of residuals using histogram
	# histogram
	plt.figure(figsize=(8,8))
	plt.hist(res.resid_pearson)

	# Checking normality of residuals using qqplot
	plt.figure(figsize=(8,8))
	probplot(res.resid_pearson, dist="norm", plot=plt)

	# Homoscedasticity 
	# Residuals VS Fitted Values 
	plt.figure(figsize=(8,8))
	plt.scatter(y_pred,res.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
	plt.show()

	y_series = y.squeeze()
	# print(type(y_series))
	frame = {'Actuals':y_series,'Predicted': y_pred}
	res_df = pd.DataFrame(frame)
	res_df['Residuals'] = res_df['Actuals'] - res_df['Predicted']
	# scale = preprocessing.StandardScaler()
	# res_df['std_Residuals'] = scale.fit_transform(res_df[['Residuals']])
	# print(res.resid_pearson)
	print(res_df)

# driver code
smf_OLS(df_new)

# The accuracy of out model is 95.1%