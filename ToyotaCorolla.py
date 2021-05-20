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
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the csv file as pandas dataframe
df = pd.read_csv("ToyotaCorolla.csv")

# View data
print(df.head())

print(df.shape) # 1436 rows, 38 columns

# Taking only ["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"] columns as per the problem statement
df_new = df.loc[:, ["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
print(df_new.head())

# print(df_new.isnull().any())

# print(df_new.describe())

# pairplot
# sns.pairplot(df_new)
# plt.title("Pairplot")
# plt.show()

corr = df_new.corr()
print(corr)

sns.heatmap(corr, annot=True)
plt.show()

def smf_OLS(df):
	global r2_value
	smf_OLS.counter += 1
	print(smf_OLS.counter)
	target = 'Price'
	x = df.iloc[:, df.columns!=target]
	y = df.iloc[:, df.columns==target]
	print(x.columns, y.columns)

	model = smf.ols(formula='y~x', data=df)
	res = model.fit()
	# print(dir(res))
	print(res.summary()) # The p>|t| for Doors columns is very high. We can eliminate that columns in future predictions

	# Confidence values 99%
	print(res.conf_int(0.01))

	r2_value = round(res.rsquared,4)
	print("R-squared: ",res.rsquared)

	# Predicting values
	y_pred = res.predict(x)

	# For each X, calculate VIF and save in dataframe
	vif = pd.DataFrame()
	vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
	vif["features"] = x.columns
	print(vif) # there isn't any multicollinearity issue

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

	return r2_value

smf_OLS.counter = 0
lst = []
# Driver code
lst.append(smf_OLS(df_new))
lst.append(smf_OLS(df_new.loc[:, df_new.columns!='Doors']))

r2 = pd.DataFrame(columns = ['Model Name', 'R-squared value'])

for i in range(smf_OLS.counter):
	lst.append(r2_value)
	r2 = r2.append({'Model Name':i+1, 'R-squared value':lst[i]}, ignore_index=True)
print(r2)