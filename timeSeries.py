import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import sys
import statsmodels.formula.api as smf

#The goal of this script is to do an analysis of the impact of GEF projects
#using a natural breaks approach.  Here, we look at the geographic region 
#of the GEF project *before* the intervention, establish the trend, and then
#look at the same region *after* the intervention to determine if the trend
#was influenced by the activities.

#Open the geojson
gDta = gpd.read_file('data/gef_apr5_southasia.geojson')

#Get column names and save them to a file for reference
with open("summaries/geojsonVariableNames.txt","wt") as f:
    for var in gDta.columns.tolist():
        print(var, file=f)

####
#GOAL: IMPROVE THIS VISUALIZATION TO BE BOOK-READY.
#WE WILL WANT TO CREATE A MAP OF GEF PROJECTS IN 2000, AS NOTED HERE.
#THIS IS A QUICK AND DIRTY IMPLEMENTATION TO SHOW THAT THE GEOSPATIAL DATA IS LOADING CORRECTLY.
#IMPROVEMENTS TO THIS VISUALIZATION ARE A GOOD TASK FOR YOU TO WORK ON.
####
fig, ax = plt.subplots(1, 1)
gDta.plot(column="treecover2000", ax=ax, legend=False, cmap="RdYlGn")
plt.savefig('visualizations/treecover2000.jpg')

#Key descriptives of the data
print("Number of Observations Initially Loaded: ", gDta.shape[0])


#Next, we need to join the GEF metadata.  This is in order to establish our best estimate as to when activities on the ground actually began.
#(The metadata has the date information)
metaData = pd.read_csv('data/ALL_GEF_Report_2021.02.16.csv')

#In this file, we see we have the following date columns, from which we want to 
#construct a single variable which is our best guess as to when on-the-ground activities
#commenced, and recognizing that some programs may be missing some (or even all) dates.
#PIF Approval Date
#Entry Into Work Program Date
#CEO Endorsement Submission Date
#CEO Endorsement Date
#CEO Aprroval Date
#Drop Date
#Cancellation Date
#Agency Approval Date
#Actual Start Date
#First Disbursement Date
#Approval Rejected Date
#Expected MTR Date
#Actual MTR Date
#Expected Completion Date
#Actual Completion Date
#Actual Terminal Eval Review Date
#Financial Closure Date
#New Deadline Date

#In order to make an estimate, we'll take the "Actual Start Date" in all cases it is available.
#If it is not available, we'll take (in order of what we want) agency approval, first disbursement, expected MTR, actual MTR.
#If none of those are available, we'll take any date *except* expected completion, actual completion, terminal eval, financial closure, or new deadline.

#First we need to convert everything over to the year (i.e., the timestep our satellite data is at)
metaData['Actual Start Date'] = metaData['Actual Start Date'].str[-4:]
metaData['PIF Approval Date'] = metaData['PIF Approval Date'].str[-4:]
metaData['Entry Into Work Program Date'] = metaData['Entry Into Work Program Date'].str[-4:]
metaData['CEO Endorsement Submission Date'] = metaData['CEO Endorsement Submission Date'].str[-4:]
metaData['CEO Endorsement Date'] = metaData['CEO Endorsement Date'].str[-4:]
metaData['CEO Aprroval Date'] = metaData['CEO Aprroval Date'].str[-4:]
metaData['Drop Date'] = metaData['Drop Date'].str[-4:]
metaData['Agency Approval Date'] = metaData['Agency Approval Date'].str[-4:]
metaData['Actual Start Date'] = metaData['Actual Start Date'].str[-4:]
metaData['First Disbursement Date'] = metaData['First Disbursement Date'].str[-4:]
metaData['Approval Rejected Date'] = metaData['Approval Rejected Date'].str[-4:]
metaData['Expected MTR Date'] = metaData['Expected MTR Date'].str[-4:]
metaData['Actual MTR Date'] = metaData['Actual MTR Date'].str[-4:]
metaData['Expected Completion Date'] = metaData['Expected Completion Date'].str[-4:]
metaData['Actual Completion Date'] = metaData['Actual Completion Date'].str[-4:]
metaData['Actual Terminal Eval Review Date'] = metaData['Actual Terminal Eval Review Date'].str[-4:]
metaData['Financial Closure Date'] = metaData['Financial Closure Date'].str[-4:]
metaData['New Deadline Date'] = metaData['New Deadline Date'].str[-4:]

#Need the same type of variable for the join
metaData["GEFID"] = metaData["GEFID"].astype(int)
gDta["GEFID"] = gDta["GEFID"].astype(int)

#Pandas behavior is to use the index as the default join key.  We need to change that to the GEFID, hence the .set_index().
metaGeoDta = gDta.join(metaData.set_index('GEFID'), on="GEFID", rsuffix="meta_")

#Now we have a bunch of date information - let's identify the names of the columns that were imported.
with open("summaries/metaGeoJsonVariableNames.txt","wt") as f:
    for var in metaGeoDta.columns.tolist():
        print(var, file=f)

print("Number of Observations After Join: ", metaGeoDta.shape[0])

print("")
print("Begin date conversions and selections.")

print("Number of Observations that do not have an Actual Start Date (post-conversion sanity check):", metaGeoDta['Actual Start Date'].isnull().sum())

#We'll just copy actual start date over to a new column:
metaGeoDta["implementationDate"] = metaGeoDta["Actual Start Date"]

#Now, we'll fill in the blanks with the first round of dates (agency approval, first disbursement, expected MTR, actual MTR):
#Note this is a very sub-optimal implementation, and mostly written as a loop to illustrate exactly 
#what's going on.  This could be upgraded to use vector operations to make it more efficient.
dateOrder = ["Agency Approval Date", "First Disbursement Date", "Expected MTR Date", "Actual MTR Date", "CEO Aprroval Date", "CEO Endorsement Date", "CEO Endorsement Submission Date", "Entry Into Work Program Date", "PIF Approval Date"]

for i, row in metaGeoDta.iterrows():
    if pd.isnull(row["implementationDate"]):
        for columnName in dateOrder:
            if(not pd.isnull(row[columnName])):
                metaGeoDta.at[i,'implementationDate'] = row[columnName]
                break

print("Total Observations: ", metaGeoDta.shape[0])
print("Number of Observations without any date:", metaGeoDta['implementationDate'].isnull().sum())
print("Total Observations post-2001: ", metaGeoDta[metaGeoDta['implementationDate'] > '2000'].shape[0])

#Subset the dataframe to only include years after 2000 (first year of deforestation / satellite data available)
dtaPost2000 = metaGeoDta[metaGeoDta['implementationDate'] > '2000']

print("")
print("Begin time series model setup")
#Now we need to create a time series model to establish if the GEF had a significant impact on deforestation.
#We'll repeat the same thing for nighttime lights later, but we don't expect much of an impact on that one 
#(our record is only through 2013, and many implementations are in low-light areas such a national parks where we don't anticipate much light anyway.)

#To establish the impact of GEF projects, we are going to implement the following model:
#total_deforestation_Y = B0 + B1 * total_deforestation_Y-1 + B2 * min_temperature_Y + B3 * pdsi_Y + B4 * T
#which can be interepreted as modeling total deforestation on the basis of a slope parameter (B0), a parameter (B1) * the total deforestation from last year, and then two parameters - B2 and B3, representing the numbers we'll multiply by this year's minimum temperature and PDSI. Finally - and most importantly for us - we'll have the fourth parameter (B4), which we will multiply by a binary value of 1 or 0 that indicates if a GEF project was implemented in the past at a site.

#Note this is a dummy model, and your job will be to do a TON of models exploring different combinations of factors.  You'll want to include more ancillary variables, test different lags, explore when the binary "1" should be set to 1 (i.e., only in the 5 years after a GEF project started? 3 years?), and even multiple lag terms (i.e., lagging pdsi in addition to total deforestation).  You'll also want to explore grouping by GEF project (i.e., taking average values across all sub-geometries).

#The first thing we need to do is transform our data from a "Wide" format to a "Long" format. In a wide format, data looks like this:
#GEF_ID | NDVI_2000 | NDVI_2001 | ... | NDVI_2021 | implementationDate | tmmn_2000 | tmmmn_2001 | ... | tmmmn_2021 | pdsi_2000 | pdsi_2001 | ... | pdsi_2021 |
#9060   |  0.15     |   0.17    | ... |   0.17    |           2010     |  12       |   10       | ... |     5      |  0.15     |   0.17    | ... |   0.17    |

#We need to convert this to a long format, which looks like this:
#GEF_ID | YEAR | NDVI | TMN | PDSI | IMPLEMENTED
#9060   | 2000 | 0.15 | 12  | 0.15 | 0
#9060   | 2001 | 0.17 | 10  | 0.17 | 0
#....
#9060   | 2021 | 0.17 | 5   | 0.17 | 1

#Where the implemented column is a "1" in years post-implementationDate, and "0" otherwise.
#This can be done - mostly - using either .melt or .wide_to_long in pandas.
#To get there we need to do one quick thing, which is ensure that every row of our
#wide dataframe has a unique ID - i.e., the GEF ID won't be unique, because the individual shapes
#may repeat the same GEF ID.  To do this, I used QGIS to add a new column to the geoJson - UNIQUEID - which
#concactenated "GEFID" || '_' || "rownumber".  But you can do it in python or however you want.  Just make sure
#to keep track of it, as you'll need your geoJson to have the field so you can join results to a map later!
longMetaGeoDta = pd.wide_to_long(dtaPost2000, stubnames=["NDVI", "pdsi", "tmmn"], i="UNIQUEID", j="YEAR", sep="_")

#Now we need to clean it up - note this line will change as you add more ancillary variables. 
longMetaGeoDta = longMetaGeoDta[["NDVI", "pdsi", "tmmn"]]

#Now we need to lag our NDVI outcome (where NDVI is our metric of vegetation):
#This post provides a decent/quick explanation of this: https://towardsdatascience.com/timeseries-data-munging-lagging-variables-that-are-distributed-across-multiple-groups-86e0a038460c
longMetaGeoDta["NDVI_LAG"] = longMetaGeoDta["NDVI"].shift(1)

#Now we need to drop our rows with NA values. 
#This will be caused by two things:
#First, the lag will always generate one NaN value for our NDVI at the start of the time series (2000), because we don't have the 1999 value to add to it.
#Second, some measurements are truly missing (i.e., 9781 107 - pdsi).
longMetaGeoDta = longMetaGeoDta.dropna()

#Next we convert everything over to a numeric type, so we can use it with statsmodels.
longMetaGeoDta["NDVI"] = pd.to_numeric(longMetaGeoDta["NDVI"])
longMetaGeoDta["pdsi"] = pd.to_numeric(longMetaGeoDta["pdsi"])
longMetaGeoDta["tmmn"] = pd.to_numeric(longMetaGeoDta["tmmn"])
longMetaGeoDta["NDVI_LAG"] = pd.to_numeric(longMetaGeoDta["NDVI_LAG"])

#Confirm our dtypes:
#longMetaGeoDta.info()

print("")
print("Begin model")
#Now, we're finally ready to model!  We're going to keep this as a very simple example,
#but I also encourage you to explore other models - see https://www.statsmodels.org/stable/tsa.html, https://pdfs.semanticscholar.org/7c96/660127fefabe926214abaa80b298066af60d.pdf, https://towardsdev.com/time-series-with-statsmodels-basic-concepts-needed-for-forecasting-1-af058aaaea0e for some ideas.



#For now, though, we'll use OLS to fit our first model - just NDVI vs. NDVI LAG:
result = smf.ols(formula="NDVI ~ NDVI_LAG", data=longMetaGeoDta).fit()
with open("models/m1_onlyLag.txt","wt") as f:
    print(result.summary(), file=f)

#Open up the file t1_onlyLag.txt, and you'll see a nice table that looks like this:
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                   NDVI   R-squared:                       0.844
# Model:                            OLS   Adj. R-squared:                  0.844
# Method:                 Least Squares   F-statistic:                 3.405e+04
# Date:                Thu, 14 Apr 2022   Prob (F-statistic):               0.00
# Time:                        13:34:19   Log-Likelihood:                 10559.
# No. Observations:                6299   AIC:                        -2.111e+04
# Df Residuals:                    6297   BIC:                        -2.110e+04
# Df Model:                           1                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      0.0507      0.003     16.223      0.000       0.045       0.057
# NDVI_LAG       0.9175      0.005    184.535      0.000       0.908       0.927
# ==============================================================================
# Omnibus:                      977.214   Durbin-Watson:                   2.682
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             9219.784
# Skew:                          -0.447   Prob(JB):                         0.00
# Kurtosis:                       8.859   Cond. No.                         12.1
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

#For our purposes, we care most about:
#(1) R-squared - this gives us an idea of how well our model predicts the outcome.  In this case, about 84% of our variance is captured based on just last year's NDVI.
#which makes sense - i.e., last year and this year's NDVI are very similar (...84%).
#(2) If we wanted to predict NDVI for 2001, we would use the equation NDVI_2001 = 0.0507 + 0.9175 * NDVI_2000
#(3) The p-values aren't too important just yet, but they indicate that it is highly unlikely that the correlation between NDVI_LAG and NDVI should *actually* be negative - i.e., our model predicts that it's 0.9175 * NDVI_LAG.  The p-value of 0.000 indicates that there is a less-than .1% chance that the correct answer is ACTUALLY a negative value * NDVI_LAG.  We'll care about this more in later models.

#Now, let's add in our PDSI and TMIN:
resultB = smf.ols(formula="NDVI ~ NDVI_LAG + pdsi + tmmn", data=longMetaGeoDta).fit()
with open("models/m2_lag_pdsi_tmmn.txt","wt") as f:
    print(resultB.summary(), file=f)
#If you open up this results file, you'll see the predicted effects of PDSI and TMIN on NDVI, controlling for the lagged NDVI trend.
#Note this model shows a new error about multicollinearity - this means you have at least two variables that are highly correlated.  You may want to drop one variable (in this case, either pdsi or tmmn) to improve the capability of the model to ascribe variance to the correct variables.  

#Now, we want to do our first-order GEF impacts model.  To do this, we need to add a new column to our data frame, which will resolve to a "1" if it is after the year of GEF implementation,
#and "0" if it is before the GEF implementation. 
#This will be a new column called "IMPLEMENTED" (1 = after, 0 = before).    We'll use the following code to do this (noting again loops are not an effective strategy, and this could be vectorized, but for the sake of interpretability we'll keep it looped.).  Note this is where you may want to adjust the length of the lag - i.e., do we only count years within 5 
# years of a GEF project implementation?  The implementation here explicitly assumes that after a GEF project occurs, the effects of that project will be felt for "all time" (i.e.
# until the end of our dataset). )              
longMetaGeoDta["IMPLEMENTED"] = 0  

for i, row in longMetaGeoDta.iterrows():
    year = i[1]
    GEF_ID = int(i[0].split("_")[0])
    implementationDate = metaGeoDta[metaGeoDta["GEFID"] == GEF_ID]['implementationDate'].iloc[0]
    if(int(year) >= int(implementationDate)):
        longMetaGeoDta.at[i, "IMPLEMENTED"] = 1

resultC = smf.ols(formula="NDVI ~ NDVI_LAG + pdsi + tmmn + IMPLEMENTED", data=longMetaGeoDta).fit()
with open("models/m3_lag_pdsi_tmmn_IMPLEMENTED.txt","wt") as f:
    print(resultC.summary(), file=f)

#This results in:
#                                 OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                   NDVI   R-squared:                       0.848
# Model:                            OLS   Adj. R-squared:                  0.848
# Method:                 Least Squares   F-statistic:                     8789.
# Date:                Thu, 14 Apr 2022   Prob (F-statistic):               0.00
# Time:                        13:57:47   Log-Likelihood:                 10645.
# No. Observations:                6299   AIC:                        -2.128e+04
# Df Residuals:                    6294   BIC:                        -2.125e+04
# Df Model:                           4                                         
# Covariance Type:            nonrobust                                         
# ===============================================================================
#                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------
# Intercept       0.0822      0.006     13.834      0.000       0.071       0.094
# NDVI_LAG        0.9081      0.005    170.235      0.000       0.898       0.919
# pdsi         2.272e-05   1.99e-06     11.404      0.000    1.88e-05    2.66e-05
# tmmn           -0.0001   1.91e-05     -6.724      0.000      -0.000   -9.11e-05
# IMPLEMENTED     0.0042      0.001      2.978      0.003       0.001       0.007
# ==============================================================================
# Omnibus:                     1103.706   Durbin-Watson:                   2.691
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):            10184.682
# Skew:                          -0.566   Prob(JB):                         0.00
# Kurtosis:                       9.126   Cond. No.                     3.90e+03
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# [2] The condition number is large, 3.9e+03. This might indicate that there are
# strong multicollinearity or other numerical problems.

#Where the thing we are most interested in is the positive (0.0042) coefficient on IMPLEMENTED, coupled with the P>t of .003.
#This can be interpreted as saying that the model predicts that the NDVI will increase by 0.0042 if a GEF project has ever been implemented
#at the location, as contrasted to by 0 if no GEF project has been implemented yet.  The p-value of .003 indicates there isa .3% chance that the
#correct answer is actually a negative coefficient (i.e., it has a negative impact) - a very low probability.  When you hear "significant at the 5% level", that
#means there is a 5% chance that the correct answer is actually a negative value.  So our evidence is much better than that!

#From here, we would want to:
#1) Map the error, by applying our equation to the data and then comparing it to the actual NDVI values - i.e., error for each year in a series of maps, or average error across all maps.
#2) Ensure we don't have multicollinearity (or, at least minimize it) - in this case, probably by dropping tmmn.  You can explore multicollinearity by creating scatterplots to see what's correlated.
#3) Try different modeling approaches, as noted in the comments above - see if the results you observe are robust.
#4) Re-run this for nighttime light data, noting you'll have fewer observations, to see if there is any observable impact on NTL.  Note you'll need to explore different model approaches etc., just like NDVI.
#5) start slotting your findings into your chapter template!

#That's it for this part - the next script will explore the socioeconomic impacts using a gridded model!