import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os.path
import statsmodels.formula.api as smf

#The goal of this script is to do an analysis of the impact of GEF projects
#using a cross sectional approach.  Here, we look at areas within or proximate
#to GEF projects (`Treatment`), and contrast those to areas that are NOT within or proximate to GEF
#projects, but otherwise similar (`Control`).  This is called a "Quasiexperimental Design", or
#sometimes a "Natural Experiment", and seeks to roughly replicate how clinical 
#trials work, in which you give a pill to one person and not a "twin".
#However, because we don't control where treatments (GEF Intereventions) go,
#we will be limited and have slightly weaker tests than would be
#in a full randomized control trial.

#Open the geojson for the grid - this should be (to start - you may test other grids later)
#a 10km grid across your full country (and, eventually, region).
gridDta = gpd.read_file('data/gridded_southeast_merge.geojson')

#Open the geojson for the location of the GEF projects.
#We'll be using this to establish what grid cells received treatments,
#the distance from a treatment for each grid cell,
#as well as the year in which the treatment was received.
gefDta = gpd.read_file('data/gef_apr5_southasia.geojson')

####
##GOAL: Add another point-based measurement dataset of socioeconomic status.
##Here, you'll want to load in a point-based survey instrument - for example,
##The Living Standards Measurement Survey (LSMS) or Demographic & Health Survey (DHS) 
##both provide latitude and longitude information on the households they survey.
##You can use those latitudes and longitudes to create a new column within your 
##grid - i.e., "Household Income".  This new column can then be leveraged in the cross
##sectional analysis.
##Please note that these datasets frequently require registration, or emailing people
##to request access.  Each country may have their own datasets as well.  IPUMS data
##may be available in some cases, in which case you would have to aggregate census
##polygons to your grid.  It will be up to you to do a deep dive to find the best datasets
##to add to this grid.
####

####
##GOAL: CREATE A VISUALIZATION OF THE DATA.
## See figure 1 in this cfor an example: https://www.mdpi.com/2071-1050/12/8/3225
####

#Get column names and save them to a file for reference
#(Note you can do the same thing for the GEF data if you want or need)
#Here, I'm just doing it for the grid.
with open("summaries/gridVariableNames.txt","wt") as f:
    print(gridDta.columns.tolist(), file=f)


#Now we're going to do our distance calculations.
#Of note, we're using latitude/longitude here ("geographic CRS"),
#which can result in small errors as distance is warped across different latitudes.
#thus...:

####
##GOAL: REPROJECT YOUR DATA TO AN EQUAL-DISTANCE PROJECTION BEFORE RUNNING THE DISTANCE
##CALCULATION - THIS WILL IMPROVE REGIONAL-SCALE ACCURACY, ESPECIALLY IN AREAS FURTHER NORTH 
#OR SOUTH (CONSIDER A GLOBE - LAT/LON CELLS GET DISTORTED MORE AS YOU APPROACH POLES)!
####

##################################################
##################################################
##################################################
##################################################
##ACCOUNTING FOR TEMPORAL DATES OF GEF PROJECTS
##################################################
##################################################
##################################################
##################################################
##Before we calculate distances, we need to ensure that we're only measuring distance to interventions that 
#occured *before* the measurement we're interested in.
#In this example, I'm going to use the variable "stable_lights_2013 - stable_lights_2000" (change in NTL between 2000 and 2013).
#Thus, I only expect GEF projects implemented on or before 2013 to have any possible impact. 
#Note that you will need to filter based on your own outcome variable - while everyone will have the stable lights
#option, the strong expectation is you will find other ancillary surveys (and, at least something like IPUMS!) to do 
#additional analyses.
#Please see timeSeries.py lines ~36 to 100 for details on how this code works.
metaData = pd.read_csv('data/ALL_GEF_Report_2021.02.16.csv')
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
metaData["GEFID"] = metaData["GEFID"].astype(int)
gefDta["GEFID"] = gefDta["GEFID"].astype(int)
metaGeoDta = gefDta.join(metaData.set_index('GEFID'), on="GEFID", rsuffix="meta_")
print("Number of Observations After Join: ", metaGeoDta.shape[0])
metaGeoDta["implementationDate"] = metaGeoDta["Actual Start Date"]
dateOrder = ["Agency Approval Date", "First Disbursement Date", "Expected MTR Date", "Actual MTR Date", "CEO Aprroval Date", "CEO Endorsement Date", "CEO Endorsement Submission Date", "Entry Into Work Program Date", "PIF Approval Date"]

for i, row in metaGeoDta.iterrows():
    if pd.isnull(row["implementationDate"]):
        for columnName in dateOrder:
            if(not pd.isnull(row[columnName])):
                metaGeoDta.at[i,'implementationDate'] = row[columnName]
                break

#Cleanup our type:
metaGeoDta['implementationDate'] = metaGeoDta['implementationDate'].astype('int')

print("Total Observations: ", metaGeoDta.shape[0])
print("Number of Observations without any date:", metaGeoDta['implementationDate'].isnull().sum())
print("Total Observations pre-2013: ", metaGeoDta[metaGeoDta['implementationDate'] < 2013].shape[0])
gefDtaPre2013 = metaGeoDta[metaGeoDta['implementationDate'] < 2013]
##################################################
##################################################
##################################################
##################################################

#Note this distance calculation takes a while, so we're actually going to cache them.
#This will be handled simply - we'll save the results into a folder called
#"cache", and if the file already exists we'll skip this calculation.
if(not os.path.exists("cache/gridWithDistances.geojson")):
    gridDta["GEF_distance"] = gridDta.geometry.apply(lambda g: gefDtaPre2013.distance(g).min())
    gridDta.to_file("cache/gridWithDistances.geojson", driver="GeoJSON")
else:
    print("Using Cached distances.")
    gridDta = gpd.read_file("cache/gridWithDistances.geojson")

#We now know the distance between every grid cell and the closest GEF project.
#This is another plot you will want to dramatically improve!

###GOAL
###Make a much better visualization of the distances!
###Optimally, including a super-imposed visual of GEF projects on top
###of the grid.
###

gridDta.plot(column="GEF_distance")
plt.savefig("visualizations/gridDistances.jpg")

###
#Now we have our distances, so we can start our matching.
#We'll be implementing something called "Propensity Score Matching".
#Here, we first estimate the propensity for every unit (grid cell) to
#have received a GEF project.  If GEF projects were assigned completely randomly,
#every grid cell would have a 0.5 (50%) probability of having received treatment.
#If there is bias (i.e., GEF projects target areas with higher NDVI or forest cover at baseline),
#then we should be able to model that relationship.
#Once we know the propensity for each unit to have been treated, we want to match units
#of similar propensity - i.e., a control unit with a 10% chance of being treated to a place
#where a GEF project was implemented, but only had a 10% chance of a project being there.
#Or, conversely, a control site (no GEF project) with a 90% chance that it WOULD have, with a 
#GEF project cell that also had a 90%.
#The important reason for this is that we are essentially biasing our matches towards varibales
#along which assignment of treatment was not random - i.e., if the GEF intentionally targeted areas
#with higher precipitation, we want to bias our controls in the same way (i.e., it's more important to capture
#fluctuations in precipitaiton in this case, as other variables are already random).

#First, we create a linear model that can estimate the probability that any given grid cell would have received 
#a GEF project.  To do this, we must create a binary value of either 0 or 1, which indicates if a grid cell was treated
#or not.  In this first case, we'll use the distance metric:

#First, we'll create a column labeling all grid cells as controls (0)
gridDta["Intervention"] = 0

#Now, we'll update that column with 1's in cases where a GEF project is within approximately 10km of a grid cell.
#Note we're in geographic projection, so each degree (1.0) is equal to around 111km at the equator.
#So this will actually resolve to about 11km (see my note above about using an equal distance projection - you'll
# need to update this too.)
gridDta.loc[gridDta['GEF_distance'] <= 0.1, "Intervention"] = 1

print("Proportion of Observations Treated: ", gridDta["Intervention"].mean())

#Now we want to create our propensity model.  This can be done using any technique you want, and
#I encourage you to construct the best you can.  All that matters is the R-squared value - i.e.,
#we're just trying to maximize our prediction accuracy.
#You can use machine learning techniques if you want here, i.e. regression trees or SVMs,
#but I'm just doing a simple linear model.  

########
###GOAL: Improve the propensity model!
##Note you want to use everything from your BASELINE
#Period here - i.e., in this example I'm starting the NTL at 2000, so I want
#to match based on data from circa 2000.
#To keep things simple I'm just using a few variables - you'll want to use all of them.
#Note also we're using a logit regression here, which ensures values between 0 and 1, as
#these are probabilities to have received treatment.
########

result = smf.logit(formula="Intervention ~ NDVI_2000 + mean_2m_air_temperature_2000 + treecover2000", data=gridDta).fit()
with open("models/propensityModel.txt","wt") as f:
    print(result.summary(), file=f)

#We're going to programmatically use the results of our linear model to construct our predictions.
#Conveniently, the statsmodels package provides these for us, so all we need to do is save them to
#our dataframe.
gridDta["propensityScore"] = result.predict(gridDta)

#Map our scores - note they'll be pretty random right now,
#as the propensity model isn't very good in this example!
gridDta.plot(column="propensityScore")
plt.savefig("visualizations/propensityScores.jpg")

#Now we want to match each of our treatment cases (where GEF projects were) to a single control case.
#There are other strategies we could use, and I encourage you to explore - i.e., one-to-many matches, 
#in which you have 5, or 10 controls matched to every treatment.  We're also going to be using 
#a simple greedy algorithm, in which each control is matched only once (no replacement), and we just
#iterate over every treatment and select the best control for it.

#Note that there are MANY matching algorithms that will dramatically outperform this, 
#both in terms of speed (this implementation is slow, and really just designed so you can see
# what's happening), as well as ensuring more optimal matches.  Thus...

#########
###STRETCH GOAL: IMPROVE THE MATCHING ALGORITHMS
###Either implement a vectorized version of a greedy algorithm,
###Or identify a better solution.
##Some resources:
#https://www.freecodecamp.org/news/when-to-use-greedy-algorithms/
#https://betterprogramming.pub/greedy-algorithms-79d0ed19aef9
#You could also do a one-to-many match and parallelize, so that you find matches
#simultaneously for all (or many) of the treatments.  As long as you weight your observations
#in the final model, this can be straightforward.
#########

#Caching this to keep things fast.

if(not os.path.exists("cache/gridWithMatches.geojson")):
    #First, we'll create a new column which will be a "0" if it is not included in the matched dataframe,
    #and a "1" if we will be including it.
    gridDta["matchedDataframe"] = 0

    #We'll also save the final match IDs and match quality, as well as the distance
    gridDta["matchID"] = 0
    gridDta["matchQuality"] = 1.0

    #We're also going to make a deep copy of our dataframe so we can remove values during the loop,
    #and limit searches to only control cases.
    controlCopy = gridDta.copy(deep=True)

    #This effectively limits our matches to only control cases, by ensuring that the minimum
    #match will never be to a GEF treated site.
    controlCopy.at[controlCopy["Intervention"] == 1, "propensityScore"] = -999
    
    #for each row, we find the smallest difference between the propensity of that treatment
    #and all controls. 

    for i, row in gridDta.iterrows():
        if(row["Intervention"] == 1):
            #First, we'll set the intervention row as being included in the matched dataframe.
            gridDta.at[i, "matchedDataframe"] = 1
            #Now, we'll search across (remaining) controls to select the best match.
            #Note here we have a special exception to remove any observations that
            #resolved to NA for the propensity score (i.e., if they had a NA measurement)
            #for one of the satellite variables due to clouds).
            if(np.isnan(row["propensityScore"])):
                gridDta.drop([i])
            else:                
                searchArr = abs(row["propensityScore"] - controlCopy["propensityScore"].values)
                searchMin = min(searchArr)
                bestMatchIndex = np.where(searchArr == searchMin)[0][0]
                #Update this in the full dataframe.
                gridDta.at[bestMatchIndex, "matchedDataframe"] = 1
                gridDta.at[bestMatchIndex, "matchQuality"] = searchMin
                gridDta.at[bestMatchIndex, "matchID"] = i
                gridDta.at[i, "matchID"] = bestMatchIndex
                gridDta.at[i, "matchQuality"]  = searchMin
                #Remove this row from the eligible matches
                controlCopy.at[bestMatchIndex, "propensityScore"] = -999
    print("Caching Matches File")
    gridDta.to_file("cache/gridWithMatches.geojson", driver="GeoJSON")
else:
    print("Using Cached matches.")
    gridDta = gpd.read_file("cache/gridWithMatches.geojson")

#Now we have our matched data and can do some actual modeling!  We just need to drop anything that isn't included
#(noting we set matchedDataframe to 1 in all cases where a match was generated).
analysisDta = gridDta[gridDta["matchedDataframe"] == 1]

#This part is fairly straightforward - we just need to build linear models with our intervention
#as well as all of our control variables.
#A few choices we need to make, like:
#1) How should we threshold match qualities?
#2) Should we remove cells within a certain distance of GEF projects from eligibility for analysis (in our controls)?
#(Note #2 can be done BEFORE matching as well. You may want to test both.  For this analysis, I'm not goign to do any thresholding
#making the assumption that the 10km cells are already large enough to account for spatial spillover (which may or may not be true!)
#3) Do we have strong collinearity in our data?  What covariates should we include or not?

#First, let's setup a drop - this is where we remove cases of poor matches (where propensity score differences from the best are still large).
#This is somewhat arbitrary, and contingent on your dataset, but using 0.1 or 0.5 standard deviations from the mean is a common strategy.
#In our case, the mean is 0.513, and the std is 0.499, so I am going to take all match qualities that are better than (0.499 + 0.1*0.499) = 0.5489.
#Note, though, this is a threshold you want to test - decreasing it will result in only cases with higher match qualities, 
#which can result in more defensible models (as you're more likely to have good twins!).  It comes at the cost of your total N.
#If you can have a small threshold + a large N, then you're good to go.
#Generally you don't want to dip below ~100 observations or so, as a rough rule of thumb here.
analysisDta = analysisDta[analysisDta["matchQuality"] <= 0.5489]
print("Observations remaining after dropping matches with large propensity differences: ", analysisDta.shape[0])

#Now we'll make a relatively straighforward linear model, just like we've done before, to see if after matching there
#is an effect we can detect.  However, before we do that we need to define our target variable.
#In this example, I am using a single difference - "stable_lights_2013 - stable_lights_2000".
#However, this is very simple and you should go much farther.  Thus..

#############
###GOAL
###Create a wide range of outcome variables.
###Difference-in-difference approaches are common (i.e., (2012-2008) - (2008-2000) 
###as these can capture differences in trend.
###You may also want to test different starting dates (i.e., dates that are important for some reason, 
###or result in you having more data available.)
#You'll also want to explore using VIIRS in the near term, in addition to constructing your own survey
#based metrics.
#############

#Here, we're doing a simple outcome: "did lights tend to increase more in GEF areas than in comparable areas without GEF projects?"
analysisDta["changeInNTL"] = analysisDta["stable_lights_2013"] - analysisDta["stable_lights_2000"]

#Note you need to include ancillary variables from the first time period as well, to control for other biases.
#This is in addition to your change variables, 
#i.e. - mean_2m_air_temperature_2012 - mean_2m_air_temperature_2000
analysisDta["changeInTemperature"] = analysisDta["mean_2m_air_temperature_2012"] - analysisDta["mean_2m_air_temperature_2000"]
analysisDta["changeInPrecip"] = analysisDta["total_precipitation_2012"] - analysisDta["total_precipitation_2000"]

resultCrossSection = smf.ols(formula="changeInNTL ~ changeInTemperature + mean_2m_air_temperature_2000 + changeInPrecip + total_precipitation_2000 + Intervention", data=analysisDta).fit()
with open("models/crossSectionNTLDiff_2013_2000_IMPLEMENTED.txt","wt") as f:
    print(resultCrossSection.summary(), file=f)


####In the model results, we care about a few things:
#                            OLS Regression Results                            
#==============================================================================
#Dep. Variable:            changeInNTL   R-squared:                       0.106
#Model:                            OLS   Adj. R-squared:                  0.105
#Method:                 Least Squares   F-statistic:                     95.79
#Date:                Fri, 03 Jun 2022   Prob (F-statistic):           1.22e-95
#Time:                        10:16:56   Log-Likelihood:                -57918.
#No. Observations:                4050   AIC:                         1.158e+05
#Df Residuals:                    4044   BIC:                         1.159e+05
#Df Model:                           5                                         
#Covariance Type:            nonrobust                                         
#================================================================================================
#                                   coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------------------------
#Intercept                    -1.188e+07   7.22e+05    -16.452      0.000   -1.33e+07   -1.05e+07
#changeInTemperature          -4.209e+05   2.64e+04    -15.938      0.000   -4.73e+05   -3.69e+05
#mean_2m_air_temperature_2000  4.144e+04   2446.340     16.941      0.000    3.66e+04    4.62e+04
#changeInPrecip               -5.623e+05   3.35e+05     -1.677      0.094   -1.22e+06     9.5e+04
#total_precipitation_2000     -1.154e+04   1.91e+05     -0.060      0.952   -3.86e+05    3.63e+05
#Intervention                  2772.2792   1.24e+04      0.224      0.823   -2.15e+04     2.7e+04
#==============================================================================
#Omnibus:                     3885.403   Durbin-Watson:                   0.879
#Prob(Omnibus):                  0.000   Jarque-Bera (JB):           170655.000
#Skew:                           4.685   Prob(JB):                         0.00
#Kurtosis:                      33.389   Cond. No.                     3.48e+04
#==============================================================================
#Notes:
#[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#[2] The condition number is large, 3.48e+04. This might indicate that there are
#strong multicollinearity or other numerical problems.

#First, the R-squared does matter here - higher values indicate that we are accounting for more things that can
#drive fluctuations in NTL.  It will likely be on the lower side for these models, as socoeconomic trends
#are hard to predict!  Information like population at baseline can be very helpful here.

#Second, we continue to care about multicollinearity - it remains bad, just like in the time series cases.

#Third, we can interpret the coef value on the Intervention (2772.2792) as an indication that GEF projects
#tend to increase nighttime lights relative to other places.  However, this is not significant,
#as indicated by the P>|t| of 0.823 - i.e., this effect could really be either positive or negative.

#Once you have a set of global models you're happy with, we'll do the next round of models which show distance-decay of these
#effects - i.e., as you get farther away from GEF projects, how the overall effect changes.  
#To do this, we simply edit our "treatment" variable and re-run the above analysis for multiple distance bands.
#For example, if I want to know how big of an effect hte GEF has on locations up to 10km away from GEF projects,
#but NOT the actual implementation locations, I could:

gridDta.loc[gridDta['GEF_distance'] <= 0.15, "Intervention"] = 1
gridDta.loc[gridDta['GEF_distance'] == 0, "Intervention"] = 0
print("Proportion of Observations Treated (up to 0.~22km away): ", gridDta["Intervention"].mean())

resultDistBand = smf.logit(formula="Intervention ~ NDVI_2000 + mean_2m_air_temperature_2000 + treecover2000", data=gridDta).fit()
with open("models/propensityModelDistanceBandA.txt","wt") as f:
    print(resultDistBand.summary(), file=f)

gridDta["propensityScore"] = resultDistBand.predict(gridDta)

#Same exact matching 
if(not os.path.exists("cache/gridWithMatches_DistanceBand.geojson")):
    gridDta["matchedDataframe"] = 0
    gridDta["matchID"] = 0
    gridDta["matchQuality"] = 1.0
    controlCopy = gridDta.copy(deep=True)
    controlCopy.at[controlCopy["Intervention"] == 1, "propensityScore"] = -999
    for i, row in gridDta.iterrows():
        if(row["Intervention"] == 1):
            gridDta.at[i, "matchedDataframe"] = 1
            if(np.isnan(row["propensityScore"])):
                gridDta.drop([i])
            else:                
                searchArr = abs(row["propensityScore"] - controlCopy["propensityScore"].values)
                searchMin = min(searchArr)
                bestMatchIndex = np.where(searchArr == searchMin)[0][0]
                gridDta.at[bestMatchIndex, "matchedDataframe"] = 1
                gridDta.at[bestMatchIndex, "matchQuality"] = searchMin
                gridDta.at[bestMatchIndex, "matchID"] = i
                gridDta.at[i, "matchID"] = bestMatchIndex
                gridDta.at[i, "matchQuality"]  = searchMin
                controlCopy.at[bestMatchIndex, "propensityScore"] = -999
    print("Caching Matches File (Distance Band)")
    gridDta.to_file("cache/gridWithMatches_DistanceBand.geojson", driver="GeoJSON")
else:
    print("Using Cached matches (Distance Band).")
    gridDta = gpd.read_file("cache/gridWithMatches_DistanceBand.geojson")

#and, same modeling!  I have an arbitrary matchQuality threshold here.
#You can make it dynamic for different distance bands, or the same for all.
#Here I'm just keeping 0.5489, the same as our global model,
#to make it more comparable.
analysisDta = gridDta[gridDta["matchedDataframe"] == 1]
analysisDta = analysisDta[analysisDta["matchQuality"] <= 0.5489]
print("Observations remaining after dropping matches with large propensity differences (Distance Band Model): ", analysisDta.shape[0])

#new utcome: "did lights tend to increase more in areas proximate to GEF areas than in comparable areas without GEF projects?"
analysisDta["changeInNTL"] = analysisDta["stable_lights_2013"] - analysisDta["stable_lights_2000"]

analysisDta["changeInTemperature"] = analysisDta["mean_2m_air_temperature_2012"] - analysisDta["mean_2m_air_temperature_2000"]
analysisDta["changeInPrecip"] = analysisDta["total_precipitation_2012"] - analysisDta["total_precipitation_2000"]

resultCrossSectionDistanceBand = smf.ols(formula="changeInNTL ~ changeInTemperature + mean_2m_air_temperature_2000 + changeInPrecip + total_precipitation_2000 + Intervention", data=analysisDta).fit()
with open("models/crossSectionNTLDiff_2013_2000_DISTANCEBAND0_15_IMPLEMENTED.txt","wt") as f:
    print(resultCrossSectionDistanceBand.summary(), file=f)

#This gives the following results:
                                #OLS Regression Results                            
#==============================================================================
#Dep. Variable:            changeInNTL   R-squared:                       0.111
#Model:                            OLS   Adj. R-squared:                  0.110
#Method:                 Least Squares   F-statistic:                     70.56
#Date:                Fri, 03 Jun 2022   Prob (F-statistic):           9.93e-70
#Time:                        10:35:29   Log-Likelihood:                -40120.
#No. Observations:                2820   AIC:                         8.025e+04
#Df Residuals:                    2814   BIC:                         8.029e+04
#Df Model:                           5                                         
#Covariance Type:            nonrobust                                         
#================================================================================================
#                                   coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------------------------
#Intercept                    -1.157e+07   7.99e+05    -14.492      0.000   -1.31e+07      -1e+07
#changeInTemperature          -3.995e+05      3e+04    -13.297      0.000   -4.58e+05   -3.41e+05
#mean_2m_air_temperature_2000  4.032e+04   2707.206     14.894      0.000     3.5e+04    4.56e+04
#changeInPrecip               -5.086e+05   3.81e+05     -1.333      0.183   -1.26e+06    2.39e+05
#total_precipitation_2000      -6.48e+04    2.1e+05     -0.309      0.757   -4.76e+05    3.46e+05
#Intervention                  2.589e+04   1.38e+04      1.879      0.060   -1126.854    5.29e+04
###==============================================================================
#Omnibus:                     2849.677   Durbin-Watson:                   1.080
#Prob(Omnibus):                  0.000   Jarque-Bera (JB):           153689.576
#Skew:                           4.979   Prob(JB):                         0.00
#Kurtosis:                      37.768   Cond. No.                     3.45e+04
#==============================================================================
#Notes:
#[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#[2] The condition number is large, 3.45e+04. This might indicate that there are
#strong multicollinearity or other numerical problems.

#As you can see in this example, here the intervention effect is positive and significant at a 0.1 level.
#The interpretation of this would be that GEF projects don't have an impact directly where they're implemented,
#but they do in neighboring areas.  This could make sense (noting these models are incomplete!!!!), as the GEF
#frequently works in uninhabited or sparsely populated areas, but their activities benefit nearby communities.

#Our goal will be to create a figure similar to Figure 3 in:
#https://www.mdpi.com/2071-1050/12/8/3225
#Figure 2 helps explain visually how we're constructing the distance bands.
#You will ultimately likely want to wrap all of this in a loop.