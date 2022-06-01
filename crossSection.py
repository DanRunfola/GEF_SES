import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import sys
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
gridDta["propensityScore"] = result.fittedvalues

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
#both in terms of speed (this implementation is wicked slow, and really just designed so you can see
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

#Caching this, as it takes forever (~hours).

if(not os.path.exists("cache/gridWithMatches.geojson")):
    #First, we'll create a new column which will be a "0" if it is not included in the matched dataframe,
    #and a "1" if we will be including it.
    gridDta["matchedDataframe"] = 0

    #We'll also save the final match IDs
    gridDta["matchID"] = 0

    for i, row in gridDta.iterrows():
        if(row["Intervention"] == 1):
            #First, we'll set the intervention row as being included in the matched dataframe.
            gridDta.at[i, "matchedDataframe"] = 1
            #Now, we'll search across (remaining) controls to select the best match.
            treatmentScore = row["propensityScore"]
            bestMatchScore = 999
            bestMatchID = 0
            for j, controlRow in gridDta.iterrows():
                if(controlRow["Intervention"] == 0):
                    if(controlRow["matchedDataframe"] == 0):
                        matchQuality = (treatmentScore - controlRow["propensityScore"])
                        if(bestMatchScore > matchQuality):
                            bestMatchScore = matchQuality
                            bestMatchID = j
            #Save the match ID into the main dataframe.
            gridDta.at[i, "matchID"] = bestMatchID
            #Set the match as included
            gridDta.at[j, "matchedDataframe"] = 1
    
    gridDta.to_file("cache/gridWithMatches.geojson", driver="GeoJSON")

else:
    print("Using Cached matches.")
    gridDta = gpd.read_file("cache/gridWithMatches.geojson")