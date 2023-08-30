import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, explained_variance_score
from datetime import date

def numDate(date):
    sums = []
    months = {
        "January" : 100,
        "February" : 200,
        "March" : 300,
        "April" : 400,
        "May" : 500,
        "June" : 600,
        "July" : 700,
        "August" : 800,
        "September" : 900,
        "October" : 1000,
        "November" : 1100,
        "December" : 1200
    }
    for val in date:
        month, day, year = val.split("-")      
        sum = (int(year) * 10000) + (months[month]) + (int(day))
        sums.append(sum)
    
    return sums

#Loading Data
housing_path = 'Irvine Housing Data - redfin_2023-08-09-14-21-31.csv'
housing_data = pd.read_csv(housing_path)
housing_data = housing_data.drop(["HOA/MONTH", "STATUS", "NEXT OPEN HOUSE START TIME", "NEXT OPEN HOUSE END TIME", "URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)", "SOURCE", "MLS#", "FAVORITE", "INTERESTED", "DAYS ON MARKET", "$/SQUARE FEET", "LOT SIZE"], axis=1)
housing_data = housing_data[(housing_data['CITY'] == "Irvine") & (housing_data['PROPERTY TYPE'] != "Vacant Land") & (housing_data['LOCATION'].notna())]
housing_data["SOLD DATE"] = numDate(housing_data["SOLD DATE"])

#Handeling Missing Values
med_beds = housing_data.BEDS.median()
med_baths = housing_data.BATHS.median()
year, month, day = str(date.today()).split("-") 
today = (int(year)*1000) + (int(month)*100) + int(day)

housing_data = housing_data.fillna({"BEDS": med_beds, "BATHS": med_baths, "SOLD DATE": today})

#Prediction Value
y = housing_data.PRICE

#Prediction Datapoints
model_features = ['BEDS', 'BATHS', 'SQUARE FEET', 'YEAR BUILT', 'PROPERTY TYPE', "LOCATION", "SOLD DATE"]
X = housing_data[model_features]

beds = input("Beds: ")
baths = input("Baths: ")
square_feet = input("Square Feet: ")
year_built = input("Year Built: ")
property_type = input("Property Type: ")
location = input("Location: ")

print(type(beds))

test_vals = {'BEDS' : beds, 'BATHS': baths, 'SQUARE FEET': square_feet, 'YEAR BUILT': year_built, 'PROPERTY TYPE': property_type, "LOCATION": location, "SOLD DATE": 2023118}
test_df = pd.DataFrame(test_vals, index=[0])
X = pd.concat([X, test_df], ignore_index=True)

#Encoding Data
le = LabelEncoder()
X.loc[:, "PROPERTY TYPE"] = le.fit_transform(X["PROPERTY TYPE"])
X.loc[:, "LOCATION"] = le.fit_transform(X["LOCATION"])

test_vals = {
    'BEDS' : X.iloc[-1]["BEDS"], 
    'BATHS': X.iloc[-1]["BATHS"], 
    'SQUARE FEET': X.iloc[-1]["SQUARE FEET"], 
    'YEAR BUILT': X.iloc[-1]["YEAR BUILT"], 
    'PROPERTY TYPE': X.iloc[-1]["PROPERTY TYPE"], 
    "LOCATION": X.iloc[-1]["LOCATION"], 
    "SOLD DATE": X.iloc[-1]["SOLD DATE"]}
pred_in = pd.DataFrame(test_vals, index=[0])
print(pred_in)

X = X.iloc[:-1]


pred_model = RandomForestRegressor()
pred_model.fit(X, y)

prediction = pred_model.predict(pred_in)
print(prediction)