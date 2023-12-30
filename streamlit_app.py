import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 

st.title(" 🏠House Prediction Tool")
st.subheader("A tool to provide the estimation of the house price based on your input to the number of features.")

# Loads the House Price Dataset
st.header("The Original Dataset")
st.text("The initial dataset is based off of Kaggle which is the Ames Housing Dataset.") 
df = pd.read_csv('train.csv')
st.write(df.head())

# Engineer
st.header("Feature Engineering")
to_drop = ['Id', 'LotFrontage', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'Foundation', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenAbvGr', 'KitchenQual', 'Fireplaces', 'Fence',
       'TotRmsAbvGrd', 'Functional', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
       'GarageArea', 'GarageType', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'];
df.drop(columns=to_drop, inplace=True)

#Converting the LotArea from sqfeet to sqmeter 
df['LotArea'] = df['LotArea'].div(10.764).round(2)
df.head()

#Encoding zoning features 
df.loc[df['MSZoning'] == 'A', 'MSZoning'] = 10
df.loc[df['MSZoning'] == 'C', 'MSZoning'] = 20
df.loc[df['MSZoning'] == 'C (all)', 'MSZoning'] = 20
df.loc[df['MSZoning'] == 'FV', 'MSZoning'] = 30
df.loc[df['MSZoning'] == 'I', 'MSZoning'] = 40
df.loc[df['MSZoning'] == 'RH', 'MSZoning'] = 50
df.loc[df['MSZoning'] == 'RL', 'MSZoning'] = 60
df.loc[df['MSZoning'] == 'RP', 'MSZoning'] = 70
df.loc[df['MSZoning'] == 'RM', 'MSZoning'] = 80

# encoding extercond
df.loc[df['ExterCond'] == 'Ex', 'ExterCond'] = 5
df.loc[df['ExterCond'] == 'Gd', 'ExterCond'] = 4
df.loc[df['ExterCond'] == 'TA', 'ExterCond'] = 3
df.loc[df['ExterCond'] == 'Fa', 'ExterCond'] = 2
df.loc[df['ExterCond'] == 'Po', 'ExterCond'] = 1

# encoding basement
df.loc[df['BsmtQual'] == 'Ex', 'BsmtQual'] = 1
df.loc[df['BsmtQual'] == 'Gd', 'BsmtQual'] = 1
df.loc[df['BsmtQual'] == 'TA', 'BsmtQual'] = 1
df.loc[df['BsmtQual'] == 'Fa', 'BsmtQual'] = 1
df.loc[df['BsmtQual'] == 'Po', 'BsmtQual'] = 1
df.loc[df['BsmtQual'] == 'NA', 'BsmtQual'] = 0

# encoding garage
df.loc[df['GarageQual'] == 'Ex', 'GarageQual'] = 1
df.loc[df['GarageQual'] == 'Gd', 'GarageQual'] = 1
df.loc[df['GarageQual'] == 'TA', 'GarageQual'] = 1
df.loc[df['GarageQual'] == 'Fa', 'GarageQual'] = 1
df.loc[df['GarageQual'] == 'Po', 'GarageQual'] = 1
df.loc[df['GarageQual'] == 'NA', 'GarageQual'] = 0

# encoding pool
df.loc[df['GarageQual'] == 'Ex', 'GarageQual'] = 1
df.loc[df['GarageQual'] == 'Gd', 'GarageQual'] = 1
df.loc[df['GarageQual'] == 'TA', 'GarageQual'] = 1
df.loc[df['GarageQual'] == 'Fa', 'GarageQual'] = 1
df.loc[df['GarageQual'] == 'NA', 'GarageQual'] = 0

# filling NaN with 0
df = df.fillna(0)
df.loc[df['PoolQC'] == 'Ex', 'PoolQC'] = 1
df.loc[df['PoolQC'] == 'Gd', 'PoolQC'] = 1
df.loc[df['PoolQC'] == 'TA', 'PoolQC'] = 1
df.loc[df['PoolQC'] == 'Fa', 'PoolQC'] = 1

df["Bathroom"] = df["FullBath"] + df["HalfBath"]

#Renaming the column data 
df.rename(columns = {'MSSubClass':'HouseType'}, inplace = True)
df.rename(columns = {'MSZoning':'ResidentialZone'}, inplace = True)
df.rename(columns = {'LotArea':'AreaInSqm'}, inplace = True)
df.rename(columns = {'BsmtQual':'Basement'}, inplace = True)    
df.rename(columns = {'BedroomAbvGr':'Bedroom'}, inplace = True)
df.rename(columns = {'GarageQual':'Garage'}, inplace = True)
df.rename(columns = {'PoolQC':'Pool'}, inplace = True)
df.rename(columns = {'SalePrice':'SalePriceUSD'}, inplace = True)

# run label encoder on neighborhood
from sklearn import preprocessing
labelEncoder = preprocessing.LabelEncoder()

df['Neighborhood'] = labelEncoder.fit_transform(df['Neighborhood'])
st.write(df.head())

y = df['SalePriceUSD']

features = ['HouseType', 'ResidentialZone', 'AreaInSqm', 'Neighborhood', 'ExterCond', 'Basement', 'FullBath', 'HalfBath', 'Bedroom', 'Garage', 'Pool', 'Bathroom']
x = df.loc[:, features].values
y = df.loc[:, ['SalePriceUSD']].values

#Train/Test split 
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2, random_state = 2)


st.markdown("* House Type")
st.text("20  1-STORY 1946 & NEWER ALL STYLES") 
st.text("30  1-STORY 1945 & OLDER")  
st.text("40  1-STORY W/FINISHED ATTIC ALL AGES")
st.text("45  1-1/2 STORY - UNFINISHED ALL AGES")
st.text("50  1-1/2 STORY FINISHED ALL AGES") 
st.text("60  2-STORY 1946 & NEWER") 
st.text("70  2-STORY 1945 & OLDER ")
st.text("75  2-1/2 STORY ALL AGES")
st.text("80  SPLIT OR MULTI-LEVEL") 
st.text("85  SPLIT FOYER") 
st.text("90  DUPLEX - ALL STYLES AND AGES")
st.text(" 120  1-STORY PUD (Planned Unit Development) - 1946 & NEWER ")
st.text(" 150  1-1/2 STORY PUD - ALL AGES") 
st.text(" 160  2-STORY PUD - 1946 & NEWER") 
st.text(" 180  PUD - MULTILEVEL - INCL SPLIT LEV/FOYER")
st.text(" 190  2 FAMILY CONVERSION - ALL STYLES AND AGES ")


st.markdown("* Residential Zone")
st.text("10 - Agriculture") 
st.text("20 - Commercial") 
st.text("30 - Floating Village Residential")
st.text("40 - Industrial ")
st.text("50 - Residential High Density") 
st.text("60 - Residential Low Density ") 
st.text("70 - Residential Low Density Park ")
st.text("80 - Residential Medium Density ")

st.markdown("* Only Apply to Exterior Condition ")
st.text("5 - Excellent") 
st.text("4 - Good") 
st.text("3 - Average/Typical")
st.text("2 - Fair to Average")
st.text("1 - Poor")

# Sidebar to specify Input Parameters
st.sidebar.header('Provide your input')

#Error handling 
def userInput():
    try: 
        HouseType = st.sidebar.selectbox("House Type", options = ['20', '30', '40', '45', '50', '60', '70', '75', '80', '85', '90', '120', '150', '160', '180', '190'])
        ResidentialZone = st.sidebar.selectbox("Residential Zone", options = ['10', '20', '30', '40', '50', '60', '70', '80'])
        AreaInSqm = st.sidebar.text_input("Area in Sqm",value = '0')
        Neighborhood = st.sidebar.slider("Neighborhood Zone", min_value = 0, max_value = 24)
        ExterCond = st.sidebar.selectbox("Exterior Condition?", options = ['1', '2', '3', '4', '5'])
        Basement = st.sidebar.selectbox("Basement: Select 1 if there exist a Basement, 0 for None.", options = ['0', '1'])
        FullBath = st.sidebar.slider("Number of FullBath ",min_value = 0, max_value = 3)
        HalfBath = st.sidebar.slider("Number of HalfBath ", min_value = 0, max_value = 2)
        Bedroom = st.sidebar.slider("Number of Bedroom ", min_value = 0, max_value = 8)
        Garage = st.sidebar.selectbox("Garage Condition: Select 1 if there exist a Garage, 0 for None.", options = ['0', '1'])
        Pool = st.sidebar.selectbox("Pool Condition: Select 1 if there exist a Pool, 0 for None.", options = ['0', '1'])   
        Bathroom = st.sidebar.slider("Total number of Bathroom ", min_value = 0, max_value = 5)
        data = {'HouseType': HouseType,
                'ResidentialZone': ResidentialZone,
                'AreaInSqm': AreaInSqm,
                'Neighborhood': Neighborhood,
                'ExterCond': ExterCond,
                'Basement': Basement,
                'FullBath': FullBath,
                'HalfBath': HalfBath,
                'Bedroom': Bedroom,
                'Garage': Garage,
                'Pool': Pool,
                'Bathroom': Bathroom}
        userFeatures = pd.DataFrame(data, index=[0])
        return userFeatures
    except ValueError as e: 
        st.error(f"Error: {e}")
        return None 

df_input = userInput()

if df_input is not None:
    # Build Regression Model
    random_forest = RandomForestRegressor(n_estimators=1000, max_depth=15, max_leaf_nodes=45)

    # Scale input features
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    random_forest.fit(x_train_scaled, y_train)

    # Predict using the user input
    pred = random_forest.predict(scaler.transform(df_input))

    # To display our result
    if st.button("Calculate"):
        st.subheader("The Prediction of the House Price")
        prediction = int(pred)
        result = f"${prediction:,.2f}"
        st.success(result)
        st.info("Calculation successful!")
    

    

