import re, pandas as pd

df = pd.read_csv('Food_Inspections.csv')

# iterate over columns and print out attribute names
# for col in df.columns:
	# print(col)

lat_values = df['Latitude']
long_values = df['Longitude']

print("\nThere are",(len(lat_values)),"latitude records in this data set.")

print("There are",(len(long_values)),"longitude records in this data set.\n")


print("\nThere are ", lat_values.isna().sum(),"null values that need removal from the Latitude column.")
lat_values = lat_values.dropna()
print("\nNow there are",(len(lat_values)),"latitude records in this data set.\n")

print("\nThere are ", long_values.isna().sum(),"null values that need removal from the Longitude column.")
long_values = long_values.dropna()
print("\nNow there are",(len(long_values)),"longitude records in this data set.\n")

# check that lat values are valid, must be float within range:

# check that long values are valid, must be float within range:
