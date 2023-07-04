import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load your dataset from the data.csv file
dataset = pd.read_csv('data.csv')

print(dataset.columns)


# # Extract the audio features from the dataset
# audio_features = dataset.drop('eq_settings', axis=1).values

# # Extract the EQ settings from the dataset
# eq_settings = dataset['eq_settings'].values

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     audio_features, eq_settings, test_size=0.2, random_state=42)

# # Normalize the input features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Train your EQ model
# model = LinearRegression()
# model.fit(X_train, y_train)
