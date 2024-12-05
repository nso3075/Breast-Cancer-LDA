import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/diabetes_012_health_indicators_BRFSS2015.csv")

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setting up RF model using wonders of sklearn <3
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get importnat features
important_features = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': important_features
})

# Sorting and grpahing important features
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# Configure dark theme
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance_df['Feature'], 
    feature_importance_df['Importance'], 
    color='#00cadaff', 
    #edgecolor='white'
)
plt.xlabel('Importance', color='white')
plt.ylabel('Feature', color='white')
plt.title('Feature Importance from Random Forest', color='white')

# Set background color
plt.gca().invert_yaxis()
plt.gcf().patch.set_facecolor('#120e24ff')  # Background for the figure

# Rotate y-axis labels for better visibility
plt.yticks(rotation=45, color='white')

plt.show()

# Save the figure
plt.savefig("Feature_Importance_Dark.png", facecolor='#120e24ff')