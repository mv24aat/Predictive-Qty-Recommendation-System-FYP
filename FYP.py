import category_encoders
import pandas as pd
import numpy as np
import pip
import pip
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('merged_patients.csv')

print(f'Shape: {df.shape}')
df.head()

df.info()

# Separate numeric and categorical columns
num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

print('Numeric columns:', num_cols)
print('\nCategorical columns:', cat_cols)

df[num_cols].describe().round(2)

df[num_cols].describe().round(2)

# Categorical value counts
for col in ['Sex', 'Ethnicity', 'CYP2C9', 'VKORC1', 'CYP4F2',
            'Alcohol_Intake', 'Smoking_Status', 'Diet_VitK_Intake',
            'Adverse_Event']:
    print(f'\n--- {col} ---')
    print(df[col].value_counts(dropna=False))


#CHECHING FOR MISSING DATA

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({'Missing count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing count'] > 0]
print(missing_df)

# Plot
fig, ax = plt.subplots(figsize=(7, 3))
missing_df['Missing %'].plot(kind='barh', ax=ax, color=['#E24B4A', '#EF9F27'])
ax.set_xlabel('Missing (%)')
ax.set_title('Missing value rate by column')
for i, v in enumerate(missing_df['Missing %']):
    ax.text(v + 0.5, i, f'{v}%', va='center', fontsize=11)
plt.tight_layout()
plt.show()

# Recode Adverse_Event as binary (blank = no event)
df['Adverse_Event_bin'] = df['Adverse_Event'].notna().astype(int)
print('Adverse events (recoded):')
print(df['Adverse_Event_bin'].value_counts())

df['Adverse_Event'] = df['Adverse_Event'].fillna('No Event')

df['Alcohol_Intake'] = df['Alcohol_Intake'].fillna('Non-drinker')


# Adjust the layout to avoid overlap of plot elements
bins = 6

fig, ax = plt.subplots(2, 3, figsize=(14, 8))

sns.histplot(df['Age'], bins=bins, kde=True, ax=ax[0, 0])
ax[0, 0].set_title('Age Distribution')

sns.histplot(df['Weight_kg'], bins=bins, kde=True, ax=ax[0, 1])
ax[0, 1].set_title('Weight_kg Distribution')

sns.histplot(df['Height_cm'], bins=bins, kde=True, ax=ax[0, 2])
ax[0, 2].set_title('Height_cm Distribution')

sns.histplot(df['Final_Stable_Dose_mg'], bins=bins, kde=True, ax=ax[1, 0])
ax[1, 0].set_title('Final_Stable_Dose_mg Distribution')

sns.histplot(df['INR_Stabilization_Days'], bins=bins, kde=True, ax=ax[1, 1])
ax[1, 1].set_title('INR_Stabilization_Days Distribution')

sns.histplot(df['TTR_pct'], bins=bins, kde=True, ax=ax[1, 2])
ax[1, 2].set_title('TTR_pct Distribution')

plt.tight_layout()
plt.show()


# Create a 2x4 grid of subplots to fit 7 categorical columns
fig, ax = plt.subplots(2, 4, figsize=(18, 8))

# Plot the count distribution of 'Sex'
sns.countplot(data=df, x='Sex', ax=ax[0, 0])
ax[0, 0].set_title('Sex Distribution')

# Plot the count distribution of 'Ethnicity'
sns.countplot(data=df, x='Ethnicity', ax=ax[0, 1])
ax[0, 1].set_title('Ethnicity Distribution')
ax[0, 1].tick_params(axis='x', rotation=30)

# Plot the count distribution of 'CYP2C9'
sns.countplot(data=df, x='CYP2C9', ax=ax[0, 2])
ax[0, 2].set_title('CYP2C9 Distribution')

# Plot the count distribution of 'VKORC1'
sns.countplot(data=df, x='VKORC1', ax=ax[0, 3])
ax[0, 3].set_title('VKORC1 Distribution')

# Plot the count distribution of 'CYP4F2'
sns.countplot(data=df, x='CYP4F2', ax=ax[1, 0])
ax[1, 0].set_title('CYP4F2 Distribution')

# Plot the count distribution of 'Alcohol_Intake'
sns.countplot(data=df, x='Alcohol_Intake', ax=ax[1, 1])
ax[1, 1].set_title('Alcohol_Intake Distribution')

# Plot the count distribution of 'Smoking_Status'
sns.countplot(data=df, x='Smoking_Status', ax=ax[1, 2])
ax[1, 2].set_title('Smoking_Status Distribution')
ax[1, 2].tick_params(axis='x', rotation=20)

# Hide the unused last subplot
ax[1, 3].set_visible(False)

# Adjust the layout to avoid overlap of plot elements
plt.tight_layout()

# Display the plot
plt.show()

# Loop through the list of genes and print the normalized value counts for each
for gene in ['CYP2C9', 'VKORC1', 'CYP4F2']:
    # Print the distribution of values for the current gene
    # 'value_counts(normalize=True)' returns the relative frequencies of each value
    print(f'\n{gene} Distribution:\n', df[gene].value_counts(normalize=True))

display('Alcohol Intake:', df['Alcohol_Intake'].value_counts(normalize=True))

display('Adverse Events (Percentage):', df['Adverse_Event'].value_counts(normalize=True))
display('Adverse Events (Count):', df['Adverse_Event'].value_counts())

# Create a boxplot to visualize the relationship between 'CYP2C9' genotype and 'Final_Stable_Dose_mg'
plt.figure(figsize=(7, 5))  # Set the figure size

# Plot the boxplot using Seaborn
sns.boxplot(data=df, x='CYP2C9', y='Final_Stable_Dose_mg')

# Add a title to the plot
plt.title('Warfarin Dose by CYP2C9 Genotype')
plt.show()

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from category_encoders import OneHotEncoder

pip install category_encoders

df.columns