import pandas as pd
import json

df = pd.read_csv('fertilizer_recommendation_dataset.csv')

out = {
    "columns": df.columns.tolist(),
    "nulls": df.isnull().sum().to_dict(),
    "dupes": int(df.duplicated().sum()),
    "soil_types": list(df['Soil Type'].dropna().unique()),
    "crop_types": list(df['Crop Type'].dropna().unique()),
    "fertilizers": list(df['Fertilizer Name'].dropna().unique()),
    "describe": df.describe().to_dict()
}

with open('eda_output.json', 'w') as f:
    json.dump(out, f, indent=4)

print("EDA completed!")
