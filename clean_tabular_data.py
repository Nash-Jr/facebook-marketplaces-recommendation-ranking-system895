import pandas as pd
import csv
import re

images_df = pd.read_csv(
    r"C:\Users\nacho\New folder\AiCore\Facebook_Project\Images.csv")

csv_file = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\Products.csv"
rows = []
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append(row)

prod_df = pd.DataFrame(rows)

column_names = prod_df.iloc[0]

prod_df.columns = column_names
prod_df = prod_df.drop(0)


class Data_Cleaning:
    def Clean_images(images_df):
        images_df = images_df.dropna(how='all')
        images_df = images_df.copy()
        return images_df

    def Clean_products(prod_df):
        prod_df = prod_df.dropna(how='all')
        prod_df['price'] = prod_df['price'].apply(
            lambda x: re.sub(r'[^\d.]', '', str(x)))
        return prod_df


cleaned_images_df = Data_Cleaning.Clean_images(images_df)
cleaned_prod_df = Data_Cleaning.Clean_products(prod_df)


cleaned_prod_df = cleaned_prod_df.dropna(subset=['category'])
cleaned_prod_df['root_category'] = cleaned_prod_df['category'].apply(
    lambda x: x.split('/')[0].strip() if pd.notnull(x) else None)
label_encoder = {category: label for label, category in enumerate(
    cleaned_prod_df['root_category'].unique())}

merged_df = pd.merge(images_df, cleaned_prod_df[['id', 'root_category']],
                     left_on='product_id', right_on='id', how='inner')
merged_df = merged_df.dropna(subset=['root_category'])

merged_df['label'] = merged_df['root_category'].map(label_encoder)
merged_df.to_csv("training_data.csv", index=False)

print(cleaned_prod_df.head)
print(cleaned_images_df.head())
print(merged_df.head())
