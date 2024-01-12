import pandas as pd
import jsonlines

# Path to the JSONL file
jsonl_file_path = 'data\\json\\s2_full_figures_oa_nonroco_combined_medical_top4_public.jsonl'

# Lists to store data
image_name = []
fig_key = []
fig_uri = []
s2_caption = []
s2orc_caption = []

# Reading data from JSONL file
with jsonlines.open(jsonl_file_path) as f:
    for line in f.iter():
        image_name.append(line['pdf_hash'])
        fig_key.append(line['fig_key'])
        fig_uri.append(line['fig_uri'])
        s2_caption.append(line['s2_caption'])
        s2orc_caption.append(line['s2orc_caption'])

# Creating a DataFrame from the lists
df = pd.DataFrame({
    'image_name': image_name,
    'fig_key': fig_key,
    'fig_uri': fig_uri,
    's2_caption': s2_caption,
    's2orc_caption': s2orc_caption
})

# Displaying the first few rows of the dataframe
# print(df.head())

# Saving the dataframe as a CSV file
df.to_csv('data\\csv\\s2_full_figures_oa_nonroco_combined_medical_top4_public.csv', index=False)