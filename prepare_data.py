import pandas as pd
from pyvi import ViTokenizer
from random import sample, seed

seed(42)

pids = pd.read_csv('data/product_ids.csv')

random_sample = sample(range(len(pids['product_id'].tolist())), 100)
print(random_sample)
print(set(random_sample))

pids = pd.read_csv('data/product_ids.csv')
sample_pids = pids.iloc[random_sample]['product_id'].tolist()

img_data = pd.read_csv('data/data_image.csv')
text_data = pd.read_csv('data/data_text.csv')

sample_imgs = img_data[img_data['id'].isin(sample_pids)].reset_index().drop(columns=['Unnamed: 0']).drop(columns=['index'])
sample_text = text_data[text_data['id'].isin(sample_pids)].reset_index().drop(columns=['index'])

print(sample_imgs)
print(sample_imgs['keywords'])
print(sample_imgs['keywords'][0])
print(sample_text)

sample_imgs.to_csv('data/pdata/sample_imgs.csv', encoding='utf-8')
sample_text.to_csv('data/pdata/sample_text.csv', encoding='utf-8')