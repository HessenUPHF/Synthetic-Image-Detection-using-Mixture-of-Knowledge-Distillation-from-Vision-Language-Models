import os
from PIL import Image
import pandas as pd


realSample = os.listdir('test/Test_Data/0_real/')
synthetic = os.listdir('test/Test_Data/1_fake/')


for i in range(len(realSample)):
    realSample[i] = 'test/Test_Data/0_real/' +realSample[i]

for i in range(len(synthetic)):
    synthetic[i] = 'test/Test_Data/1_fake/'+synthetic[i]

real = pd.DataFrame({
    'image': realSample,
    'text': ["real" for _ in range(len(realSample))],
    'class': [0 for _ in range(len(realSample))]
})

fake = pd.DataFrame({
    'image': synthetic,
    'text': ["fake" for _ in range(len(synthetic))],
    'class': [ 1 for _ in range(len(synthetic))]
})

data = pd.concat([real, fake]).sample(frac=1).reset_index(drop=True)

data.to_csv('Labled_Data.csv', index=False)
print(data)
