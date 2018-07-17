import pandas as pd
import matplotlib.pyplot as plt


raw_data = pd.read_csv('../data/raw-data/falhas_plantio_hashing.csv',
                       delimiter=';', encoding=' Latin-1')

raw_data_t = raw_data.T
perc = raw_data.isnull().sum() / raw_data.shape[0]
perc_sort = perc.sort_values

perc_t = raw_data_t.isnull().sum() / raw_data_t.shape[0]
perc_sort_t = perc_t.sort_values

plt.figure()
perc.hist()
plt.show()
perc_t.hist()
plt.show()



