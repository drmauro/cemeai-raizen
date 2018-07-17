import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def hist_nas(raw_data):
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


raw_data = pd.read_csv('../data/raw_data/falhas_v1.csv',
                       delimiter=';', encoding=' Latin-1')
print("Raw data --> {0}".format(raw_data.shape))

idx_to_rem = ['Unnamed: 0',  # ids
              'Zona',  # undefined
              'Nome_Estacao_SM1',  # undefined
              'CD_EMPRESA1',  # undefined
              'Fazenda',
              'Talhao',
              'INSTANCIA1']
idx_non_numeric = ['Variedade',
                   'Ambiente',
                   'Dt_anal1',  # need some transformation
                   'Data_Plantio',  # need some transformation
                   'Estagio',
                   'Ciclo']

raw_data = raw_data.drop(idx_to_rem, axis=1)
raw_data = raw_data.drop(idx_non_numeric, axis=1)
print("idx and undefined attr removed --> {0}".format(raw_data.shape))

# Removing att with >= 15% nas
perc = (raw_data.isnull().sum() / raw_data.shape[0]) >= 0.15
raw_data = raw_data.drop(raw_data.columns[perc], axis=1)
print("attr with >= 15% na's removed --> {0}".format(raw_data.shape))
# hist_nas(raw_data)

# Removing examples with >= 15% nas
perc = (raw_data.T.isnull().sum() / raw_data.T.shape[0]) >= 0.15
raw_data = raw_data.drop(raw_data.index[perc])
print("examples with >= 15% na's removed --> {0}".format(raw_data.shape))

# Train
targets = raw_data['Perc_Falha']
data = raw_data.drop('Perc_Falha', axis=1)
data = [pd.to_numeric(data[i]) for i in data.columns]

X_train, X_test, y_train, y_test = train_test_split(
    data, targets, train_size=0.8)

rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
rf.fit(X_train, y_train)


# all_rem = idx_to_rem + idx_non_numeric
# new_data = raw_data.drop(all_rem, axis=1)
