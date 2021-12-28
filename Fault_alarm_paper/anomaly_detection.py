from IO_function import *
from SparseAutoEncoder import *
from sklearn.ensemble import IsolationForest
import pickle


normal_data, anomaly_data = read_data()
normal_data = normal_data.values
anomaly_data = anomaly_data.values
anomaly_data = np.reshape(anomaly_data, (-1, 25))

sae_model = sae_train(normal_data, 2)
sae_model.save_weights("sae_train_model.h5")

iForest = IsolationForest(n_estimators=20).fit(normal_data)
with open('./iForest.pk', 'wb') as f:
    pickle.dump(iForest, f)
