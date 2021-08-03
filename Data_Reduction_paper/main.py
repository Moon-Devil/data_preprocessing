from combine_function import *


index = 24
batch_size = 16
epochs = 200

PCA_DNN_function(index, batch_size, epochs)
Isomap_DNN_function(index, batch_size, epochs)
LLE_DNN_function(index, batch_size, epochs)
AE_DNN_function(index, batch_size, epochs)
VAE_DNN_function(index, batch_size, epochs)
# RBM_DNN_function(index, batch_size, epochs)

PCA_SVR_function(index)
Isomap_SVR_function(index)
LLE_SVR_function(index)
AE_SVR_function(index, batch_size, epochs)
VAE_SVR_function(index, epochs)
# RBM_SVR_function(index, epochs)

PCA_GPR_function(index)
Isomap_GPR_function(index)
LLE_GPR_function(index)
AE_GPR_function(index, batch_size, epochs)
VAE_GPR_function(index, epochs)
# RBM_GPR_function(index, epochs)

PCA_RNN_function(index, batch_size, epochs)
Isomap_RNN_function(index, batch_size, epochs)
LLE_RNN_function(index, batch_size, epochs)
AE_RNN_function(index, batch_size, epochs)
VAE_RNN_function(index, batch_size, epochs)
# RBM_RNN_function(index, batch_size, epochs)

print("done...")
