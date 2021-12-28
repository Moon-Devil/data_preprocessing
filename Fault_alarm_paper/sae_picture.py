from IO_function import *
from SparseAutoEncoder import sae_train
from VariationalAutoEncoder import vae_train


def sae_reduction_picture(normal_data, anomaly_data, reduction_dimension):
    sae_model = sae_train(normal_data, reduction_dimension)
    reduction_normal_data = sae_model.encoder(normal_data)
    reduction_anomaly_data = sae_model.encoder(anomaly_data)

    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    document = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\sae_normal_data_" +
                            str(reduction_dimension) + "dimension.csv")

    if os.path.exists(document):
        os.remove(document)

    with open(document, "w", newline="") as f:
        writer = csv.writer(f)
        for i_index in range(1000):
            writer.writerow(reduction_normal_data[i_index, :])

    document = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\sae_anomaly_data_" +
                            str(reduction_dimension) + "dimension.csv")

    if os.path.exists(document):
        os.remove(document)

    with open(document, "w", newline="") as f:
        writer = csv.writer(f)
        for i_index in range(6):
            for j_index in range(1000):
                index = i_index * 1000 + j_index
                writer.writerow(reduction_anomaly_data[index, :])


def vae_reduction_picture(normal_data, anomaly_data, reduction_dimension):
    vae_model = vae_train(normal_data, reduction_dimension)
    reduction_normal_data = vae_model.encoder(normal_data)
    reduction_anomaly_data = vae_model.encoder(anomaly_data)

    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    document = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\vae_normal_data_" +
                            str(reduction_dimension) + "dimension.csv")

    if os.path.exists(document):
        os.remove(document)

    with open(document, "w", newline="") as f:
        writer = csv.writer(f)
        for i_index in range(1000):
            writer.writerow(reduction_normal_data[i_index, :])

    document = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\vae_anomaly_data_" +
                            str(reduction_dimension) + "dimension.csv")

    if os.path.exists(document):
        os.remove(document)

    with open(document, "w", newline="") as f:
        writer = csv.writer(f)
        for i_index in range(6):
            for j_index in range(1000):
                index = i_index * 1000 + j_index
                writer.writerow(reduction_anomaly_data[index, :])


_normal_data, _anomaly_data = read_data()
_normal_data = _normal_data.values
_anomaly_data = _anomaly_data.values
_anomaly_data = np.reshape(_anomaly_data, (-1, 25))

# sae_reduction_picture(_normal_data, _anomaly_data, 2)
# sae_reduction_picture(_normal_data, _anomaly_data, 3)
vae_reduction_picture(_normal_data, _anomaly_data, 1)

print("done...")

