from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import time
from SparseAutoEncoder import *
from VariationalAutoEncoder import *
from Embedding import *
import pickle


def pca_isolation():
    normal_data, anomaly_data = read_data()
    normal_data = normal_data.values
    anomaly_data = anomaly_data.values
    anomaly_data = np.reshape(anomaly_data, (-1, 25))

    start_time = time.time()
    iForest = IsolationForest(n_estimators=20).fit(normal_data)
    train_time = time.time() - start_time

    start_time = time.time()
    normal_result = iForest.predict(normal_data).tolist()
    anomaly_result = iForest.predict(anomaly_data).tolist()
    test_time = (time.time() - start_time)/42000.0

    TP = normal_result.count(1)
    TN = normal_result.count(-1)
    FP = anomaly_result.count(1)
    FN = anomaly_result.count(-1)
    Accuracy = (TP + FN) / (TP + TN + FP + FN)
    result = [0, Accuracy, TP, TN, FP, FN, train_time, test_time]

    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    document = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\pca_result.csv")
    if os.path.exists(document):
        os.remove(document)

    with open(document, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['dimension', 'accuracy', 'TP', 'TN', 'FP', 'FN', 'train_time', 'test_time'])
        writer.writerow(result)

    for dimension in range(1, 26):
        start_time = time.time()
        pca = PCA(n_components=dimension).fit(normal_data)
        reduction_normal_data = pca.transform(normal_data)
        iForest = IsolationForest(n_estimators=20).fit(reduction_normal_data)
        train_time = time.time() - start_time

        start_time = time.time()
        reduction_normal_data = pca.transform(normal_data)
        reduction_anomaly_data = pca.transform(anomaly_data)
        normal_result = iForest.predict(reduction_normal_data).tolist()
        anomaly_result = iForest.predict(reduction_anomaly_data).tolist()
        test_time = (time.time() - start_time)/42000.0

        TP = normal_result.count(1)
        TN = normal_result.count(-1)
        FP = anomaly_result.count(1)
        FN = anomaly_result.count(-1)
        Accuracy = (TP + FN) / (TP + TN + FP + FN)
        result = [dimension, Accuracy, TP, TN, FP, FN, train_time, test_time]

        with open(document, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(result)


def sae_isolation():
    normal_data, anomaly_data = read_data()
    normal_data = normal_data.values
    anomaly_data = anomaly_data.values
    anomaly_data = np.reshape(anomaly_data, (-1, 25))

    start_time = time.time()
    iForest = IsolationForest(n_estimators=20).fit(normal_data)
    train_time = time.time() - start_time

    start_time = time.time()
    normal_result = iForest.predict(normal_data).tolist()
    anomaly_result = iForest.predict(anomaly_data).tolist()
    test_time = (time.time() - start_time)/42000.0

    TP = normal_result.count(1)
    TN = normal_result.count(-1)
    FP = anomaly_result.count(1)
    FN = anomaly_result.count(-1)
    Accuracy = (TP + FN) / (TP + TN + FP + FN)
    result = [0, Accuracy, TP, TN, FP, FN, train_time, test_time]

    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    document = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\sae_result.csv")
    if os.path.exists(document):
        os.remove(document)

    with open(document, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['dimension', 'accuracy', 'TP', 'TN', 'FP', 'FN', 'train_time', 'test_time'])
        writer.writerow(result)

    best_Accuracy = -1.0
    for dimension in range(1, 26):
        start_time = time.time()
        sae_model = sae_train(normal_data, dimension)
        reduction_normal_data = sae_model.encoder(normal_data)
        iForest = IsolationForest(n_estimators=20).fit(reduction_normal_data)
        train_time = time.time() - start_time

        start_time = time.time()
        reduction_normal_data = sae_model.encoder(normal_data)
        reduction_anomaly_data = sae_model.encoder(anomaly_data)
        normal_result = iForest.predict(reduction_normal_data).tolist()
        anomaly_result = iForest.predict(reduction_anomaly_data).tolist()
        test_time = (time.time() - start_time)/42000.0

        TP = normal_result.count(1)
        TN = normal_result.count(-1)
        FP = anomaly_result.count(1)
        FN = anomaly_result.count(-1)
        Accuracy = (TP + FN) / (TP + TN + FP + FN)
        result = [dimension, Accuracy, TP, TN, FP, FN, train_time, test_time]

        if best_Accuracy < Accuracy:
            sae_model.save_weights('sae_train_model.h5')
            with open('./sae_iForest.pk', 'wb') as f:
                pickle.dump(iForest, f)
            best_Accuracy = Accuracy

        with open(document, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(result)


def vae_isolation():
    normal_data, anomaly_data = read_data()
    normal_data = normal_data.values
    anomaly_data = anomaly_data.values
    anomaly_data = np.reshape(anomaly_data, (-1, 25))

    start_time = time.time()
    iForest = IsolationForest(n_estimators=20).fit(normal_data)
    train_time = time.time() - start_time

    start_time = time.time()
    normal_result = iForest.predict(normal_data).tolist()
    anomaly_result = iForest.predict(anomaly_data).tolist()
    test_time = (time.time() - start_time)/42000.0

    TP = normal_result.count(1)
    TN = normal_result.count(-1)
    FP = anomaly_result.count(1)
    FN = anomaly_result.count(-1)
    Accuracy = (TP + FN) / (TP + TN + FP + FN)
    result = [0, Accuracy, TP, TN, FP, FN, train_time, test_time]

    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    document = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\vae_result.csv")
    if os.path.exists(document):
        os.remove(document)

    with open(document, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['dimension', 'accuracy', 'TP', 'TN', 'FP', 'FN', 'train_time', 'test_time'])
        writer.writerow(result)

    best_Accuracy = -1.0
    for dimension in range(1, 26):
        start_time = time.time()
        vae_model = vae_train(normal_data, dimension)
        reduction_normal_data = vae_model.encoder(normal_data)
        iForest = IsolationForest(n_estimators=20).fit(reduction_normal_data)
        train_time = time.time() - start_time

        start_time = time.time()
        reduction_normal_data = vae_model.encoder(normal_data)
        reduction_anomaly_data = vae_model.encoder(anomaly_data)
        normal_result = iForest.predict(reduction_normal_data).tolist()
        anomaly_result = iForest.predict(reduction_anomaly_data).tolist()
        test_time = (time.time() - start_time)/42000.0

        TP = normal_result.count(1)
        TN = normal_result.count(-1)
        FP = anomaly_result.count(1)
        FN = anomaly_result.count(-1)
        Accuracy = (TP + FN) / (TP + TN + FP + FN)
        result = [dimension, Accuracy, TP, TN, FP, FN, train_time, test_time]

        if best_Accuracy < Accuracy:
            vae_model.save_weights('vae_train_model.h5')
            with open('./vae_iForest.pk', 'wb') as f:
                pickle.dump(iForest, f)
            best_Accuracy = Accuracy

        with open(document, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(result)


def embedding_isolation():
    normal_data, anomaly_data = read_data()
    normal_data = normal_data.values
    anomaly_data = anomaly_data.values
    anomaly_data = np.reshape(anomaly_data, (-1, 25))

    start_time = time.time()
    iForest = IsolationForest(n_estimators=20).fit(normal_data)
    train_time = time.time() - start_time

    start_time = time.time()
    normal_result = iForest.predict(normal_data).tolist()
    anomaly_result = iForest.predict(anomaly_data).tolist()
    test_time = (time.time() - start_time)/42000.0

    TP = normal_result.count(1)
    TN = normal_result.count(-1)
    FP = anomaly_result.count(1)
    FN = anomaly_result.count(-1)
    Accuracy = (TP + FN) / (TP + TN + FP + FN)
    result = [0, Accuracy, TP, TN, FP, FN, train_time, test_time]

    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    document = os.path.join(grandfather_path, "Calculations\\Fault alarm paper\\embedding_result.csv")
    if os.path.exists(document):
        os.remove(document)

    with open(document, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['dimension', 'accuracy', 'TP', 'TN', 'FP', 'FN', 'train_time', 'test_time'])
        writer.writerow(result)

    for dimension in range(1, 26):
        start_time = time.time()
        embedding_model = embedding_train(normal_data, dimension)
        reduction_normal_data = embedding_model.encoder(normal_data)
        iForest = IsolationForest(n_estimators=20).fit(reduction_normal_data)
        train_time = time.time() - start_time

        start_time = time.time()
        reduction_normal_data = embedding_model.encoder(normal_data)
        reduction_anomaly_data = embedding_model.encoder(anomaly_data)
        normal_result = iForest.predict(reduction_normal_data).tolist()
        anomaly_result = iForest.predict(reduction_anomaly_data).tolist()
        test_time = (time.time() - start_time)/42000.0

        TP = normal_result.count(1)
        TN = normal_result.count(-1)
        FP = anomaly_result.count(1)
        FN = anomaly_result.count(-1)
        Accuracy = (TP + FN) / (TP + TN + FP + FN)
        result = [dimension, Accuracy, TP, TN, FP, FN, train_time, test_time]

        with open(document, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(result)


# pca_isolation()
# sae_isolation()
vae_isolation()
# embedding_isolation()
