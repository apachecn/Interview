import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import functions

def get_result():
    train_data, train_label, test_data = functions.read_data_from_csv()

    pca_model = PCA(n_components=35)
    pca_model.fit(train_data)
    new_train_data = pca_model.transform(train_data)
    new_test_data = pca_model.transform(test_data)

    rf_model = RandomForestClassifier(n_estimators=500, oob_score=True)        
    rf_model.fit(new_train_data, train_label)
    test_label = rf_model.predict(new_test_data)
    return test_label


if __name__ == '__main__':
    result = get_result()
    np.save('result_rf.npy', result)
    functions.save_result(result, 'result_rf.csv')