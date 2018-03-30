import functions
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import LocallyLinearEmbedding as LLE

data, label, test_data = functions.read_data_from_csv()
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42)

def genearte_knn_model():
    weights = 'distance'
    for n_neighbors in range(1, 7):
        for metric in ['euclidean', 'manhattan']:
            model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                        weights=weights,
                                        metric=metric)
            print('n_neighbors={}\n weights={}\n metric={} \n'.format(n_neighbors, weights, metric))
            yield model
def generate_lle_model():
    n_neighbors = 5
    for n_components in range(2, 6):
        model = LLE(n_neighbors=n_neighbors, n_components=n_components)
        print('lle:\n n_neighbors={}\n n_components={}\n'.format(n_neighbors, n_components))
        yield model

for knn_model in genearte_knn_model():
    for lle_model in generate_lle_model():
        lle_model.fit(x_train)
        new_x_train = lle_model.transform(x_train)
        new_x_test = lle_model.transform(x_test)
        
        knn_model.fit(new_x_train, y_train)
        score = knn_model.score(new_x_test, y_test)
        print('score={}\n'.format(score))