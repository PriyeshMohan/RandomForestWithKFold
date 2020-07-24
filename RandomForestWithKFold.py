from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


x,y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant= 0, n_clusters_per_class=1, random_state=60)
x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y,random_state=42)

%matplotlib inline
import matplotlib.pyplot as plt
colors = {0:'red', 1:'blue'}
plt.scatter(X_test[:,0], X_test[:,1],c=y_test)
plt.show()

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")

# Will return test indices from. start to end and rest indices as train indices
def GetIndices(start, end, length):
    print("")
    print("Printing indices")
    print("start - ",start)
    print("end - ",end)
    print("total length - ",length)
    cv_indices = []
    train_indices = []
    for i in range(length):
        if i >= start and i < end:
            cv_indices.append(i)
        else:
            train_indices.append(i)
    return cv_indices, train_indices

def TrainTestAndPredict(x_train,y_train,classifier,train_indices,test_indices,neightbour):
    #Split the train and test based on the train and test indices
    print("training indices count", len(train_indices))
    print("testing indices are", len(test_indices))
    X_train = x_train[train_indices]
    Y_train = y_train[train_indices]
    X_test  = x_train[test_indices]
    Y_test  = y_train[test_indices]
    #set the neighbour value
    classifier.n_neighbors = neightbour
    classifier.fit(X_train,Y_train)
    Y_train_predicted = classifier.predict(X_train)
    Y_test_predicted = classifier.predict(X_test)
    #return the accuracy value
    return accuracy_score(Y_train, Y_train_predicted), accuracy_score(Y_test, Y_test_predicted)

def RandomSearchCV(x_train,y_train,classifier, param_range, folds):
    train_scores = []
    test_scores  = []
    random_k_values = random.sample(range(param_range[0],param_range[len(param_range) - 1]), 10)
    random_k_values.sort()
    print("Radom Values generated for K Task #1")
    print(random_k_values)
    print("-"*50)
    # what percentile of the value is k fold ? Eg- if k = 3? then 33.33 %
    percentile = (10/folds)*10
    # index for that percentile value in the list
    index = int(percentile*len(x_train)/100)
    print("Percentile index",index)
    for k in random_k_values:
        print("-"*50)
        trainscores_folds = []
        testscores_folds  = []
        j = 0
        for _fold in range(folds):
            # repeat for each k fold batch, except for the last batch
            if _fold < folds - 1 and j < index + j:
                test_indices, train_indices = GetIndices(j, index + j, len(x_train))
                each_train_score_folds, each_test_score_folds = TrainTestAndPredict(x_train,y_train,classifier,train_indices,test_indices,k)
                print("train score in fold batch ",_fold," is", each_train_score_folds)
                trainscores_folds.append(each_train_score_folds)
                print("test score in fold batch ",_fold," is", each_test_score_folds)
                testscores_folds.append(each_test_score_folds)
                j = index + j
            else:
                # last batch will inlcude the remaining indices
                test_indexes, train_indexes = GetIndices(j, len(x_train), len(x_train))
                each_train_score_folds, each_test_score_folds = TrainTestAndPredict(x_train,y_train,classifier,train_indices,test_indices,k)
                trainscores_folds.append(each_train_score_folds)
                print("train score in fold batch ",_fold," is", each_train_score_folds)
                print("test score in fold batch ",_fold," is", each_test_score_folds)
                testscores_folds.append(each_test_score_folds)
        training_mean = np.mean(np.array(trainscores_folds))
        testing_mean = np.mean(np.array(testscores_folds))
        print("")
        print("taking mean of training for k = ",k,"from",trainscores_folds,"mean = ",training_mean)
        print("taking mean of testing for k = ",k,"from",testscores_folds,"mean = ",testing_mean)
        print("")
        train_scores.append(training_mean)
        test_scores.append(testing_mean)
        print("-"*50)
    return train_scores, test_scores, random_k_values

# param range
param_range = (1, 50)
classifier = KNeighborsClassifier()
# fold
folds = 3
train_scores, test_scores, random_k_values = RandomSearchCV(x_train,y_train,classifier, param_range, folds)


print(train_scores)
print(test_scores)
print(random_k_values)
plt.plot(random_k_values,train_scores, label='train cruve')
plt.plot(random_k_values,test_scores, label='test cruve')
plt.title('Hyper-parameter VS accuracy plot')
plt.legend()
plt.show()


# understanding this code line by line is not that importent 
def plot_decision_boundary(X1, X2, y, clf):
        # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    x_min, x_max = X1.min() - 1, X1.max() + 1
    y_min, y_max = X2.min() - 1, X2.max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X1, X2, c=y, cmap=cmap_bold)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class classification (k = %i)" % (clf.n_neighbors))
    plt.show()

from matplotlib.colors import ListedColormap
neigh = KNeighborsClassifier(n_neighbors = 43)
neigh.fit(X_train, y_train)
plot_decision_boundary(X_train[:, 0], X_train[:, 1], y_train, neigh)


