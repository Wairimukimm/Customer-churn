import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def open_file(filepath):
    try:
        df = pd.read_csv(filepath)

        print("Data types")
        print(df.dtypes)


        print('Shape')
        print(df.shape)

        print('Info')
        print(df.info())

    except FileNotFoundError:
        print("File not found. Enter a correct format")

def box_plots(df, column):
    plt.figure()
    sns.boxplot(x=df[column])
    plt.show()


class model:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.decision_tree = None
        self.knn = None
        self.random_forest = None

    def train_decision_tree(self):
        self.decision_tree = DecisionTreeClassifier()
        self.decision_tree.fit(self.X_train, self.y_train)

    def train_knn_neighbors(self):
        self.train_knn_neighbors = KNeighborsClassifier()
        self.train_knn_neighbors.fit(self.X_train, self.y_train)

    def train_random_classifier(self):
        self.train_random_classifier = RandomForestClassifier()
        self.train_random_classifier.fit(self.X_train, self.y_train)

    def predict_decision_tree(self, X_test):
        self.predict_decision_tree.predict(self.X_test)

    def predict_knn_neighbors(self, X_test):
        self.predict_knn_neighbors.predict(self.X_test)

    def predict_random_classifier(self, X_test):
        self.predict_random_classifier(self.X_test)

