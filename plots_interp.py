import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict

class DenseNetwork(torch.nn.Module):
    def __init__(self, in_features, hidden_size, n_classes, n_layers, dropout, activation=nn.ReLU):
        super().__init__()

        self.layers = n_layers
        self.layer_size = hidden_size
        self.in_features = in_features
        self.activation = activation
        l = []
        if n_layers == 1:
          l.append(('conv0', nn.Linear(self.in_features, n_classes)))
        else:
          l.append(('conv0', nn.Linear(self.in_features, self.layer_size)))
          l.append(('relu0', self.activation()))
          for i in range(1, self.layers - 1):
            l.append((f'conv{i}', nn.Linear(self.layer_size, self.layer_size)))
            l.append((f'relu{i}', self.activation()))
            l.append((f'dropout{i}', nn.Dropout(p=dropout)))
          l.append(('conv_final', nn.Linear(self.layer_size, n_classes)))
          l.append(('sigmoid', nn.Sigmoid()))
        self.layers = nn.Sequential(OrderedDict(l))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

def main():
    data_train_base = pd.read_csv("interp_extra_train_nn.csv")
    X_train_base = data_train_base.drop(columns=["y"]).values
    y_train_base = data_train_base["y"]

    data_test_base = pd.read_csv("interp_extra_test_nn.csv")
    X_test_base = data_test_base.drop(columns=["y"]).values
    y_test_base = data_test_base["y"]

    data_train_tree = pd.read_csv("interp_extra_train_tree.csv")
    X_train_tree = data_train_tree.drop(columns=["y"]).values
    y_train_tree = data_train_tree["y"]

    data_test_tree = pd.read_csv("interp_extra_test_tree.csv")
    X_test_tree = data_test_tree.drop(columns=["y"]).values
    y_test_tree = data_test_tree["y"]

    svm = SVC(kernel="rbf", C=1000)
    grad = GradientBoostingClassifier()
    forest = RandomForestClassifier()
    net = DenseNetwork(19, 32, 1, 5, 0.1)
    net.load_state_dict(torch.load("interp_extra_sigmoid.pth", map_location=torch.device('cpu')))

    grad.fit(X_train_base, y_train_base)
    svm.fit(X_train_base, y_train_base)
    forest.fit(X_train_base, y_train_base)

    new_y_svm = svm.predict(X_train_tree)
    new_y_grad = grad.predict(X_train_tree)
    new_y_forest = forest.predict(X_train_tree)
    y_tmp = net(torch.tensor(X_train_tree, dtype=torch.float32))
    y_tmp = y_tmp > 0.5
    new_y_nn = y_tmp.detach().flatten().numpy()

    acc_reg = []
    acc_svm = []
    acc_grad = []
    acc_forest = []
    acc_nn = []

    for i in range(1, 16):
        model_reg = DecisionTreeClassifier(max_depth=i, criterion="gini")
        model_svm = DecisionTreeClassifier(max_depth=i, criterion="gini")
        model_grad = DecisionTreeClassifier(max_depth=i, criterion="gini")
        model_forest = DecisionTreeClassifier(max_depth=i, criterion="gini")
        model_nn = DecisionTreeClassifier(max_depth=i, criterion="gini")

        model_reg.fit(X_train_tree, y_train_tree)
        model_svm.fit(X_train_tree, new_y_svm)
        model_grad.fit(X_train_tree, new_y_grad)
        model_forest.fit(X_train_tree, new_y_forest)
        model_nn.fit(X_train_tree, new_y_nn)

        acc_reg.append(accuracy_score(model_reg.predict(X_test_tree), y_test_tree))
        acc_svm.append(accuracy_score(model_svm.predict(X_test_tree), y_test_tree))
        acc_grad.append(accuracy_score(model_grad.predict(X_test_tree), y_test_tree))
        acc_forest.append(accuracy_score(model_forest.predict(X_test_tree), y_test_tree))
        acc_nn.append(accuracy_score(model_nn.predict(X_test_tree), y_test_tree))

    plt.figure(figsize=(15, 5))
    plt.grid(True)
    plt.title('Зависимость точности моделей от максимальной глубины', fontsize=10)
    plt.ylabel('Точность', fontsize=10)
    plt.xlabel('Глубина', fontsize=10)
    plt.plot(range(1, 16), acc_reg, color="b", label="Модель без предобработки")
    plt.plot(range(1, 16), acc_svm, color="r", label="Модель с предобработкой SVM")
    plt.plot(range(1, 16), acc_grad, color="g", label="Модель с предобработкой градиентным бустингом")
    plt.plot(range(1, 16), acc_forest, color="c", label="Модель с предобработкой случайным лесом")
    plt.plot(range(1, 16), acc_nn, color="m", label="Модель с предобработкой нейронной сетью")
    plt.legend()
    plt.xticks(range(1, 16))
    plt.savefig('all_models.png', bbox_inches='tight', dpi=100)
    plt.show()


if __name__ == '__main__':
    main()