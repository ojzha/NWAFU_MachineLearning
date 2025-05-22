import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (precision_recall_fscore_support,
                             classification_report,
                             confusion_matrix)


os.makedirs('images', exist_ok=True)


plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.titlepad': 15,
    'figure.figsize': (8, 6)
})
sns.set_palette("Set2")


class RBFClassifier:
    def __init__(self, n_centers=50, random_state=23):
        self.n_centers = n_centers
        self.random_state = random_state
        self.centers = None
        self.sigma = None
        self.W = None
        self.classes_ = None
        self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        kmeans = KMeans(n_clusters=self.n_centers,
                        random_state=self.random_state,
                        n_init=10)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        distances = cdist(self.centers, self.centers, 'euclidean')
        self.sigma = np.mean(distances) / np.sqrt(2 * self.n_centers)

        rbf_features = self._compute_rbf_features(X)

        y_reshaped = y.reshape(-1, 1)
        self.encoder.fit(y_reshaped)
        y_onehot = self.encoder.transform(y_reshaped)

        self.W = np.linalg.pinv(rbf_features) @ y_onehot

        return self

    def _compute_rbf_features(self, X):
        sq_dist = cdist(X, self.centers, 'sqeuclidean')
        return np.exp(-sq_dist / (2 * self.sigma ** 2))

    def predict(self, X):
        rbf_features = self._compute_rbf_features(X)

        y_pred_onehot = rbf_features @ self.W

        y_pred_indices = np.argmax(y_pred_onehot, axis=1)

        return self.classes_[y_pred_indices]

    def predict_proba(self, X):
        rbf_features = self._compute_rbf_features(X)

        y_pred_onehot = rbf_features @ self.W

        y_pred_exp = np.exp(y_pred_onehot - np.max(y_pred_onehot, axis=1, keepdims=True))
        return y_pred_exp / np.sum(y_pred_exp, axis=1, keepdims=True)



def plot_class_performance(name, classes, precision, recall, f1):
    plt.figure()
    x = np.arange(len(classes))
    width = 0.25

    bars = []
    for i, (values, label) in enumerate(zip([precision, recall, f1],
                                            ['Precision', 'Recall', 'F1'])):
        bar = plt.bar(x + i * width - width, values, width, label=label)
        bars.append(bar)

    plt.xticks(x, classes, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.ylabel('Score')
    plt.title(f'{name} - Class-wise Performance')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.savefig(f'images/{name}_class_performance.png')
    plt.close()


def plot_confusion_matrix(name, y_true, y_pred, classes):

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=classes, yticklabels=classes)

    plt.title(f'{name} - Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'images/{name}_confusion_matrix.png')
    plt.close()


def plot_accuracy_comparison(labels, accuracies):
    plt.figure()
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    plt.bar(range(len(labels)), accuracies, color=colors, tick_label=labels)
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Cross-dataset Accuracy Comparison')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('images/accuracy_comparison.png')
    plt.close()


def plot_metric_trends(labels, metrics):
    plt.figure()
    markers = ['o', 's', 'D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (metric, color, marker) in enumerate(zip(['precision', 'recall', 'f1'],
                                                    colors, markers)):
        values = [metrics[metric][i] for i in range(len(labels))]
        plt.plot(range(len(labels)), values,
                 marker=marker, color=color,
                 linestyle='--', linewidth=2,
                 label=metric.capitalize())

    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.title('Performance Metric Trends')
    plt.xticks(range(len(labels)), labels, rotation=15)
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/metric_trends.png')
    plt.close()


def main():
    datasets = {
        'MNIST': 'datasets/MNIST.mat',
        'Yale': 'datasets/Yale.mat',
        'Lung': 'datasets/lung.mat'
    }

    models = [
        ('rbf', RBFClassifier(n_centers=50, random_state=23)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu',
                              max_iter=300, learning_rate_init=0.001, random_state=23))
    ]

    performance_metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy': [],
        'labels': []
    }

    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"Dataset {name} not found at {path}")
            continue
        data = loadmat(path)
        X = data[list(data.keys())[-2]]
        y = data[list(data.keys())[-1]].flatten()
        classes = np.unique(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=23
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for model_name, model in models:
            combined_name = f"{name}_{model_name}"
            performance_metrics['labels'].append(combined_name)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = np.mean(y_pred == y_test) * 100
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

            performance_metrics['precision'].append(np.mean(precision))
            performance_metrics['recall'].append(np.mean(recall))
            performance_metrics['f1'].append(np.mean(f1))
            performance_metrics['accuracy'].append(accuracy)

            plot_class_performance(combined_name, classes, precision, recall, f1)
            plot_confusion_matrix(combined_name, y_test, y_pred, classes)

            print(f"\n=== {combined_name} Classification Report ===")
            print(classification_report(y_test, y_pred, zero_division=0))
            print(f"准确率: {accuracy:.2f}%")

    plot_accuracy_comparison(performance_metrics['labels'], performance_metrics['accuracy'])
    plot_metric_trends(performance_metrics['labels'], performance_metrics)


if __name__ == '__main__':
    main()