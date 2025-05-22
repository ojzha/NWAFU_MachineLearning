import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (precision_recall_fscore_support,
                             classification_report,
                             confusion_matrix)
from scipy.spatial.distance import cdist


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


def laplacian_kernel(X, Y=None, gamma=1.0):
    """计算拉普拉斯核矩阵 K(x, y) = exp(-gamma * ||x - y||_1)"""
    if Y is None:
        Y = X
    distances = cdist(X, Y, metric='cityblock')
    return np.exp(-gamma * distances)


def plot_class_performance(name, classes, precision, recall, f1):
    """绘制类别性能柱状图"""
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
    """绘制标准化混淆矩阵"""
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

def plot_accuracy_comparison(datasets, accuracies):
    """绘制数据集准确率对比图"""
    plt.figure()
    colors = plt.cm.tab10(np.linspace(0, 1, len(accuracies)))

    plt.bar(range(len(accuracies)), accuracies, color=colors, tick_label=list(datasets))
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Cross-dataset Accuracy Comparison')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('images/accuracy_comparison.png')
    plt.close()

def plot_metric_trends(datasets, metrics):
    """绘制指标趋势图"""
    plt.figure()
    markers = ['o', 's', 'D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (metric, color, marker) in enumerate(zip(['precision', 'recall', 'f1'],
                                                    colors, markers)):
        values = [metrics[metric][i] for i in range(len(datasets))]
        plt.plot(range(len(datasets)), values,
                 marker=marker, color=color,
                 linestyle='--', linewidth=2,
                 label=metric.capitalize())

    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.title('Performance Metric Trends')
    plt.xticks(range(len(datasets)), datasets, rotation=15)
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

    kernels = [
        ('linear', None),
        ('rbf', None),
        ('laplacian', laplacian_kernel)
    ]

    performance_metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy': [],
        'labels': []  # 存储数据集和核函数的组合标签
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

        for kernel_name, kernel_func in kernels:
            combined_name = f"{name}_{kernel_name}"
            performance_metrics['labels'].append(combined_name)

            if kernel_name == 'laplacian':
                # 计算训练和测试的拉普拉斯核矩阵
                K_train = laplacian_kernel(X_train, gamma=1.0 / X_train.shape[1])
                K_test = laplacian_kernel(X_test, X_train, gamma=1.0 / X_train.shape[1])
                clf = SVC(kernel='precomputed', C=1.0)
                clf.fit(K_train, y_train)
                y_pred = clf.predict(K_test)
            else:
                clf = SVC(kernel=kernel_name, C=1.0, gamma='scale')
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

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

    plot_accuracy_comparison(performance_metrics['labels'], performance_metrics['accuracy'])
    plot_metric_trends(performance_metrics['labels'], performance_metrics)

if __name__ == '__main__':
    main()