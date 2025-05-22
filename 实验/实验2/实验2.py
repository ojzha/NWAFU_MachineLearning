import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def lr_cost_function(theta, X, y, lambda_=0.1):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-1 / m) * np.sum(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))
    reg = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    grad = (1 / m) * (X.T @ (h - y)) + (lambda_ / m) * np.r_[[0], theta[1:]]
    return cost + reg, grad



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
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    plt.bar(datasets.keys(), accuracies, color=colors)
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
        plt.plot(datasets.keys(), values,
                 marker=marker, color=color,
                 linestyle='--', linewidth=2,
                 label=metric.capitalize())

    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.title('Performance Metric Trends')
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

    performance_metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy': []
    }

    for name, path in datasets.items():
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
        X_train = np.c_[np.ones(len(X_train)), X_train]
        X_test = np.c_[np.ones(len(X_test)), X_test]

        # 模型训练
        theta = np.zeros((X_train.shape[1], len(classes)))
        for i, cls in enumerate(classes):
            y_i = (y_train == cls).astype(int)
            t = np.zeros(X_train.shape[1])
            for _ in range(300):
                cost, grad = lr_cost_function(t, X_train, y_i)
                t -= 0.1 * grad
            theta[:, i] = t

        y_pred = classes[np.argmax(sigmoid(X_test @ theta), axis=1)]
        accuracy = np.mean(y_pred == y_test) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

        performance_metrics['precision'].append(np.mean(precision))
        performance_metrics['recall'].append(np.mean(recall))
        performance_metrics['f1'].append(np.mean(f1))
        performance_metrics['accuracy'].append(accuracy)

        plot_class_performance(name, classes, precision, recall, f1)
        plot_confusion_matrix(name, y_test, y_pred, classes)

        print(f"\n=== {name} Classification Report ===")
        print(classification_report(y_test, y_pred, zero_division=0))

    plot_accuracy_comparison(datasets, performance_metrics['accuracy'])
    plot_metric_trends(datasets, performance_metrics)


if __name__ == '__main__':
    main()