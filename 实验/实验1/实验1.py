import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os


def Fig0():

    os.makedirs('./images', exist_ok=True)

    required_cols = ['sex', 'shell weight', 'rings']
    missing_cols = [col for col in required_cols if col not in abalone.columns]
    if missing_cols:
        raise KeyError(f"缺失必要列：{missing_cols}，实际列名：{abalone.columns.tolist()}")

    print("\n=== 基础统计 ===")
    print(abalone.describe())

    numeric_cols = [col for col in abalone.columns if col != 'sex']
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()

    for i, col in enumerate(numeric_cols[:8]):  #
        axes[i].hist(abalone[col], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        axes[i].set_title(f'{col} distribution', fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.6)

    for ax in axes[len(numeric_cols):]:
        ax.remove()

    plt.tight_layout()
    plt.savefig('./images/numeric_distributions.png', dpi=300)

    plt.figure(figsize=(8, 6))
    sex_counts = abalone['sex'].value_counts()
    plt.pie(sex_counts,
            labels=['Male', 'Female', 'Infant'],
            autopct='%1.1f%%',
            colors=['lightgreen', 'gold', 'violet'])
    plt.title('Gender distribution')
    plt.savefig('./images/gender_distribution.png', dpi=300)

    plt.figure(figsize=(10, 8))
    corr_matrix = abalone.corr()
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Correlation coefficient')

    plt.xticks(ticks=range(len(corr_matrix)),
               labels=[col[:10] for col in corr_matrix.columns],  # 截断长列名
               rotation=45)
    plt.yticks(ticks=range(len(corr_matrix)),
               labels=[col[:10] for col in corr_matrix.columns])

    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                     ha='center', va='center',
                     color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')

    plt.title('Feature correlation matrix')
    plt.savefig('./images/correlation_matrix.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(12, 5))

    # plt.subplot(1, 2, 1)
    # box_data = [abalone[abalone['sex'] == 0]['rings'],
    #             abalone[abalone['sex'] == 1]['rings'],
    #             abalone[abalone['sex'] == 2]['rings']]
    # plt.boxplot(box_data,
    #             labels=['Male', 'Female', 'Infant'],
    #             patch_artist=True,
    #             boxprops=dict(facecolor='lightblue', edgecolor='darkblue'),
    #             medianprops=dict(color='red'))
    # plt.xlabel('Gender')
    # plt.ylabel('Rings')
    # plt.title('Rings distribution by gender')
    #
    # plt.subplot(1, 2, 2)
    # colors = ['green', 'gold', 'purple']
    # for gender in [0, 1, 2]:
    #     mask = abalone['sex'] == gender
    #     plt.scatter(abalone[mask]['shell weight'],
    #                 abalone[mask]['rings'],
    #                 color=colors[gender],
    #                 alpha=0.6,
    #                 label=['Male', 'Female', 'Infant'][gender])
    # plt.xlabel('Shell weight')
    # plt.ylabel('Rings')
    # plt.legend()
    # plt.title('Shell weight vs rings')
    #
    # plt.tight_layout()
    # plt.savefig('./images/target_analysis.png', dpi=300)

def Fig1(): # 轨迹图
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    feature_names = abalone.columns[:-1]
    for i in range(len(feature_names)):
        ax.plot(lambdas,
                [coef[i] for coef in coefs],
                linewidth=1.5,
                label=feature_names[i])

    ax.set_xscale('log')
    plt.xlabel(r'$\lambda$ (Regularization parameter)', fontsize=12)
    plt.ylabel('Standardized Coefficients', fontsize=12)
    plt.title('Ridge Regression Coefficient Trajectories', fontsize=14)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left',
              bbox_to_anchor=(1, 0.5),
              frameon=False,
              fontsize=10)

    ax.grid(True, linestyle='--', alpha=0.6)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.savefig('./images/ridge_trajectory_academic.png',
                dpi=300,
                bbox_inches='tight',
                transparent=True)
    plt.show()

def Fig2(): # 误差曲线
    plt.figure(figsize=(10, 6))
    ax = plt.gca()


    (line,) = ax.plot(
        lambdas,
        errors,
        color="#2b8cbe",
        linewidth=2,
        linestyle="-",
        marker="",
        markersize=4,
    )


    min_error_idx = np.argmin(errors)
    min_lambda = lambdas[min_error_idx]
    min_error = errors[min_error_idx]
    ax.scatter(
        min_lambda,
        min_error,
        color="#e41a1c",
        s=80,
        zorder=5,
        edgecolors="black",
        label=f"Minimum MSE ({min_error:.2f})",
    )


    ax.set_xscale("log")
    plt.xlabel(r"Regularization parameter ($\lambda$)", fontsize=12, labelpad=10)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12, labelpad=10)
    plt.title("Model Error vs Regularization Strength", fontsize=14, pad=20)


    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.6, which="both")


    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="black",
        fontsize=10,
    )


    ax.set_xlim(lambdas.min(), lambdas.max())
    ax.set_ylim(0.9 * min(errors), 1.1 * max(errors))


    plt.savefig(
        "./images/error_curve_academic.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()

def Fig3(): #真实值与预测值对比图
    plt.figure(figsize=(8, 6))

    best_idx = np.argmax(r2_scores)
    best_lambda = lambdas[best_idx]
    best_r2 = r2_scores[best_idx]

    best_ridge = Ridge(alpha=best_lambda)
    best_ridge.fit(x, y)
    y_pred = best_ridge.predict(x)

    rmse = np.sqrt(mean_squared_error(y, y_pred))  # 计算RMSE
    rpd = np.std(y) / rmse                         # 计算RPD（标准差/RMSE）

    scatter = plt.scatter(y, y_pred,
                          c='#1f77b4',
                          alpha=0.6,
                          edgecolors='w',
                          linewidth=0.5,
                          s=60,
                          label='Predicted Value')

    diag_line = plt.plot([y.min(), y.max()], [y.min(), y.max()],
                         color='#ff7f0e',
                         linestyle='--',
                         linewidth=3,
                         label='Perfect Fit')

    plt.text(0.05, 0.90,  # 上移所有文本
             r'Best $\lambda$ = {:.3f}'.format(best_lambda),
             transform=plt.gca().transAxes,
             fontsize=12)
    plt.text(0.05, 0.83,
             r'$R^2$ = {:.3f}'.format(best_r2),
             transform=plt.gca().transAxes,
             fontsize=12)
    plt.text(0.05, 0.76,  # 新增RMSE
             r'RMSE = {:.3f}'.format(rmse),
             transform=plt.gca().transAxes,
             fontsize=12)
    plt.text(0.05, 0.69,  # 新增RPD
             r'RPD = {:.3f}'.format(rpd),
             transform=plt.gca().transAxes,
             fontsize=12)

    plt.xlabel('True Rings', fontsize=12, labelpad=10)
    plt.ylabel('Predicted Rings', fontsize=12, labelpad=10)
    plt.title('True vs Predicted Values (Ridge Regression)', fontsize=14, pad=15)

    plt.xlim(0, 30)
    plt.ylim(0, 30)

    plt.grid(True, linestyle='--', alpha=0.4)

    plt.legend(
        loc='lower right',
        frameon=True,
        framealpha=0.8,
        edgecolor='black',
        fontsize=10
    )

    plt.savefig('./images/true_vs_pred_scatter.png',
                dpi=300,
                bbox_inches='tight',
                transparent=False)
    plt.show()

if __name__ == "__main__":
    os.makedirs('./images', exist_ok=True)

    abalone = pd.read_csv("./abalone_dataset.csv", header=0)
    print(abalone.head())

    abalone['sex'] = abalone['sex'].map( {'M': 0, 'F': 1, 'I': 2} )  # 将性别转换为数值,方便后面计算

    x = abalone.iloc[:, :-1].values
    y = abalone.iloc[:, -1].values


    scaler = StandardScaler() # 数据标准化
    x = scaler.fit_transform(x)

    lambdas = np.logspace(-3, 3, 1000) # 岭参数的设置

    coefs = [] # 存参数
    errors = [] #存误差
    r2_scores = []  # 存R²
    for l in lambdas:
        ridge = Ridge(alpha=l)
        ridge.fit(x, y)
        coefs.append(ridge.coef_)
        y_pred = ridge.predict(x)
        errors.append(mean_squared_error(y, y_pred))
        r2_scores.append(ridge.score(x, y))  # 计算并存储R²分数

    feature_names = abalone.columns[:-1]

    Fig0() # 数据集的分析
    Fig1()
    Fig2()
    Fig3()






