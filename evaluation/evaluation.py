# evaluation/evaluate.py

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
import numpy as np


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    混同行列をプロットする関数
    Args:
        y_true (numpy.ndarray): 実際のラベル
        y_pred (numpy.ndarray): 予測されたラベル
        labels (list): クラスラベル
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def plot_roc_curve(y_true, y_prob, labels=None):
    """
    ROC曲線をプロットする関数
    Args:
        y_true (numpy.ndarray): 実際のラベル
        y_prob (numpy.ndarray): 予測確率
        labels (list): クラスラベル
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def print_classification_report(y_true, y_pred):
    """
    分類レポートを出力する関数
    Args:
        y_true (numpy.ndarray): 実際のラベル
        y_pred (numpy.ndarray): 予測されたラベル
    """
    report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(report)


def evaluate_model(model, X_val, y_val):
    """
    モデルを評価する関数
    Args:
        model: 訓練済みモデル
        X_val (numpy.ndarray): 検証データ
        y_val (numpy.ndarray): 実際のラベル
    """
    # 予測
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)

    # 評価結果
    print_classification_report(y_val, y_pred)
    plot_confusion_matrix(y_val, y_pred)
    plot_roc_curve(y_val, y_prob)


if __name__ == "__main__":
    # モデルと検証データを仮定して評価を実行
    # model = trained_model  # 訓練済みモデル
    # X_val, y_val = validation_data  # 検証データ

    # 評価実行
    # evaluate_model(model, X_val, y_val)