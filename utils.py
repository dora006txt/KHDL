# File này chứa các hàm hỗ trợ
# Các hàm được sử dụng nhiều sẽ được định nghĩa tại đây và import vào notebook

# Định nghĩa thứ tự mức độ béo phì để sử dụng thống nhất trong các biểu đồ
OBESITY_ORDER = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=None):
    """
    Hàm vẽ Confusion Matrix sử dụng seaborn heatmap.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if cmap is None:
        cmap = plt.cm.Blues

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importances, feature_names, top_n=10):
    """
    Hàm vẽ biểu đồ Feature Importance.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    # Tạo DataFrame
    feature_imp = pd.DataFrame(sorted(zip(importances, feature_names)), columns=['Value','Feature'])
    
    # Lấy top n features quan trọng nhất
    feature_imp_top = feature_imp.tail(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Value", y="Feature", data=feature_imp_top.sort_values(by="Value", ascending=False))
    plt.title(f'Top {top_n} Features Important')
    plt.tight_layout()
    plt.show()
