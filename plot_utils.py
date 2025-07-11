
import matplotlib
matplotlib.use('Agg')  # ✅ Use non-GUI backend for Flask compatibility

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_charts(results, df):
    os.makedirs("static/images", exist_ok=True)

    # Pie Chart - Normal vs DDoS
    plt.figure(figsize=(6, 6))
    df['label'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Normal', 'DDoS'])
    plt.title("Normal vs DDoS Traffic")
    plt.savefig("static/images/pie_chart.png")
    plt.close()

    # Bar Chart - Accuracy Comparison
    plt.figure(figsize=(8, 5))
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    sns.barplot(x=models, y=accuracies)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/images/bar_chart.png")
    plt.close()

    # Time-Series Plot
    plt.figure(figsize=(10, 4))
    df['label_num'] = df['label'].replace({'ddos': 1, 'normal': 0})
    df['label_num'].plot()
    plt.title("Traffic Pattern Over Time")
    plt.xlabel("Row Index")
    plt.ylabel("Anomaly (1=ddos)")
    plt.tight_layout()
    plt.savefig("static/images/timeseries_plot.png")
    plt.close()

    # Feature Importance (Random Forest only)
    if 'RandomForest' in results:
        from sklearn.ensemble import RandomForestClassifier
        df['label'] = df['label'].replace({'ddos': 1, 'normal': 0}).astype(int)
        X = df.select_dtypes(include=['float64', 'int64']).drop(columns=['label'], errors='ignore')
        y = df['label']
        model = RandomForestClassifier()
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances.sort_values().plot(kind='barh', figsize=(8, 5))
        plt.title("Feature Importance (Random Forest)")
        plt.tight_layout()
        plt.savefig("static/images/feature_importance.png")
        plt.close()
