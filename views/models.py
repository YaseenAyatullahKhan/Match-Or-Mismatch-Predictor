import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix)

def show_models(df, models, scaler, target_enc, model_columns, RF_STATS, KNN_STATS):
    st.title("📊 Model Performance")
    st.markdown("Compare the performance of all trained models side by side.")
    st.divider()

    # Rebuild X_test_scaled for live evaluation
    # We need to reproduce the same preprocessing pipeline from the notebook
    df_processed = df.copy()

    # Feature engineering
    df_processed['swipe_to_match_ratio'] = df_processed['swipe_right_ratio'] / (df_processed['mutual_matches'] + 1)
    df_processed['engagement_score'] = (
        df_processed['app_usage_time_min'] / 100 +
        df_processed['message_sent_count'] / 50 +
        df_processed['emoji_usage_rate'] * 10 +
        df_processed['swipe_right_ratio'] * 5
    )
    df_processed['match_success_rate'] = df_processed['mutual_matches'] / (df_processed['likes_received'] + 1)
    df_processed['profile_completeness'] = (
        (df_processed['profile_pics_count'] / 6) * 0.3 +
        (df_processed['bio_length'] / 500) * 0.3 +
        (df_processed['interest_tags'].apply(lambda x: len(str(x).split(','))) / 10) * 0.4
    )
    df_processed['is_late_night'] = (
        df_processed['last_active_hour'].between(0, 5) |
        df_processed['last_active_hour'].between(22, 23)
    ).astype(int)
    df_processed['age_group'] = pd.cut(
        df_processed['age'],
        bins=[0, 25, 35, 45, 55, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '55+']
    )
    df_processed['bmi'] = df_processed['weight_kg'] / ((df_processed['height_cm'] / 100) ** 2)
    df_processed['bmi_category'] = pd.cut(
        df_processed['bmi'],
        bins=[0, 18.5, 25, 30, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )

    # Interest tags
    from collections import Counter
    from sklearn.preprocessing import OrdinalEncoder
    df_processed['interest_count'] = df_processed['interest_tags'].apply(lambda x: len(str(x).split(',')))
    all_interests = []
    for tags in df_processed['interest_tags']:
        all_interests.extend(str(tags).split(', '))
    top_interests = [i for i, _ in Counter(all_interests).most_common(10)]
    for interest in top_interests:
        df_processed[f'interest_{interest.lower().replace(" ", "_")}'] = df_processed['interest_tags'].apply(
            lambda x: 1 if interest in str(x) else 0
        )

    # Ordinal encoding
    income_order = ['Very Low', 'Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High', 'Very High']
    oe = OrdinalEncoder(categories=[income_order], handle_unknown='use_encoded_value', unknown_value=-1)
    df_processed['income_bracket_encoded'] = oe.fit_transform(df_processed[['income_bracket']])

    # One-hot encoding
    nominal_features = ['gender', 'sexual_orientation', 'location_type', 'education_level',
                        'zodiac_sign', 'body_type', 'relationship_intent', 'swipe_time_of_day',
                        'app_usage_time_label', 'swipe_right_label', 'age_group', 'bmi_category']
    df_processed = pd.get_dummies(df_processed, columns=nominal_features, prefix=nominal_features)

    # Drop same columns as notebook
    columns_to_drop = ['interest_tags', 'age', 'height_cm', 'weight_kg',
                       'app_usage_time_label', 'swipe_right_label']
    original_nominal = ['gender', 'sexual_orientation', 'location_type', 'education_level',
                        'zodiac_sign', 'body_type', 'relationship_intent', 'swipe_time_of_day',
                        'app_usage_time_label', 'swipe_right_label', 'age_group', 'bmi_category']
    for col in original_nominal:
        if col in df_processed.columns and col not in columns_to_drop:
            columns_to_drop.append(col)

    df_final = df_processed.drop(columns=[c for c in columns_to_drop if c in df_processed.columns])

    target = 'match_outcome'
    X = df_final.drop(columns=[target])
    y = target_enc.transform(df_final[target])

    non_numeric = X.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Add any missing columns as 0, drop any extras, enforce exact training order
    for col in model_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[model_columns]

    numeric_features = X_test.select_dtypes(include=['int64', 'float64']).columns
    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

    # Compute live metrics for loaded models
    live_results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='weighted')
        try:
            y_prob = model.predict_proba(X_test_scaled)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        except:
            auc = None
        live_results[name] = {
            'Accuracy': acc,
            'Weighted F1 Score': f1,
            'ROC-AUC': auc,
            'y_pred': y_pred
        }

    live_results["Random Forest"] = {
        'Accuracy':          RF_STATS['Accuracy'],
        'Weighted F1 Score': RF_STATS['Weighted F1 Score'],
        'ROC-AUC':           RF_STATS['ROC-AUC'],
        'y_pred':            None  # no pkl for rf so no confusion matrix
    }

    live_results["K-Nearest Neighbors"] = {
        'Accuracy':          KNN_STATS['Accuracy'],
        'Weighted F1 Score': KNN_STATS['Weighted F1 Score'],
        'ROC-AUC':           KNN_STATS['ROC-AUC'],
        'y_pred':            None   # no pkl for knn so no confusion matrix
    }

    # Metrics table
    st.markdown("### Model Comparison Table")

    rows = []
    best_f1 = max(v['Weighted F1 Score'] for v in live_results.values())
    for name, metrics in live_results.items():
        is_best = "⭐ " if metrics['Weighted F1 Score'] == best_f1 else ""
        rows.append({
            "Model":            f"{is_best}{name}",
            "Accuracy":         f"{metrics['Accuracy']:.4f}",
            "Weighted F1":      f"{metrics['Weighted F1 Score']:.4f}",
            "ROC-AUC":          f"{metrics['ROC-AUC']:.4f}" if metrics['ROC-AUC'] else "N/A",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("⭐ = best performing model by Weighted F1 Score")

    st.divider()

    # Bar chart comparison
    st.markdown("### Visual Comparison")

    metric_choice = st.selectbox("Select metric to visualise", 
                                 ["Accuracy", "Weighted F1 Score", "ROC-AUC"])

    model_names  = list(live_results.keys())
    metric_vals  = []
    for name in model_names:
        val = live_results[name][metric_choice]
        metric_vals.append(val if val is not None else 0)

    best_val = max(metric_vals)
    bar_colors = ['#FF6B6B' if v == best_val else '#4ECDC4' for v in metric_vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(model_names, metric_vals, color=bar_colors, edgecolor='white')
    ax.set_ylabel(metric_choice, fontsize=12)
    ax.set_title(f"Model Comparison — {metric_choice}", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha='center', fontsize=10, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("🔴 Red bar = best performing model for selected metric")

    st.divider()

    # Per model deep dive
    st.markdown("### Model Deep Dive")

    selected_model = st.selectbox("Select a model to inspect",
                                  [n for n in live_results if live_results[n]['y_pred'] is not None])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### Confusion Matrix — {selected_model}")
        y_pred_sel = live_results[selected_model]['y_pred']
        cm = confusion_matrix(y_test, y_pred_sel)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_enc.classes_,
            yticklabels=target_enc.classes_,
            ax=ax
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title(f"Confusion Matrix", fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown(f"#### Classification Report — {selected_model}")
        report = classification_report(
            y_test, y_pred_sel,
            target_names=target_enc.classes_,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)