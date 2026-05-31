import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_eda(df):
    st.title("🔍 Data Explorer")
    st.markdown("Explore the dating app dataset through interactive visualizations.")
    st.divider()

    # Raw data preview
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.divider()

    # Feature distribution
    st.markdown("### Feature Distributions")
    st.markdown("Select a feature to see how it is distributed across the dataset.")

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Numerical Feature")
        num_feature = st.selectbox("Select numerical feature", numerical_cols)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[num_feature].dropna(), bins=30, color='#4ECDC4', edgecolor='white')
        ax.set_xlabel(num_feature, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Distribution of {num_feature}", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Categorical Feature")
        cat_feature = st.selectbox("Select categorical feature", categorical_cols)
        fig, ax = plt.subplots(figsize=(8, 4))
        counts = df[cat_feature].value_counts()
        ax.bar(counts.index, counts.values, color='#FF6B6B', edgecolor='white')
        ax.set_xlabel(cat_feature, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Distribution of {cat_feature}", fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Outcome breakdown
    st.markdown("### Feature vs Match Outcome")
    st.markdown("See how a selected feature varies across different match outcomes.")

    feature_vs_outcome = st.selectbox(
        "Select feature to compare across outcomes",
        numerical_cols,
        key="outcome_compare"
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    outcomes = df['match_outcome'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for i, outcome in enumerate(outcomes):
        subset = df[df['match_outcome'] == outcome][feature_vs_outcome].dropna()
        ax.hist(subset, bins=25, alpha=0.6, label=outcome, color=colors[i % len(colors)])
    ax.set_xlabel(feature_vs_outcome, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{feature_vs_outcome} by Match Outcome", fontweight='bold')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Correlation heatmap
    st.markdown("### Correlation Heatmap")
    st.markdown("Pearson correlation between all numerical features.")

    fig, ax = plt.subplots(figsize=(14, 10))
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        annot_kws={"size": 7},
        linewidths=0.5
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Top correlations with outcome
    st.markdown("### Strongest Predictors of Match Outcome")
    st.markdown("Numerical features most correlated with each other — useful for understanding relationships in the data.")

    corr_pairs = (
        corr_matrix
        .abs()
        .unstack()
        .reset_index()
    )
    corr_pairs.columns = ['Feature A', 'Feature B', 'Correlation']
    corr_pairs = (
        corr_pairs[corr_pairs['Feature A'] != corr_pairs['Feature B']]
        .drop_duplicates(subset='Correlation')
        .sort_values('Correlation', ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    st.dataframe(corr_pairs, use_container_width=True)