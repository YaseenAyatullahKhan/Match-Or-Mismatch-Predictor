import streamlit as st
import matplotlib.pyplot as plt

def show_overview(df):
    # Header
    st.title("💘 Match or Mismatch")
    st.subheader("Predicting Dating App Outcomes with Machine Learning")
    st.markdown("*WIA1006 (Machine Learning) Group Assignment — Semester 2, Session 2025/2026*")
    st.divider()

    # Project summary
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### About This Project")
        st.markdown("""
        In the age of digital relationships, modern connections evolve through patterns of 
        reply time, online presence, and social media engagement — giving rise to phenomena 
        like **ghosting** and **situationships**.

        This project, **"Tying the Data Knot: Love, Life & Likes"**, uses machine learning 
        to analyze behavioral signals from a fictional dating app and predict relationship 
        outcomes. We trained and compared **5 machine learning models** to predict whether 
        a user ends up with a Mutual Match, gets Ghosted, is Catfished, and more.
        """)

    with col2:
        st.markdown("### Dataset at a Glance")
        st.metric("Total Records", "50,000")
        st.metric("Features", "19")
        st.metric("Target Classes", "5")
        st.metric("Models Trained", "5")

    st.divider()

    # Target distribution chart
    st.markdown("### Match Outcome Distribution")

    outcome_counts = df['match_outcome'].value_counts().reset_index()
    outcome_counts.columns = ['Outcome', 'Count']

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    bars = ax.bar(outcome_counts['Outcome'], outcome_counts['Count'],
                  color=colors[:len(outcome_counts)])
    ax.set_xlabel('Match Outcome', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Match Outcomes', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, outcome_counts['Count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                str(val), ha='center', fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ML pipeline summary
    st.markdown("### Our Machine Learning Pipeline")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.info("**1. Data Collection**\n\nDating app behavior dataset — 50,000 synthetic records")
    with c2:
        st.info("**2. Preprocessing**\n\nMissing value handling, feature engineering, encoding, scaling")
    with c3:
        st.info("**3. Feature Selection**\n\nDropped redundant columns, created 7 engineered features")
    with c4:
        st.info("**4. Model Training**\n\nTrained 5 models with 80/20 train-test split")
    with c5:
        st.info("**5. Evaluation**\n\nCompared Accuracy, Weighted F1, and ROC-AUC across all models")

    st.divider()

    # Team Members
    st.markdown("### The Team")
    st.markdown("""
    | Name | Student ID |
    |------|------------|
    | Ahmad Naufal bin Azlan | 24001404 |
    | Aleeya Nazirah binti Jamil | 24001115 |
    | Alliyana binti Abdul Rahman | 24001243 |
    | Aidel Zaref bin Afifuddin | 24001654 |
    | Abdul Rizwan Mohammed | 24237402 |
    | Yaseen Ayatullah Khan | 24216496 |
    """)