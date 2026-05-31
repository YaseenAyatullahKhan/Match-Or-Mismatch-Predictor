import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder

def show_predictor(df, models, scaler, target_enc, model_columns):
    st.title("💘 Predict My Outcome")
    st.markdown("Fill in your dating profile below and see what our models predict for you.")
    st.divider()

    from sklearn.preprocessing import OrdinalEncoder
    from collections import Counter

    # User input form
    st.markdown("### Your Dating Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Demographics")
        age             = st.slider("Age", 18, 60, 25)
        gender          = st.selectbox("Gender", df['gender'].unique())
        sexual_orient   = st.selectbox("Sexual Orientation", df['sexual_orientation'].unique())
        location_type   = st.selectbox("Location Type", df['location_type'].unique())
        income_bracket  = st.selectbox("Income Bracket",
                                       ['Very Low', 'Low', 'Lower-Middle',
                                        'Middle', 'Upper-Middle', 'High', 'Very High'])
        education_level = st.selectbox("Education Level", df['education_level'].unique())
        height_cm       = st.slider("Height (cm)", 140, 210, 170)
        weight_kg       = st.slider("Weight (kg)", 40, 150, 70)
        zodiac_sign     = st.selectbox("Zodiac Sign", df['zodiac_sign'].unique())
        body_type       = st.selectbox("Body Type", df['body_type'].unique())

    with col2:
        st.markdown("#### App Behaviour")
        app_usage_time_min  = st.slider("Daily App Usage (mins)", 0, 300, 60)
        swipe_right_ratio   = st.slider("Swipe Right Ratio", 0.0, 1.0, 0.5)
        likes_received      = st.slider("Likes Received", 0, 500, 50)
        mutual_matches      = st.slider("Mutual Matches", 0, 100, 10)
        message_sent_count  = st.slider("Messages Sent", 0, 500, 50)
        emoji_usage_rate    = st.slider("Emoji Usage Rate", 0.0, 1.0, 0.5)
        profile_pics_count  = st.slider("Profile Pics", 1, 6, 3)
        bio_length          = st.slider("Bio Length (chars)", 0, 500, 150)
        last_active_hour    = st.slider("Last Active Hour (0-23)", 0, 23, 12)
        swipe_time_of_day   = st.selectbox("Swipe Time of Day", df['swipe_time_of_day'].unique())

    with col3:
        st.markdown("#### Preferences")
        relationship_intent = st.selectbox("Relationship Intent", df['relationship_intent'].unique())
        interest_tags       = st.multiselect("Interests (pick a few)",
                                              ['Music', 'Travel', 'Fitness', 'Gaming',
                                               'Cooking', 'Reading', 'Movies', 'Art',
                                               'Sports', 'Technology'],
                                              default=['Cooking', 'Travel'])
        model_choice        = st.selectbox("Choose Model to Predict With",
                                           list(models.keys()))

    st.divider()

    # Predict button
    predict_clicked = st.button("💘 Predict My Outcome", use_container_width=True)

    if predict_clicked:
        from collections import Counter
        from sklearn.preprocessing import OrdinalEncoder

        # Append user input as a temporary row to the real df
        # This way it goes through the exact same pipeline as training
        user_interests_str = ', '.join(interest_tags) if interest_tags else 'Cooking'

        # Infer app_usage_time_label and swipe_right_label from dataset
        app_usage_label = df[df['app_usage_time_min'] <= app_usage_time_min]['app_usage_time_label'].mode()
        app_usage_label = app_usage_label[0] if len(app_usage_label) > 0 else df['app_usage_time_label'].mode()[0]
        swipe_right_label = df[df['swipe_right_ratio'] <= swipe_right_ratio]['swipe_right_label'].mode()
        swipe_right_label = swipe_right_label[0] if len(swipe_right_label) > 0 else df['swipe_right_label'].mode()[0]

        user_row = {
            'age':                  age,
            'gender':               gender,
            'sexual_orientation':   sexual_orient,
            'location_type':        location_type,
            'income_bracket':       income_bracket,
            'education_level':      education_level,
            'height_cm':            height_cm,
            'weight_kg':            weight_kg,
            'zodiac_sign':          zodiac_sign,
            'body_type':            body_type,
            'app_usage_time_min':   app_usage_time_min,
            'app_usage_time_label': app_usage_label,
            'swipe_right_ratio':    swipe_right_ratio,
            'swipe_right_label':    swipe_right_label,
            'likes_received':       likes_received,
            'mutual_matches':       mutual_matches,
            'message_sent_count':   message_sent_count,
            'emoji_usage_rate':     emoji_usage_rate,
            'profile_pics_count':   profile_pics_count,
            'bio_length':           bio_length,
            'last_active_hour':     last_active_hour,
            'swipe_time_of_day':    swipe_time_of_day,
            'relationship_intent':  relationship_intent,
            'interest_tags':        user_interests_str,
            'match_outcome':        df['match_outcome'].iloc[0],  # placeholder, gets dropped later
        }

        # Add user row to a copy of the full dataframe
        df_with_user = pd.concat([df, pd.DataFrame([user_row])], ignore_index=True)
        user_idx = len(df_with_user) - 1

        # Run exact same preprocessing pipeline
        df_proc = df_with_user.copy()

        df_proc['swipe_to_match_ratio'] = df_proc['swipe_right_ratio'] / (df_proc['mutual_matches'] + 1)
        df_proc['engagement_score'] = (
            df_proc['app_usage_time_min'] / 100 +
            df_proc['message_sent_count'] / 50 +
            df_proc['emoji_usage_rate'] * 10 +
            df_proc['swipe_right_ratio'] * 5
        )
        df_proc['match_success_rate'] = df_proc['mutual_matches'] / (df_proc['likes_received'] + 1)
        df_proc['profile_completeness'] = (
            (df_proc['profile_pics_count'] / 6) * 0.3 +
            (df_proc['bio_length'] / 500) * 0.3 +
            (df_proc['interest_tags'].apply(lambda x: len(str(x).split(','))) / 10) * 0.4
        )
        df_proc['is_late_night'] = (
            df_proc['last_active_hour'].between(0, 5) |
            df_proc['last_active_hour'].between(22, 23)
        ).astype(int)
        df_proc['age_group'] = pd.cut(
            df_proc['age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '55+']
        )
        df_proc['bmi'] = df_proc['weight_kg'] / ((df_proc['height_cm'] / 100) ** 2)
        df_proc['bmi_category'] = pd.cut(
            df_proc['bmi'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )

        # Interest tags
        all_interests = []
        for tags in df_proc['interest_tags']:
            all_interests.extend(str(tags).split(', '))
        top_interests = [i for i, _ in Counter(all_interests).most_common(10)]
        df_proc['interest_count'] = df_proc['interest_tags'].apply(
            lambda x: len(str(x).split(',')))
        for interest in top_interests:
            df_proc[f'interest_{interest.lower().replace(" ", "_")}'] = df_proc['interest_tags'].apply(
                lambda x: 1 if interest in str(x) else 0
            )

        # Ordinal encoding
        income_order = ['Very Low', 'Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High', 'Very High']
        oe = OrdinalEncoder(categories=[income_order], handle_unknown='use_encoded_value', unknown_value=-1)
        df_proc['income_bracket_encoded'] = oe.fit_transform(df_proc[['income_bracket']])

        # One-hot encoding
        nominal_features = ['gender', 'sexual_orientation', 'location_type', 'education_level',
                            'zodiac_sign', 'body_type', 'relationship_intent', 'swipe_time_of_day',
                            'app_usage_time_label', 'swipe_right_label', 'age_group', 'bmi_category']
        df_proc = pd.get_dummies(df_proc, columns=nominal_features, prefix=nominal_features)

        # Drop same columns as training
        columns_to_drop = ['interest_tags', 'age', 'height_cm', 'weight_kg',
                        'app_usage_time_label', 'swipe_right_label', 'match_outcome',
                        'income_bracket']
        original_nominal = ['gender', 'sexual_orientation', 'location_type', 'education_level',
                            'zodiac_sign', 'body_type', 'relationship_intent', 'swipe_time_of_day',
                            'app_usage_time_label', 'swipe_right_label', 'age_group', 'bmi_category']
        for col in original_nominal:
            if col in df_proc.columns and col not in columns_to_drop:
                columns_to_drop.append(col)

        df_proc = df_proc.drop(columns=[c for c in columns_to_drop if c in df_proc.columns])
        non_numeric = df_proc.select_dtypes(include=['object']).columns.tolist()
        if non_numeric:
            df_proc = df_proc.drop(columns=non_numeric)

        # Extract just the user row and align to model columns
        user_df = df_proc.iloc[[user_idx]].copy()
        for col in model_columns:
            if col not in user_df.columns:
                user_df[col] = 0
        user_df = user_df[model_columns]

        # Scale
        numeric_cols = user_df.select_dtypes(include=['int64', 'float64']).columns
        user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])

        # Predict 
        selected_model   = models[model_choice]
        prediction_enc   = selected_model.predict(user_df)[0]
        prediction_label = target_enc.inverse_transform([prediction_enc])[0]
        probabilities    = selected_model.predict_proba(user_df)[0]
        confidence       = probabilities.max() * 100

        # Result display
        st.divider()
        st.markdown("### 🔮 Prediction Result")

        outcome_emoji = {
            "Mutual Match": "💚",
            "Ghosted":      "👻",
            "Catfished":    "🎣",
            "Just Friends": "🤝",
            "Blocked":      "🚫",
        }
        emoji = outcome_emoji.get(prediction_label, "💘")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if prediction_label == "Mutual Match":
                st.success(f"## {emoji} {prediction_label}")
            elif prediction_label in ["Ghosted", "Catfished", "Blocked"]:
                st.error(f"## {emoji} {prediction_label}")
            else:
                st.info(f"## {emoji} {prediction_label}")

            st.markdown(f"**Model used:** {model_choice}")
            st.markdown(f"**Confidence:** {confidence:.1f}%")
            st.progress(min(int(confidence), 100))

        st.divider()

        # Probability breakdown
        st.markdown("### Probability Breakdown")
        st.markdown("How confident the model is across all possible outcomes:")

        prob_df = pd.DataFrame({
            'Outcome':     target_enc.classes_,
            'Probability': probabilities
        }).sort_values('Probability', ascending=False)

        fig, ax = plt.subplots(figsize=(14, 5))
        colors = ['#FF6B6B' if o == prediction_label else '#4ECDC4' for o in prob_df['Outcome']]
        bars = ax.bar(prob_df['Outcome'], prob_df['Probability'],
                    color=colors, edgecolor='white')
        ax.set_ylabel("Probability", fontsize=11)
        ax.set_title("Predicted Probability per Outcome", fontweight='bold')
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, prob_df['Probability']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2%}", ha='center', fontsize=10, fontweight='bold')
        plt.xticks(rotation=30, ha='right', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()