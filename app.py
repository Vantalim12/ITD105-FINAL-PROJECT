"""
SDG 14: Fish Species Conservation Status Prediction
Multi-Page Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SDG 14: Fish Conservation Predictor",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0277BD;
        margin-top: 1rem;
    }
    .metric-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .prediction-good {
        background-color: #C8E6C9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .prediction-moderate {
        background-color: #FFF9C4;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FDD835;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .prediction-poor {
        background-color: #FFCDD2;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #F44336;
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('fish_conservation_data.csv')
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def load_models():
    try:
        with open('best_model.pkl', 'rb') as f:
            best_model_data = pickle.load(f)
        with open('all_models.pkl', 'rb') as f:
            all_models_data = pickle.load(f)
        return best_model_data, all_models_data
    except FileNotFoundError:
        return None, None

# Sidebar navigation
st.sidebar.title("🐟 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["SDG 14 Overview", "Dataset Explorer", "Model Comparison", "Prediction Tool", "Conclusion & Impact"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**ITD105 - Big Data Analytics**\n\n"
    "Case Study Project\n\n"
    "Aligned with UN SDG 14:\n"
    "Life Below Water 🌊"
)

# Load data
df = load_data()
best_model_data, all_models_data = load_models()

# PAGE 1: SDG 14 Overview
if page == "SDG 14 Overview":
    st.markdown('<h1 class="main-header">🌊 SDG 14: Life Below Water</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Fish Species Conservation Status Prediction System</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 What is SDG 14?")
        st.write("""
        **Sustainable Development Goal 14** aims to conserve and sustainably use the oceans, 
        seas, and marine resources for sustainable development. The health of our oceans is 
        critical to the survival of all life on Earth.
        """)
        
        st.markdown("### 🐠 Key Challenges")
        st.write("""
        - **Overfishing**: Unsustainable fishing practices deplete fish populations
        - **Habitat Loss**: Coral reef degradation and coastal ecosystem destruction
        - **Climate Change**: Ocean warming and acidification affect marine life
        - **Pollution**: Plastic waste and chemical pollutants harm marine species
        """)
        
        st.markdown("### 💡 Project Motivation")
        st.write("""
        This project uses **machine learning** to predict the conservation status of fish species 
        based on various environmental and biological factors. By identifying species at risk, 
        we can:
        
        - Support data-driven conservation decisions
        - Raise awareness about threatened species
        - Help prioritize marine protection efforts
        - Contribute to sustainable fishing practices
        """)
    
    with col2:
        st.image("https://www.un.org/sustainabledevelopment/wp-content/uploads/2019/08/E-Inverted-Icons_WEB-14.png", 
                 caption="UN SDG 14", use_column_width=True)
        
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Target Species Analyzed", "1000+")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Conservation Categories", "3")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("ML Models Compared", "5")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔬 How This System Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **1. Data Collection**
        
        Gather information about fish species including habitat, 
        population trends, fishing pressure, and biological characteristics.
        """)
    
    with col2:
        st.info("""
        **2. Machine Learning**
        
        Train multiple classification algorithms to predict conservation 
        status: Good, Moderate, or Poor.
        """)
    
    with col3:
        st.info("""
        **3. Prediction & Action**
        
        Use the best model to assess species risk and support 
        conservation planning decisions.
        """)
    
    st.markdown("---")
    st.markdown("### 👥 Project Team")
    
    st.write("""
    This project was developed by the following team members:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        - **Brianna Glaze Barnett**
        - **Jasper Gumora**
        - **Luis Marco Quilantang**
        """)
    
    with col2:
        st.write("""
        - **Maria Vianell Tadoy**
        - **Jubil Leo Ventic**
        """)
    
    st.markdown("---")
    st.markdown("**Course:** ITD105 - Big Data Analytics | **Project:** Case Study - ML Web Application | **Focus:** UN SDG 14 - Life Below Water")

# PAGE 2: Dataset Explorer
elif page == "Dataset Explorer":
    st.markdown('<h1 class="main-header">📊 Dataset Explorer</h1>', unsafe_allow_html=True)
    
    if df is None:
        st.error("⚠️ Dataset not found. Please run 'generate_dataset.py' first.")
    else:
        st.success(f"✅ Dataset loaded successfully! Total samples: **{len(df)}**")
        
        # Dataset overview
        st.markdown("### 📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", len(df))
        col2.metric("Features", len(df.columns) - 2)  # Exclude species_id and target
        col3.metric("Categories", df['conservation_status'].nunique())
        col4.metric("Missing Values", df.isnull().sum().sum())
        
        st.markdown("---")
        
        # Interactive data table
        st.markdown("### 🔍 Interactive Data Table")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_status = st.multiselect(
                "Filter by Conservation Status",
                options=df['conservation_status'].unique(),
                default=df['conservation_status'].unique()
            )
        with col2:
            num_rows = st.slider("Number of rows to display", 5, 50, 10)
        
        filtered_df = df[df['conservation_status'].isin(search_status)]
        st.dataframe(filtered_df.head(num_rows), use_container_width=True)
        
        st.markdown("---")
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        
        tab1, tab2 = st.tabs(["Numerical Features", "Categorical Features"])
        
        with tab1:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            st.dataframe(df[numerical_cols].describe(), use_container_width=True)
        
        with tab2:
            categorical_cols = ['habitat_type', 'population_trend', 'fishing_pressure', 
                              'geographic_region', 'conservation_status']
            for col in categorical_cols:
                st.write(f"**{col.replace('_', ' ').title()}**")
                st.write(df[col].value_counts())
                st.write("")
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### 📊 Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Conservation status distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            status_counts = df['conservation_status'].value_counts()
            colors = {'Good': '#4CAF50', 'Moderate': '#FDD835', 'Poor': '#F44336'}
            bars = ax.bar(status_counts.index, status_counts.values, 
                         color=[colors[x] for x in status_counts.index])
            ax.set_xlabel('Conservation Status', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title('Conservation Status Distribution', fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Fishing pressure distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            fishing_counts = df['fishing_pressure'].value_counts()
            ax.pie(fishing_counts.values, labels=fishing_counts.index, autopct='%1.1f%%',
                  startangle=90, colors=sns.color_palette("RdYlGn_r", len(fishing_counts)))
            ax.set_title('Fishing Pressure Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Population trend by conservation status
            fig, ax = plt.subplots(figsize=(8, 6))
            pd.crosstab(df['population_trend'], df['conservation_status']).plot(
                kind='bar', stacked=True, ax=ax, 
                color=['#4CAF50', '#FDD835', '#F44336']
            )
            ax.set_xlabel('Population Trend', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title('Population Trend vs Conservation Status', fontsize=14, fontweight='bold')
            ax.legend(title='Status')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Habitat type distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            habitat_counts = df['habitat_type'].value_counts()
            ax.barh(habitat_counts.index, habitat_counts.values, color=sns.color_palette("viridis", len(habitat_counts)))
            ax.set_xlabel('Count', fontsize=12, fontweight='bold')
            ax.set_ylabel('Habitat Type', fontsize=12, fontweight='bold')
            ax.set_title('Habitat Type Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

# PAGE 3: Model Comparison Dashboard
elif page == "Model Comparison":
    st.markdown('<h1 class="main-header">🤖 Model Comparison Dashboard</h1>', unsafe_allow_html=True)
    
    if all_models_data is None:
        st.error("⚠️ Model data not found. Please run 'train_models.py' first.")
    else:
        results_df = all_models_data['results']
        models = all_models_data['models']
        X_test = all_models_data['X_test']
        y_test = all_models_data['y_test']
        target_encoder = all_models_data['target_encoder']
        
        st.success("✅ Model comparison data loaded successfully!")
        
        # Performance summary
        st.markdown("### 📊 Model Performance Summary")
        
        # Highlight best model
        best_model_name = results_df.iloc[0]['Model']
        st.info(f"🏆 **Best Performing Model:** {best_model_name} with {results_df.iloc[0]['Accuracy']:.2%} accuracy")
        
        # Display results table
        st.dataframe(
            results_df.style.format({
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1-Score': '{:.4f}'
            }).background_gradient(subset=['Accuracy', 'F1-Score'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Accuracy comparison chart
        st.markdown("### 📈 Accuracy Comparison")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(results_df['Model'], results_df['Accuracy'], 
                     color=sns.color_palette("viridis", len(results_df)))
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Detailed metrics comparison
        st.markdown("### 📊 Detailed Metrics Comparison")
        
        # Prepare data for grouped bar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(results_df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            offset = width * (i - 1.5)
            ax.bar(x + offset, results_df[metric], width, label=metric)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Comprehensive Model Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Confusion matrices
        st.markdown("### 🔍 Confusion Matrices")
        
        selected_model = st.selectbox("Select a model to view confusion matrix", results_df['Model'].tolist())
        
        model = models[selected_model]
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=target_encoder.classes_,
                       yticklabels=target_encoder.classes_,
                       ax=ax)
            ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
            ax.set_title(f'Confusion Matrix: {selected_model}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Classification report
            st.markdown(f"**Classification Report: {selected_model}**")
            report = classification_report(y_test, y_pred, 
                                          target_names=target_encoder.classes_,
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format('{:.4f}'), use_container_width=True)

# PAGE 4: Prediction Tool
elif page == "Prediction Tool":
    st.markdown('<h1 class="main-header">🔮 Fish Conservation Status Predictor</h1>', unsafe_allow_html=True)
    
    if best_model_data is None:
        st.error("⚠️ Model not found. Please run 'train_models.py' first.")
    else:
        st.success("✅ Prediction model loaded and ready!")
        
        st.markdown("### 🐟 Enter Fish Species Characteristics")
        st.write("Provide information about the fish species to predict its conservation status.")
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            habitat_type = st.selectbox(
                "Habitat Type",
                ['Coral Reef', 'Deep Sea', 'Coastal', 'Open Ocean', 'Estuary']
            )
            
            population_trend = st.selectbox(
                "Population Trend",
                ['Increasing', 'Stable', 'Declining', 'Critical']
            )
            
            fishing_pressure = st.selectbox(
                "Fishing Pressure Level",
                ['Low', 'Moderate', 'High', 'Very High']
            )
            
            geographic_region = st.selectbox(
                "Geographic Region",
                ['Pacific', 'Atlantic', 'Indian Ocean', 'Arctic', 'Mediterranean']
            )
            
            average_size_cm = st.slider(
                "Average Size (cm)",
                min_value=10.0,
                max_value=200.0,
                value=50.0,
                step=1.0
            )
        
        with col2:
            reproduction_rate = st.slider(
                "Reproduction Rate (offspring per year)",
                min_value=0.5,
                max_value=10.0,
                value=5.0,
                step=0.1
            )
            
            depth_range_m = st.slider(
                "Depth Range (meters)",
                min_value=5.0,
                max_value=2000.0,
                value=100.0,
                step=10.0
            )
            
            water_temperature_c = st.slider(
                "Water Temperature (°C)",
                min_value=5.0,
                max_value=30.0,
                value=20.0,
                step=0.5
            )
            
            population_size_thousands = st.slider(
                "Population Size (thousands)",
                min_value=1.0,
                max_value=500.0,
                value=100.0,
                step=5.0
            )
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔍 Predict Conservation Status", type="primary", use_container_width=True):
            # Prepare input data
            input_data = pd.DataFrame({
                'habitat_type': [habitat_type],
                'population_trend': [population_trend],
                'fishing_pressure': [fishing_pressure],
                'average_size_cm': [average_size_cm],
                'geographic_region': [geographic_region],
                'reproduction_rate': [reproduction_rate],
                'depth_range_m': [depth_range_m],
                'water_temperature_c': [water_temperature_c],
                'population_size_thousands': [population_size_thousands]
            })
            
            # Encode categorical variables
            for col in best_model_data['categorical_columns']:
                input_data[col] = best_model_data['label_encoders'][col].transform(input_data[col])
            
            # Scale features
            input_scaled = best_model_data['scaler'].transform(input_data)
            
            # Make prediction
            prediction = best_model_data['model'].predict(input_scaled)
            prediction_proba = best_model_data['model'].predict_proba(input_scaled)
            
            # Decode prediction
            status = best_model_data['target_encoder'].inverse_transform(prediction)[0]
            
            # Display result
            st.markdown("### 📊 Prediction Result")
            
            if status == 'Good':
                st.markdown(f'<div class="prediction-good">🟢 Conservation Status: GOOD<br>This species has a healthy and sustainable population.</div>', unsafe_allow_html=True)
            elif status == 'Moderate':
                st.markdown(f'<div class="prediction-moderate">🟡 Conservation Status: MODERATE<br>This species shows warning signs and requires monitoring.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-poor">🔴 Conservation Status: POOR<br>This species is at high risk and requires immediate conservation efforts!</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Prediction confidence
            st.markdown("### 📈 Prediction Confidence")
            
            col1, col2, col3 = st.columns(3)
            
            classes = best_model_data['target_encoder'].classes_
            
            for i, cls in enumerate(classes):
                if cls == 'Good':
                    col1.metric("Good", f"{prediction_proba[0][i]:.2%}")
                elif cls == 'Moderate':
                    col2.metric("Moderate", f"{prediction_proba[0][i]:.2%}")
                else:
                    col3.metric("Poor", f"{prediction_proba[0][i]:.2%}")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 4))
            colors_map = {'Good': '#4CAF50', 'Moderate': '#FDD835', 'Poor': '#F44336'}
            colors = [colors_map[cls] for cls in classes]
            bars = ax.barh(classes, prediction_proba[0], color=colors)
            ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
            ax.set_title('Prediction Confidence by Category', fontsize=14, fontweight='bold')
            ax.set_xlim([0, 1])
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{prediction_proba[0][i]:.2%}',
                       ha='left', va='center', fontweight='bold', fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Recommendations
            st.markdown("---")
            st.markdown("### 💡 Recommendations")
            
            if status == 'Good':
                st.success("""
                **Maintain Current Practices:**
                - Continue monitoring population trends
                - Maintain sustainable fishing practices
                - Protect critical habitats
                - Share successful conservation strategies
                """)
            elif status == 'Moderate':
                st.warning("""
                **Increased Monitoring Required:**
                - Implement stricter fishing quotas
                - Monitor population trends more frequently
                - Assess habitat quality and threats
                - Consider protective measures if decline continues
                """)
            else:
                st.error("""
                **Urgent Conservation Action Needed:**
                - Implement immediate fishing restrictions or bans
                - Establish marine protected areas
                - Conduct emergency population assessment
                - Develop species recovery plan
                - Increase public awareness and enforcement
                """)

# PAGE 5: Conclusion & Impact
elif page == "Conclusion & Impact":
    st.markdown('<h1 class="main-header">🎯 Conclusion & Impact</h1>', unsafe_allow_html=True)
    
    if all_models_data is not None:
        results_df = all_models_data['results']
        best_model_name = results_df.iloc[0]['Model']
        best_accuracy = results_df.iloc[0]['Accuracy']
        
        st.markdown("### 🏆 Best Performing Model")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f'<div class="metric-box">', unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color: #1E88E5;'>{best_model_name}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; color: #0277BD;'>Accuracy: {best_accuracy:.2%}</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
    
    st.markdown("### 🔍 Project Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Key Achievements")
        st.write("""
        - **Successfully compared 5 classification algorithms** for fish conservation status prediction
        - **Developed an interactive web application** to make ML predictions accessible
        - **Integrated real-world features** including habitat, population trends, and fishing pressure
        - **Created a user-friendly interface** for conservation decision-making
        - **Demonstrated practical application** of big data analytics for environmental sustainability
        """)
        
        st.markdown("#### 🎓 Technical Learnings")
        st.write("""
        - Data preprocessing and feature engineering techniques
        - Multi-class classification model comparison
        - Model evaluation using multiple metrics
        - Web application development with Streamlit
        - Deployment strategies for ML applications
        """)
    
    with col2:
        st.markdown("#### 🌍 SDG 14 Contribution")
        st.write("""
        This project directly supports **SDG 14: Life Below Water** by:
        
        - **Identifying at-risk species** through predictive analytics
        - **Supporting evidence-based conservation** decisions
        - **Raising awareness** about marine biodiversity threats
        - **Enabling proactive management** of marine resources
        - **Facilitating data-driven policy** development
        """)
        
        st.markdown("#### 🚀 Future Enhancements")
        st.write("""
        - Integration with real-world fishery databases
        - Time-series analysis for trend prediction
        - Geographic information system (GIS) integration
        - Multi-language support for global accessibility
        - Mobile application development
        - Real-time data updates from monitoring systems
        """)
    
    st.markdown("---")
    
    st.markdown("### 💭 Final Thoughts")
    
    st.info("""
    **Machine learning and big data analytics are powerful tools for addressing environmental challenges.**
    
    This project demonstrates how technology can support sustainable development goals by providing 
    actionable insights for marine conservation. By predicting fish species conservation status, 
    we enable stakeholders to make informed decisions about resource management and protection efforts.
    
    The success of this initiative highlights the importance of interdisciplinary approaches that 
    combine data science, environmental science, and policy-making to create meaningful impact.
    
    **Together, we can protect life below water for current and future generations.** 🌊🐟
    """)
    
    st.markdown("---")
    
    st.markdown("### 📚 Project Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔗 Useful Links")
        st.markdown("""
        - [UN SDG 14 Official Page](https://sdgs.un.org/goals/goal14)
        - [Marine Conservation Research](https://www.iucn.org/theme/marine-and-polar)
        - [Sustainable Fisheries](https://www.worldwildlife.org/industries/sustainable-seafood)
        """)
    
    with col2:
        st.markdown("#### 📖 Technologies Used")
        st.markdown("""
        - Python
        - scikit-learn
        - Streamlit
        - Pandas & NumPy
        - Matplotlib & Seaborn
        """)
    
    with col3:
        st.markdown("#### 👥 Project Team")
        st.markdown("""
        - Dataset Collection
        - Feature Engineering
        - Model Development
        - Dashboard Design
        - Documentation
        """)
    
    st.markdown("---")
    
    st.success("✅ Thank you for exploring our SDG 14 Fish Conservation Status Prediction System!")
    
    st.balloons()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "ITD105 - Big Data Analytics<br>"
    "Fish Conservation Predictor<br>"
    "© 2025"
    "</p>",
    unsafe_allow_html=True
)

