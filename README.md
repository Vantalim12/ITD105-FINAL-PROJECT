# SDG 14: Fish Species Conservation Status Prediction System

## ITD105 - Big Data Analytics | Case Study Project

A machine learning-powered web application that predicts fish species conservation status to support **UN Sustainable Development Goal 14: Life Below Water**.

![SDG 14](https://www.un.org/sustainabledevelopment/wp-content/uploads/2019/08/E-Inverted-Icons_WEB-14.png)

---

## Project Overview

This project uses **classification-based predictive modeling** to assess the conservation status of fish species, helping identify marine biodiversity risks related to overfishing and ecosystem degradation.

### Key Features
- Compares 5 beginner-friendly classification algorithms
- Interactive Streamlit dashboard with 5 comprehensive pages
- Real-time conservation status predictions
- Beautiful data visualizations
- Model performance comparison dashboard
- Ready for deployment on Streamlit Cloud

---

## SDG 14 Alignment

**SDG 14: Life Below Water** focuses on conserving and sustainably using oceans, seas, and marine resources.

This project supports SDG 14 by:
- Predicting fish species conservation status (Good, Moderate, Poor)
- Identifying species at risk
- Supporting data-driven marine conservation decisions
- Raising awareness about threatened marine life

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Generate the dataset**
```bash
python generate_dataset.py
```
This creates `fish_conservation_data.csv` with 1000 synthetic fish species samples.

4. **Train and compare models**
```bash
python train_models.py
```
This trains 5 classification models and saves the best performer.

5. **Launch the web application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Machine Learning Pipeline

### Classification Algorithms Compared
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree Classifier**
4. **Random Forest Classifier**
5. **Naive Bayes**

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Dataset Features
- **Habitat Type**: Coral Reef, Deep Sea, Coastal, Open Ocean, Estuary
- **Population Trend**: Increasing, Stable, Declining, Critical
- **Fishing Pressure**: Low, Moderate, High, Very High
- **Average Size (cm)**: Fish body length
- **Geographic Region**: Pacific, Atlantic, Indian Ocean, Arctic, Mediterranean
- **Reproduction Rate**: Offspring per year
- **Depth Range (m)**: Habitat depth
- **Water Temperature (°C)**: Preferred temperature
- **Population Size**: Number of individuals (in thousands)

### Target Variable (Conservation Status)
- **Good** - Healthy and sustainable population
- **Moderate** - Population showing warning signs
- **Poor** - High risk or endangered

---

## Web Application Pages

### Page 1: SDG 14 Overview
- Description of SDG 14 and project motivation
- Key challenges in marine conservation
- System workflow explanation

### Page 2: Dataset Explorer
- Interactive data table with filters
- Summary statistics
- Data visualizations (distribution charts, pie charts, bar graphs)
- Conservation status breakdown

### Page 3: Model Comparison Dashboard
- Accuracy comparison bar chart
- Detailed evaluation metrics table
- Confusion matrix for each model
- Classification reports

### Page 4: Prediction Tool
- User-friendly input form for fish characteristics
- Real-time conservation status prediction
- Color-coded prediction results
- Prediction confidence scores
- Conservation recommendations

### Page 5: Conclusion & Impact
- Best model summary
- Project achievements and insights
- SDG 14 contribution analysis
- Future enhancement suggestions

---

## Project Structure

```
ITD105-Final-Project/
│
├── generate_dataset.py          # Synthetic dataset generator
├── train_models.py               # Model training and comparison script
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
│
├── fish_conservation_data.csv    # Generated dataset (after running generate_dataset.py)
├── best_model.pkl                # Trained best model (after running train_models.py)
└── all_models.pkl                # All models data (after running train_models.py)
```

---

## Application Screenshots

### Dashboard Features
- Interactive visualizations
- Real-time predictions
- Model performance metrics
- Conservation insights

---

## Deployment

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Deploy!

### Important Files for Deployment
- `app.py` - Main application
- `requirements.txt` - Dependencies
- `fish_conservation_data.csv` - Dataset
- `best_model.pkl` - Trained model
- `all_models.pkl` - Model comparison data

**Note:** Make sure to run `generate_dataset.py` and `train_models.py` before deployment to create the necessary data files.

---

## Group Task Distribution

| Member | Responsibility |
|--------|---------------|
| Member 1 | Dataset collection & cleaning |
| Member 2 | Feature engineering |
| Member 3 | Model training & comparison |
| Member 4 | Streamlit dashboard development |
| Member 5 | Documentation & SDG alignment |

---

## Technologies Used

- **Python 3.8+** - Programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning
- **Matplotlib & Seaborn** - Data visualization
- **Streamlit** - Web application framework

---

## Expected Outputs

- Trained multi-class classification model  
- Interactive Streamlit web application  
- Model comparison dashboard  
- Real-time prediction tool  
- Comprehensive documentation  
- Ready for online deployment  

---

## Key Insights

- Machine learning can effectively predict conservation status with high accuracy
- Multiple models should be compared to select the best performer
- Interactive visualizations make data insights accessible
- Web applications democratize ML predictions for non-technical users
- Data-driven approaches support better conservation decisions

---

## Future Enhancements

- Integration with real-world fishery databases
- Time-series analysis for population trend forecasting
- Geographic Information System (GIS) mapping
- Multi-language support
- Mobile application development
- Real-time monitoring system integration
- API development for external system integration

---

## References

- [UN SDG 14: Life Below Water](https://sdgs.un.org/goals/goal14)
- [IUCN Red List of Threatened Species](https://www.iucnredlist.org/)
- [FAO Fisheries and Aquaculture](http://www.fao.org/fishery/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## License

This project is developed for educational purposes as part of ITD105 - Big Data Analytics course.

---

## Contributing

This is a course project, but suggestions and feedback are welcome!

---

## Contact

For questions or collaboration opportunities, please contact the project team.

---

## Course Information

**Course:** ITD105 - Big Data Analytics  
**Project:** Case Study - Development of an Integrated Web Application with Machine Learning Predictive Modeling  
**Focus:** UN Sustainable Development Goal 14 - Life Below Water  

---

## Acknowledgments

Special thanks to:
- United Nations for the Sustainable Development Goals framework
- The open-source community for excellent ML and web development tools
- Our course instructor for guidance and support

---

<p align="center">
  <strong>Together, we can protect life below water!</strong>
</p>

<p align="center">
  Made with dedication for marine conservation
</p>

