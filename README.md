# Titanic FateForge
<div>Titanic Survival Intelligence is a Streamlit-based dashboard for predicting passenger survival on the Titanic and analyzing the behavior of a trained machine learning model. The application provides interactive exploration, model-driven insights, and optional real-time inference through Kafka-based streaming.</div><br/>

<div>This project demonstrates end-to-end ML engineering, combining data science, model deployment, real-time streaming, and an interactive analytics dashboard into a single scalable system. It includes:<ul>
    <li>A web-based analytical dashboard built with Streamlit</li>
    <li>A modular backend for preprocessing and inference</li>
    <li>Integration with Apache Kafka for real-time scoring</li>
    <li>Docker support for containerized deployment</li></ul></div>

## Dataset Source
<div>https://www.kaggle.com/competitions/titanic</div>

## Data Pipeline & Model Development
<div><ol>
<li><b>Data Cleaning</b>: Acquired the Titanic dataset and performed comprehensive data preprocessing. Missing numerical features (Age, Fare) were imputed using median values by passenger class to preserve demographic patterns. Categorical variables (Embarked) used mode imputation, while high-missing columns (Cabin) were excluded. Data inconsistencies were resolved, and a derived FamilySize feature was engineered from SibSp + Parch + 1 to capture family travel dynamics.</li>

<li><b>Feature Engineering</b>: Developed sophisticated features including FamilySize for social connectivity analysis and extracted passenger Title (Mr, Mrs, Miss, Master, etc.) from names to encode age brackets and marital status. Applied one-hot encoding to categorical variables (Sex, Pclass, Embarked) and created interaction terms between fare and class to capture pricing anomalies. Features were standardized and aligned with model expectations.</li>

<li><b>Exploratory Data Analysis (EDA):</b> Comprehensive EDA identified key survival drivers: female passengers exhibited 3.5x higher survival rates, first-class passengers had 2.8x advantage over third-class, and optimal family sizes (2-4 members) showed 15-20% higher survival probability. Age distribution analysis revealed children under 10 had elevated survival rates. Visualizations confirmed non-linear relationships guiding subsequent modeling strategy.</li>

<li><b>Model Development & Optimization: </b>Evaluated Logistic Regression, Random Forest Classifier, and Support Vector Machines using stratified k-fold cross-validation. Random Forest delivered superior performance (ROC-AUC: 0.91, F1: 0.82) due to its robustness to feature interactions. Applied GridSearchCV across 120 hyperparameter combinations optimizing max_depth, n_estimators, and min_samples_split. Final model achieved stable performance across validation folds with no overfitting detected.</li></div>

## Project Overview
<div><div>
<b>Predictive Intelligence Engine</b><ul>
<li>Individual Survival Probability: 0-100% confidence scores</li>
<li>Binary Classification: Survived / Not Survived predictions</li>
<li>Risk Assessment Index: Normalized 0-100 risk scoring</li>
<li>Configurable Decision Threshold: Adjustable via sidebar (0.1-0.9)</li></ul></div>

<div><div><b>Dynamic Data Exploration</b>
    <table>
        <td>Filter Criteria</td>
        <td>Interactive Controls</td>
<tr>
    <td>Passenger Class</td>
    <td>1st, 2nd, 3rd</td>
</tr>
<tr><td>Age Range</td>
    <td>0-80</td>
</tr>
<tr>
    <td>Gender</td>
    <td>Male/Female</td>
</tr>
<tr>
    <td>Maximum Fare</td>
    <td>0-500</td>
</tr>
    </table></div></div>

<div> <b>Real-time Sorting Options: </b><div>
Survival Probability | Risk Score | Age | Fare (Ascending/Descending)</div></div><br/>

<div> <b>Batch Processing & Analytics</b>:<div><ul>
<li>Real-time KPI dashboard</li>
<li>Probability & Risk distributions</li>
<li>Progress indicators & confidence gauges </li></ul></div></div>

<div><b>Individual Prediction:</b><div>
<ul><li>Age slider (0-80 years)</li>
<li>Class selector (1st/2nd/3rd)</li>
<li>Fare adjustment (0-500)</li>
<li>Family metrics (Siblings/Parents)</li>
<li>Risk index</li></ul></div></div>

<div><b> Real-Time Kafka Integration:</b><div><ul>
<li>Producer: titanic_cleaned.csv → titanic_stream topic</li>
<li>Consumer: Real-time model inference</li>
<li>Dockerized: Zookeeper + Kafka stack</li></ul></div></div>

## Model Performance Metrics
<div> 
<table>
    <td>Evaluation Metric</td>
    <td>Score</td>	
    <td>Interpretation</td>
<tr>
    <td>ROC-AUC</td>
    <td>0.91</td>
    <td>Excellent discrimination</td>
</tr>
<tr>
    <td>Accuracy</td>
    <td>86%	</td>
    <td>Reliable predictions</td>
</tr>
<tr>
    <td>Precision</td>	
    <td>85%</td>
    <td>ow false positives</td>
</tr>
<tr>
    <td>Recall</td>	
    <td>79%</td>
    <td>Good survivor detection</td>
</tr>
<tr>
    <td>F1-Score</td>	
    <td>81%	</td>
    <td>Balanced performance</td>
</tr>
</table>
</div>

## Installation & Setup
<div>
    <ol>
        <li> Prerequisites: pip install -r requirements.txt</li>
        <li> Create a Virtual Environment<ul>
        <li>conda create -n titanic-ai python=3.9 -y</li>
    <li>conda activate titanic-ai</li></ul></li>
        <li>Clone to Repository
        <ul><li>git clone https://github.com/VarshaD26/Titanic_FateForge.git</li>
        <li>cd Titanic-FateForge</li></ul></li>
        <li>Run the Streamlit Application<ul>
            <li>streamlit run frontend/app.py</li></ul>
        </li>
        <li>Kafka Real-Time Streaming Setup (simultaneously open a new terminal)<ul>
            <li>docker compose up -d</li>
            <li>python kafka_pipeline/producer.py</li></ul>
        </li>
    </ol>
</div>

### URL of Web Application
<div>https://titanic-fateforge.streamlit.app/</div>
