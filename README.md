# Prediction-Population-Growth

Project Description: For our project, we plan to explore what factors best predict a country’s population growth. This will involve using a historical global population dataset that contains additional socioeconomic indicators like GDP per capita, life expectancy, development indices, and demographic features. This topic is relevant to a wide variety of stakeholders, including governments planning resource allocation, economists studying the progression of populations and development, and international organizations analyzing global trends. These predictions support market sizing, demand forecasting, and investment strategy. Businesses can use them to spot emerging markets, plan expansion, optimize supply chains, and identify regions with growing labor forces or consumer bases. Because population growth is influenced by several complex, nonlinear factors, machine learning would provide an ideal approach to this project. Machine learning models can capture these relationships much better than traditional linear models, as well as manage high-dimensional data, which would offer insights that can benefit all stakeholders involved.


# Population Growth Prediction Pipeline

## Overview
This notebook implements a machine learning pipeline to predict **population growth rates** for countries worldwide. The model learns to predict the percentage change in population over different time horizons (1, 2, 3, 5, 10, 15, and 20 years ahead) based on current socioeconomic indicators.

## Data Processing

### 1. Data Sources
- **World Population Data**: Historical population figures for all countries (wide format → converted to long format)
- **World Bank Indicators**: Socioeconomic features including birth/death rates, GDP, life expectancy, infrastructure metrics

### 2. Feature Engineering
The model uses the following features to predict growth rates:
- **Demographic**: Birth rate, Death rate, Life expectancy, Population density
- **Economic**: GDP, GDP per capita
- **Temporal**: Current year, Years ahead (prediction horizon)

### 3. Target Variable
**Growth Rate (%)** = `((future_population - current_population) / current_population) × 100`

The model predicts the growth rate percentage, which is then used to calculate future population:
`Predicted Population = Current Population × (1 + Growth Rate / 100)`

### 4. Training Data Creation
- Creates multiple training pairs per country across different time horizons [1, 2, 3, 5, 10, 15, 20 years]
- This expands the dataset significantly, enabling the model to learn growth patterns at various timescales
- Final training set: ~4,000+ country-year-horizon combinations

## Machine Learning Approach

### Models Trained
Three regression models are compared:
1. **Linear Regression** - Baseline model for linear relationships
2. **Random Forest** - Ensemble model capturing non-linear patterns
3. **Gradient Boosting** - Advanced ensemble with sequential error correction

### Preprocessing Pipeline
Each model uses an independent preprocessing pipeline:
- **KNN Imputation** (k=5): Fills missing values using nearest neighbors
- **Standard Scaling**: Normalizes features to zero mean and unit variance

### Model Selection Criteria
- **Primary Metric**: RMSE (Root Mean Squared Error)
- **Secondary Metric**: MAE (Mean Absolute Error)
- Best model selected based on lowest test RMSE

### Validation Strategy
- **80/20 Train-Test Split**: Ensures no data leakage
- **5-Fold Cross-Validation**: Validates model robustness on training set
- **Learning Curves**: Analyzes overfitting and model convergence

## Predictions
The trained model generates predictions for:
- **2025**: 10 years ahead from 2015 baseline
- **2035**: 20 years ahead from 2015 baseline

Using 2015 socioeconomic indicators, the model predicts growth rates, which are converted to absolute population figures for each country.

## Key Features
✓ No data leakage (independent preprocessing per model, proper train-test split)  
✓ Multiple time horizons for robust long-term predictions  
✓ Comprehensive model comparison and validation  
✓ SHAP analysis for feature importance interpretation  
✓ Professional visualizations for growth patterns and regional trends