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

### 3. Target Variable (Label)
**Growth Rate (%)** = `((Population_future - Population_current) / Population_current) × 100`

**Why predict growth rate instead of absolute population?**
- Growth rates are more comparable across countries of different sizes
- Percentage changes normalize the scale (India's growth vs Monaco's growth)
- Easier for models to learn relative changes than absolute numbers
- Avoids bias toward large-population countries

**Example Calculation**:
- Current Population (2015): 1,000,000
- Future Population (2025): 1,100,000
- Growth Rate = ((1,100,000 - 1,000,000) / 1,000,000) × 100 = **10.0%**

**Converting predictions back to population**:
```
Predicted Population_2025 = Population_2015 × (1 + Growth_Rate_predicted / 100)
```

### 4. Training Data Creation Strategy
**Multi-horizon Training**: For each country-year pair, we create multiple training samples with different prediction horizons:

| Prediction Horizon | Years Ahead | Example |
|-------------------|-------------|---------|
| Short-term | 5 years | 2010 → 2015 |
| Medium-term | 10 years | 2010 → 2020 |
| Long-term | 15 years | 2010 → 2025 |

**Benefits**:
- Enables model to learn growth patterns across different timescales
- Dramatically expands dataset (each country contributes multiple samples)
- Model learns that growth rates depend on time horizon (encoded as "Years ahead" feature)
- Focused on practical prediction horizons (5, 10, 15 years)

**Final Dataset**:
- **~4,000+ training samples** from 150+ countries across 50+ years
- Each sample: (Country features + Year + Horizon → Growth rate)
- After quality filtering and outlier removal: ~3,200 training samples, ~800 test samples

## Machine Learning Approach

### Problem Formulation
**Task**: Regression problem predicting population growth rate percentage  
**Label/Target Variable**: Growth Rate (%) = `((Population_future - Population_current) / Population_current) × 100`  
**Input Features**: 9 numeric features (Population, Birth rate, Death rate, Life expectancy, GDP per capita, GDP, Density, Year, Years ahead)

### Data Quality & Preprocessing
**Data Quality Filtering**:
- Rows with >30% missing values removed from training set
- Ensures high-quality training data
- Typical retention rate: ~95-98% of samples

**Outlier Detection & Removal** (Training Set Only):
- **Isolation Forest** algorithm detects anomalies across all numeric features
- Contamination threshold: 5% (expects ~5% outliers)
- Applied **only to training set** to prevent data leakage
- Test set remains completely unchanged
- Removes extreme growth rate outliers (e.g., war-torn countries, major migrations)

### Training Data Statistics
- **Total Training Samples**: ~4,000+ country-year-horizon pairs (after quality filtering)
- **Train Set**: 80% (~3,200 samples after outlier removal)
- **Test Set**: 20% (~800 samples, unchanged)
- **Random State**: 42 (for reproducibility)
- **Year Coverage**: Historical data from 1960-2016
- **Geographic Coverage**: 150+ countries across 7 world regions
- **Prediction Horizons**: 5, 10, and 15 years ahead

### Preprocessing Pipeline
All models use identical preprocessing via scikit-learn `Pipeline`:
1. **KNN Imputation** (k=5): Fills missing values using 5 nearest neighbors based on Euclidean distance
2. **Standard Scaling**: Normalizes all features to μ=0, σ=1

### Models Trained & Compared
Three regression models trained with identical preprocessing:

1. **Linear Regression**
   - Baseline model assuming linear relationships
   - No hyperparameters tuned
   - Fastest training time

2. **Random Forest Regressor**
   - `n_estimators=500`: 500 decision trees (increased for better performance)
   - `max_depth=10`: Maximum tree depth to prevent overfitting
   - `random_state=42`: Reproducible results
   - `n_jobs=-1`: Parallel processing on all CPU cores
   - Captures non-linear interactions

3. **Gradient Boosting Regressor**
   - `n_estimators=500`: 500 sequential boosting stages (increased for better performance)
   - `max_depth=5`: Shallow trees to prevent overfitting
   - `learning_rate=0.1`: Step size for gradient descent
   - `random_state=42`: Reproducible results
   - Sequential error correction approach
   - Best performance on test set

### Model Selection Strategy

**Two-Stage Selection Process**:

**Stage 1: Cross-Validation** (Model Selection)
- **5-Fold Cross-Validation** on training set
- All models evaluated using identical CV folds
- **Scoring Metric**: RMSE (Root Mean Squared Error)
- **Best Model Selection**: Model with lowest mean CV RMSE
- Provides robust performance estimate before test set evaluation
- Prevents overfitting to validation splits

**Stage 2: Final Evaluation** (Performance Confirmation)
- Best model (from CV) trained on full training set
- Final evaluation on held-out test set
- Confirms generalization performance

**Evaluation Metrics**:

**Primary Metric**: RMSE (Root Mean Squared Error)
- Formula: `√(Σ(y_pred - y_true)² / n)`
- Measures average magnitude of prediction errors in percentage points
- Penalizes large errors more heavily than MAE
- Used for both CV selection and final evaluation

**Secondary Metric**: MAE (Mean Absolute Error)
- Formula: `Σ|y_pred - y_true| / n`
- Measures average absolute error in percentage points
- More robust to outliers
- Provides interpretable average error

### Error Analysis & Diagnostics
For each model, comprehensive diagnostics include:

**Performance Metrics**:
- **Train RMSE/MAE**: Performance on training set (overfitting check)
- **CV RMSE**: 5-fold cross-validation performance (mean ± std dev)
- **Test RMSE/MAE**: Performance on held-out test set (final generalization)

**Visual Diagnostics** (3x3 Grid):
- **Row 1**: Metric comparisons (Test RMSE, Test MAE, Train vs Test RMSE)
- **Row 2**: Prediction accuracy scatter plots for all 3 models
- **Row 3**: Residual analysis plots with mean/std statistics

**Residual Analysis**:
- Mean residual close to 0 indicates unbiased predictions
- Standard deviation of residuals indicates prediction consistency
- Residual plots reveal systematic errors or patterns

**Typical Performance** (Gradient Boosting - Best Model):
- CV RMSE: ~2-3% (5-fold average)
- Test RMSE: ~2-3% (average error of 2-3 percentage points)
- Test MAE: ~1-2% (average absolute error)
- No significant overfitting (train/test/CV metrics similar)

### Validation Strategy
- **Hold-out Validation**: 80/20 train-test split with `random_state=42`
- **5-Fold Cross-Validation**: For model selection on training set
- **No data leakage**: 
  - Preprocessing fitted only on training set, then applied to test set
  - Outlier detection fitted only on training set
  - Test set never seen during model selection
- **Temporal considerations**: Predictions span multiple time horizons (5, 10, 15 years)
- **Model comparison**: All models evaluated on identical splits and folds

## Model Interpretability

### Feature Importance (SHAP Analysis)
We use **SHAP (SHapley Additive exPlanations)** values to understand which features drive predictions:

**SHAP Value Interpretation**:
- **Positive SHAP**: Feature increases predicted growth rate
- **Negative SHAP**: Feature decreases predicted growth rate
- **Magnitude**: Strength of feature's impact

**Typical Feature Rankings** (from Gradient Boosting model):
1. **Birth rate**: Strong positive impact (↑ births → ↑ growth)
2. **Death rate**: Strong negative impact (↑ deaths → ↓ growth)
3. **Years ahead**: Longer horizons typically show different growth patterns
4. **Life expectancy**: Complex relationship with demographic transitions
5. **GDP/Population density**: Moderate impact on growth patterns

**Analysis Process**:
- Sample 100 random training instances for efficiency
- Compute SHAP values using `TreeExplainer` (optimized for tree models)
- Visualize directional impacts and feature interactions

## Predictions & Applications

### Generated Predictions
Using 2016 baseline data (most recent available), the model predicts:

| Target Year | Horizon | Use Case |
|------------|---------|----------|
| **2021** | 5 years | Short-term planning, validation against actual 2021 data |
| **2026** | 10 years | Medium-term resource allocation |
| **2031** | 15 years | Long-term strategic planning |

### Prediction Process
For each country:
1. Extract 2016 socioeconomic features (latest available data)
2. Set prediction horizon (5, 10, or 15 years)
3. Model predicts growth rate percentage
4. Convert to absolute population: `Pop_2021 = Pop_2016 × (1 + rate/100)`
5. Save predictions with metadata (country, region, income group)

### Output Files
**Prediction CSVs**:
- `population_predictions_2021.csv`: Country-level 2021 predictions (+5 years)
- `population_predictions_2026.csv`: Country-level 2026 predictions (+10 years)
- `population_predictions_2031.csv`: Country-level 2031 predictions (+15 years)

**Visualizations**:
- `cross_validation_results.png`: 5-fold CV performance comparison
- `model_performance.png`: 3x3 grid of diagnostic plots (metrics, scatter plots, residuals)
- `data_visualization.png`: Training data distributions and statistics
- `outlier_detection.png`: Before/after outlier removal comparison
- `shap_importance.png`: Feature importance and directional impacts
- `population_prediction_multi_horizon.png`: Multi-horizon predictions overview

### Interactive Country Lookup
The notebook includes an interactive tool to query predictions:
- Enter any country name (partial matches supported)
- Automatically displays all three prediction horizons (2021, 2026, 2031)
- Shows baseline population, predicted populations, growth rates, and changes
- Formatted as easy-to-read table with 15-year summary

## Key Features & Methodology Highlights

### Data Quality & Robustness
✓ **Data quality filtering**: Removes rows with >30% missing values  
✓ **Outlier detection**: Isolation Forest removes anomalies from training set only  
✓ **No data leakage**: Test set completely untouched during preprocessing and model selection  
✓ **Proper preprocessing**: KNN imputation + scaling in pipeline (fit on train, transform on test)

### Model Selection & Validation
✓ **5-Fold Cross-Validation**: Scientific model selection before test set evaluation  
✓ **Rigorous train-test split**: 80/20 split with held-out test set  
✓ **Multi-horizon training**: 3 time horizons (5, 10, 15 years) for robust predictions  
✓ **Objective selection**: Best model chosen by lowest CV RMSE  
✓ **Two-stage validation**: CV selection + final test set confirmation

### Model Performance & Diagnostics
✓ **Comprehensive error analysis**: RMSE, MAE, residual plots, overfitting checks  
✓ **Visual diagnostics**: 3x3 grid comparing all models (metrics, scatter plots, residuals)  
✓ **CV performance tracking**: Mean RMSE with standard deviation and 95% confidence intervals  
✓ **Production-ready models**: Tuned hyperparameters (500 estimators, optimized depth)

### Interpretability & Insights
✓ **SHAP analysis**: Explains which features drive predictions and their directional impacts  
✓ **Feature importance**: Identifies key drivers of population growth  
✓ **Regional analysis**: Growth patterns by world region and income group  
✓ **Interactive lookup**: Query any country to see all prediction horizons

### Reproducibility & Documentation
✓ **Reproducible**: All random seeds set (42) for consistent results  
✓ **Professional visualizations**: Publication-quality plots with detailed annotations  
✓ **Comprehensive outputs**: 6 visualizations + 3 prediction CSVs saved  
✓ **Well-documented**: Clear code comments and print statements throughout