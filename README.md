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
| Short-term | 1, 2, 3 years | 2010 → 2011, 2012, 2013 |
| Medium-term | 5, 10 years | 2010 → 2015, 2020 |
| Long-term | 15, 20 years | 2010 → 2025, 2030 |

**Benefits**:
- Enables model to learn growth patterns across different timescales
- Dramatically expands dataset (each country contributes multiple samples)
- Model learns that growth rates depend on time horizon (encoded as "Years ahead" feature)

**Final Dataset**:
- **~4,000+ training samples** from 150+ countries across 50+ years
- Each sample: (Country features + Year + Horizon → Growth rate)

## Machine Learning Approach

### Problem Formulation
**Task**: Regression problem predicting population growth rate percentage  
**Label/Target Variable**: Growth Rate (%) = `((Population_future - Population_current) / Population_current) × 100`  
**Input Features**: 9 numeric features (Population, Birth rate, Death rate, Life expectancy, GDP per capita, GDP, Density, Year, Years ahead)

### Training Data Statistics
- **Total Training Samples**: ~4,000+ country-year-horizon pairs
- **Train Set**: 80% (~3,200 samples)
- **Test Set**: 20% (~800 samples)
- **Random State**: 42 (for reproducibility)
- **Year Coverage**: Historical data from 1960-2015
- **Geographic Coverage**: 150+ countries across 7 world regions

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
   - `n_estimators=100`: 100 decision trees
   - `random_state=42`: Reproducible results
   - `n_jobs=-1`: Parallel processing on all CPU cores
   - Captures non-linear interactions

3. **Gradient Boosting Regressor**
   - `n_estimators=100`: 100 sequential boosting stages
   - `random_state=42`: Reproducible results
   - Sequential error correction approach
   - Best performance on test set

### Model Selection Criteria
**Primary Metric**: Test RMSE (Root Mean Squared Error)
- Formula: `√(Σ(y_pred - y_true)² / n)`
- Measures average magnitude of prediction errors in percentage points
- Penalizes large errors more heavily than MAE

**Secondary Metric**: Test MAE (Mean Absolute Error)
- Formula: `Σ|y_pred - y_true| / n`
- Measures average absolute error in percentage points
- More robust to outliers

**Selection Process**:
```python
best_model = min(models, key=lambda m: test_rmse)
```
The model with lowest test RMSE is automatically selected as the best model.

### Error Analysis & Metrics
For each model, we track:
- **Train RMSE/MAE**: Performance on training set (overfitting check)
- **Test RMSE/MAE**: Performance on held-out test set (generalization)
- **Residual Statistics**: Mean and standard deviation of prediction errors
- **Residual Plots**: Visual inspection of error patterns

**Typical Performance** (Gradient Boosting - Best Model):
- Test RMSE: ~2-3% (average error of 2-3 percentage points)
- Test MAE: ~1-2% (average absolute error)
- No significant overfitting (train/test metrics similar)

### Validation Strategy
- **Hold-out Validation**: 80/20 train-test split with `random_state=42`
- **No data leakage**: Preprocessing fitted only on training set, then applied to test set
- **Temporal considerations**: Predictions span multiple time horizons (1-20 years)
- **Model comparison**: All models evaluated on identical test set

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
Using 2015 baseline data, the model predicts:

| Target Year | Horizon | Use Case |
|------------|---------|----------|
| **2025** | 10 years | Near-term planning, validation against actual 2025 data |
| **2030** | 15 years | Medium-term resource allocation |
| **2035** | 20 years | Long-term strategic planning |

### Prediction Process
For each country:
1. Extract 2015 socioeconomic features
2. Set prediction horizon (e.g., 10 years for 2025)
3. Model predicts growth rate percentage
4. Convert to absolute population: `Pop_2025 = Pop_2015 × (1 + rate/100)`
5. Save predictions with metadata (country, region, income group)

### Output Files
- `population_predictions_2025.csv`: Country-level 2025 predictions
- `population_predictions_2030.csv`: Country-level 2030 predictions
- Visualizations: Regional growth rates, top growing/declining countries

## Key Features & Methodology Highlights
✓ **Rigorous train-test split**: 80/20 split prevents data leakage  
✓ **Proper preprocessing**: KNN imputation + scaling in pipeline (fit on train, transform on test)  
✓ **Multi-horizon training**: 7 different time horizons for robust predictions  
✓ **Objective model selection**: Automatic selection based on test RMSE  
✓ **Comprehensive error analysis**: RMSE, MAE, residual plots, overfitting checks  
✓ **Interpretability**: SHAP values explain which features drive predictions  
✓ **Professional visualizations**: Growth patterns, regional trends, model comparisons  
✓ **Reproducible**: Random seeds set (42) for consistent results