 🎯 Goal
In this project, we implemented a **basic regression analysis** using **pandas + matplotlib + scikit-learn**.  
The aim is to predict **Sales** from advertising budgets (TV, Radio, Social Media) and product Price.  

### 🔨 Steps
1. **Data Generation:**  
   - Synthetic data for TV, Radio, Social budgets and Price.  
   - Sales (`y`) derived from these features.  

2. **Data Preparation:**  
   - Defined features (`X`) and target (`y`) in a pandas DataFrame.  
   - Train/Test split (75% / 25%).  

3. **Modeling:**  
   - Fitted a `LinearRegression` model.  
   - Learned coefficients representing each feature’s effect on Sales.  

4. **Evaluation:**  
   - Metrics: **R², MAE, RMSE**.  
   - Results: Test R² ≈ 0.80 (good explanatory power).  

5. **Visualization:**  
   - Scatter plot: TV vs Sales with regression line.  
   - Scatter plot: Actual vs Predicted Sales (Test set).  

### 📦 Libraries Used
- **numpy** → numerical operations, synthetic data generation.  
- **pandas** → data management with DataFrame.  
- **matplotlib** → visualizations.  
- **scikit-learn** → modeling and metrics.  

### 📊 Outputs
- Coefficients:  
  - Price negative → higher price decreases sales.  
  - TV, Radio, Social positive → higher ads increase sales.  
- Saved plots:  
  - `scatter_tv_sales.png`  
  - `actual_vs_pred.png`

<img width="1542" height="893" alt="Screenshot 2025-09-06 at 14 26 47" src="https://github.com/user-attachments/assets/92b446fb-32af-465f-990d-af5a28a9f160" />

  
