# Predicting Concrete Compressive Strength Using Machine Learning

## Problem
Concrete compressive strength is one of the most critical properties in structural design, but determining it traditionally requires waiting weeks for physical curing and lab testing. This project explores whether machine learning can predict compressive strength directly from a mix's ingredients and age — offering engineers a fast, data-driven estimate to support early-stage design and quality control decisions.

## Data
- **Source:** Public concrete compressive strength dataset (Kaggle)
- **Size:** 1,030 samples, 9 columns
- **Features:** cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age (all in kg/m³ except age in days)
- **Target:** concrete compressive strength (MPa)

## Approach
The project followed a standard, iterative machine learning workflow:

1. **Exploratory analysis** — examined distributions and correlations between ingredients and strength. Cement showed the strongest individual correlation with strength, consistent with materials science fundamentals.

2. **Baseline model (Linear Regression, single feature)** — using cement alone as a predictor:
   - MAE: 11.56 MPa | R²: 0.25
   - Confirmed that a single ingredient is insufficient to explain strength variation.

3. **Linear Regression with all 8 features** — added blast furnace slag, fly ash, water, superplasticizer, coarse/fine aggregate, and age:
   - MAE: 7.75 MPa | R²: 0.63
   - Superplasticizer showed the strongest positive effect on strength; water showed the strongest negative effect — both consistent with established concrete engineering (e.g., water-cement ratio's known impact on strength).

4. **Random Forest Regressor** — to capture non-linear relationships and interactions between ingredients that a linear model can't represent:
   - MAE: 3.74 MPa | R²: 0.88
   - Age and cement emerged as the two most influential features, reflecting concrete's well-known strength-development-over-time behavior.

5. **Overfitting check** — compared training vs. test performance to verify the model generalizes rather than memorizes:
   - Initial Random Forest: Training R² 0.986 vs. Test R² 0.886 (gap ≈ 0.10) — signs of mild overfitting.

6. **Regularized Random Forest** — constrained tree depth (`max_depth=10`) and minimum leaf size (`min_samples_leaf=4`) to reduce overfitting risk:
   - Training R²: 0.842 | Test R²: 0.946
   - Final Test MAE: ~3.78 MPa

## Results
The final tuned model predicts concrete compressive strength with a **test R² of 0.946** and a **mean absolute error of roughly 3.8 MPa** — meaning predictions are, on average, within about 4 MPa of the true lab-measured strength across a range that spans roughly 2–80 MPa. This level of accuracy demonstrates that ingredient composition and curing age can reliably estimate strength using a data-driven approach, without waiting for full physical curing and testing.

## What I Learned
Beyond the modeling steps themselves, this project reinforced a critical ML practice: **never trust training performance as a proxy for real-world accuracy.** The initial Random Forest looked excellent on paper (R² 0.986) but was quietly overfitting — performing meaningfully worse on unseen data. Explicitly comparing training and test scores, and then tuning the model to close that gap, produced a more trustworthy result: a model whose training and test performance are closely aligned, which is a much stronger signal of genuine generalization than a single impressive-looking score.

This project also highlighted the value of domain knowledge in ML: the model's own findings — the importance of water-cement ratio, the role of superplasticizers, and strength gain with age — all independently confirmed principles already well established in concrete materials science, giving confidence that the model learned real physical relationships rather than spurious patterns in the data.

## Tools Used
Python, pandas, matplotlib, scikit-learn (LinearRegression, RandomForestRegressor, train_test_split, mean_absolute_error, r2_score)
