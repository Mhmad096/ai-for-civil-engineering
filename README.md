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
   - MAE: 11.56 MPa & R²: 0.25
   - Confirmed that a single ingredient is insufficient to explain strength variation.

3. **Linear Regression with all 8 features** — added blast furnace slag, fly ash, water, superplasticizer, coarse/fine aggregate, and age:
   - MAE: 7.75 MPa & R²: 0.63
   - Superplasticizer showed the strongest positive effect on strength; water showed the strongest negative effect — both consistent with established concrete engineering (e.g., water-cement ratio's known impact on strength).

4. **Random Forest Regressor** — to capture non-linear relationships and interactions between ingredients that a linear model can't represent:
   - MAE: 3.74 MPa & R²: 0.88
   - Age and cement emerged as the two most influential features, reflecting concrete's well-known strength-development-over-time behavior.

5. **Overfitting check** — compared training vs. test performance to verify the model generalizes rather than memorizes:
   - Initial Random Forest: Training R² 0.986 vs. Test R² 0.886 (gap ≈ 0.10) — signs of mild overfitting.

6. **Regularized Random Forest** — constrained tree depth (`max_depth=10`) and minimum leaf size (`min_samples_leaf=4`) to reduce overfitting risk:
   - Training R²: 0.95259 & Test R²: 0.8617


## Results
The final regularized Random Forest (max_depth=10, min_samples_leaf=4) predicts concrete compressive strength with a test R² of 0.862 (training R² of 0.953). In practical terms, the model explains roughly 86% of the variance in lab-measured strength on unseen data, across a range that spans roughly 2-80 MPa. This shows that ingredient composition and curing age can give a solid data-driven estimate of strength without waiting for full physical curing and testing.

## What I Learned
BBeyond the modeling steps themselves, this project reinforced a critical ML practice: never trust training performance as a proxy for real-world accuracy. The initial Random Forest looked excellent on paper (R² 0.986) but was overfitting, performing noticeably worse on unseen data. I constrained the trees by limiting depth and requiring a minimum leaf size, which reduced the training score to 0.953. Some overfitting remains, since a gap of about 0.09 separates training and test R², but the test score of 0.862 is a more honest estimate of real-world performance than any single impressive-looking training number. It also shows the trade-off of regularization: the training score dropped, and the result is a model I can trust more, even if it is not perfect.

This project also highlighted the value of domain knowledge in ML: the model's own findings, namely the importance of water-cement ratio, the role of superplasticizers, and strength gain with age, all independently confirmed principles already well established in concrete materials science. That gave me confidence the model learned real physical relationships rather than spurious patterns in the data.

## Tools Used
Python, pandas, matplotlib, scikit-learn (LinearRegression, RandomForestRegressor, train_test_split, mean_absolute_error, r2_score)
