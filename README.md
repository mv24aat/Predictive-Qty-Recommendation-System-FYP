# Predictive-Qty-Recommendation-System-FYP
Predictive Quantity Recommendation System for medication dosage to reduce adverse effect with patients and built trust.
A machine learning project predicting a patient's stable warfarin maintenance dose from clinical, demographic and pharmacogenomic data (CYP2C9, VKORC1, CYP4F2), built with explainability at its core.

Warfarin has a narrow therapeutic window, and the correct dose varies widely between patients. This project uses tuned Decision Tree and Extra Trees models, trained on 50,000 de-identified patient records, to predict a patient's stable dose upfront giving clinicians a data-driven starting point instead of trial and error. Model decisions are explained using LIME and SHAP, and the best model is deployed as an interactive Gradio app.

Best result: R² of 0.88 and MAE of 0.41 after hyperparameter optimisation.

Stack: Python · scikit-learn · MLflow · LIME · SHAP · Gradio

Final year MSc Data Science project, University of Hertfordshire.
