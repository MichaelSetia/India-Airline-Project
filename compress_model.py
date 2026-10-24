import joblib

model = joblib.load('rf_default_model.pkl')
joblib.dump(model, 'rf_default_model_small.pkl', compress=3)