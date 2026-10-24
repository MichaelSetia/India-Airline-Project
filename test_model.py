import pandas as pd
from preprocessing import load_joblib, preprocess

model,scaler,le,ohe,feature = load_joblib()

sample = pd.DataFrame([{
    'airline': 'SpiceJet', 'source_city': 'Delhi', 'departure_time': 'Evening',
    'stops': 'zero', 'arrival_time': 'Night', 'destination_city': 'Mumbai',
    'class': 'Economy', 'duration': 2.17, 'days_left': 1
}])

X=preprocess(sample,scaler,ohe,le,feature)
print(X)
print(model.predict(X))