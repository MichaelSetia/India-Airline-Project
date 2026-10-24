# preprocessing.py
import joblib
import pandas as pd


def load_joblib():
    model = joblib.load('rf_default_model_small.pkl')
    scaler = joblib.load('scaler.pkl')
    le_class = joblib.load('le_dict.pkl')['class']
    ohe_dict = joblib.load('ohe_dict.pkl')
    feature_columns = joblib.load('feature.pkl')
    
    return model,scaler,le_class,ohe_dict,feature_columns


def preprocess(raw_df,scaler,ohe_dict,le_class,feature_columns):
    d = raw_df.copy()
    d['class'] = le_class.transform(d['class'])
    for c, enc in ohe_dict.items():
        enc_df = pd.DataFrame(
            enc.transform(d[[c]]),
            columns=enc.get_feature_names_out([c]),
            index=d.index
        )
        d = pd.concat([d.drop(columns=[c]), enc_df], axis=1)
        
    d= d[feature_columns]
    
    return pd.DataFrame( scaler.transform(d),columns=feature_columns, index=d.index)
    



