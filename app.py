import streamlit as st
import pandas as pd
from preprocessing import load_joblib , preprocess
from datetime import date,timedelta

st.set_page_config(page_title="Prediksi Tiket Pesawat India")

@st.cache_resource
def loadall():
    return load_joblib()

@st.cache_resource
def load_lookup():
    return pd.read_csv('./arrival_lookup.csv'), pd.read_csv('./duration_lookup.csv')

df_arrival,df_duration = load_lookup()
model,scaler,le,ohe,feature = loadall()
st.title("✈️ Prediksi Harga Tiket Pesawat")
st.write("")

airlines = {'Air Asia':'AirAsia', 'Air India':'Air_India', 'Go First':'GO_FIRST', 'Indigo': 'Indigo', 'SpiceJet':'SpiceJet', 'Vistara':'Vistara'}
cities = ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
times = {'Early Morning':'Early_Morning', 'Morning':'Morning', 'Afternoon':'Afternoon', 'Evening':'Evening', 'Night':'Night', 'Late Night':'Late_Night'}
stops_map = {'Langsung (0 transit)': 'zero', '1 transit': 'one', '2+ transit': 'two_or_more'}




airline = st.selectbox("Maskapai",list(airlines.keys()))
source_city = st.selectbox("Kota asal", cities, index=2)
destination_city = st.selectbox("Kota tujuan", cities, index=5)
departure_time = st.selectbox("Waktu berangkat", times, index=3)
stops_label = st.selectbox("Jumlah transit", list(stops_map.keys()))
flight_class = st.selectbox("Kelas", ['Economy', 'Business'])


today=date.today()
tgl = st.date_input(
    "Tanggal keberangkatan",
    value=today + timedelta(days=7),
    min_value=today + timedelta(days=1),
    max_value=today + timedelta(days=49),
)



if st.button("Prediksi Harga"):
    stops = stops_map[stops_label]
    dep = times[departure_time]
    
    if source_city == destination_city:
        st.error("Kota asal dan tujuan tidak boleh sama.")
        st.stop()
        
    d = df_duration[(df_duration.source_city == source_city) &
               (df_duration.destination_city == destination_city) &
               (df_duration.stops == stops)]
    a = df_arrival[(df_arrival.source_city == source_city) &
               (df_arrival.destination_city == destination_city) &
               (df_arrival.stops == stops) &
               (df_arrival.departure_time == dep)]
    
    if d.empty or a.empty:
        st.warning("Kombinasi rute, transit, dan waktu berangkat ini tidak ada di data. "
                   "Coba ubah waktu berangkat atau jumlah transit.")
        st.stop()

    
    duration = float(d['duration'].iloc[0])
    arrival_time = a['arrival_time'].iloc[0]
    days_left = (tgl - today).days

    raw = pd.DataFrame([{
        'airline': airlines[airline],
        'source_city': source_city,
        'departure_time': dep,
        'stops': stops,
        'arrival_time': arrival_time,
        'destination_city': destination_city,
        'class': flight_class,
        'duration': duration,
        'days_left': days_left,
    }])
    
    X = preprocess(raw, scaler, ohe ,le, feature)
    pred = model.predict(X)[0]
    st.success(f"Perkiraan harga tiket: ₹ {pred:,.0f}")
    st.caption(f"Estimasi durasi {duration:.1f} jam · tiba waktu "
               f"{arrival_time.replace('_', ' ')} · {days_left} hari lagi")