import pandas as pd

df = pd.read_csv('../raw_data/airlines_flights_data.csv')

duration = (df.groupby(['source_city','destination_city','stops'])['duration'].median().round(2).reset_index())

arrival = (df.groupby(['source_city', 'destination_city', 'stops', 'departure_time'])['arrival_time']
             .agg(lambda s: s.mode().iloc[0]).reset_index())

duration.to_csv('duration_lookup.csv', index=False)
arrival.to_csv('arrival_lookup.csv', index=False)
print(len(duration), len(arrival))