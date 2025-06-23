import pandas as pd 
import numpy as np  
from utils import clean_nba_data
from vus import generate_binned_df, get_vus, plot_vus

raw_data = pd.read_csv("nba_2018.csv")
raw_data.head()

df = clean_nba_data(raw_data)

binned_df, bin_width_map = generate_binned_df(df)
vus = get_vus(binned_df, bin_width_map)
print(vus)
plot_vus(binned_df, bin_width_map)
