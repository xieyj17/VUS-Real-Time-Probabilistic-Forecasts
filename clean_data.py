import pandas as pd 
import numpy as np  
from utils import clean_nba_data

raw_data = pd.read_csv("nba_2018.csv")
raw_data.head()

df = clean_nba_data(raw_data)

