import csv_to_sqlite
import pandas as pd

options = csv_to_sqlite.CsvOptions(typing_style="full", encoding="windows-1250")
input_files = ["2022_LoL_esports_match_data_from_OraclesElixir_20220330.csv", "champion_2022_spring.csv",
               "player_2022_spring.csv"]
csv_to_sqlite.write_csv(input_files, "database.db", options)

df = pd.read_csv("2022_LoL_esports_match_data_from_OraclesElixir_20220330.csv")
datatype = df.dtypes
datatype.to_csv("datatype.csv")
