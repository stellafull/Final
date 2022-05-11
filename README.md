# Final
My college final work
This Repository is my Graduation Thesis of my College, all data and code can be found in the pacakge

U need to create your conda env with requirements

```conda
conda create --name <env_name> --file requirements.txt
```

If U want to use you need first download [Oracle's Elixir](https://oracleselixir.com/tools/downloads) to get match meta data, and get player and champion data from [gol.gg](https://gol.gg), and put them into load/data, maybe U need refactor file name or modify the Load/data/load_csv_to_db.py
U need to run data/load_csv_to_db.py first load them into database
Second run load/load.py to create a match for training and test
Finally run load/model/player_champion_model.py to train the model, and use model.py to sump model then use evaluate.py to evaluate the Acc and ROC.
