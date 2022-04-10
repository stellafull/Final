# import pandas as pd

# import numpy as np
#
#
# # def findPlayer(player_name):
# #     for row in player_reader:
# #         if player_name == row[0]:
# #             return row
# #
# #
# # def findChampion(champion_name):
# #     for row in champion_reader:
# #         if champion_name == row[0]:
# #             return row
#
#
# match_data = 'C:/Users/CZY_C/Desktop/Final/Datasets/Match/match_total_spring.csv'
# player_data = 'C:/Users/CZY_C/Desktop/Final/Datasets/Player/player_2022_spring.csv'
# champion_data = 'C:/Users/CZY_C/Desktop/Final/Datasets/Champion/champion_2022_spring.csv'
#
# match_reader = pd.read_csv(match_data, sep=',', chunksize=12)
# player_reader = pd.read_csv(player_data, sep=',')
# champion_reader = pd.read_csv(champion_data, sep=',')
#
# for row in match_reader:
#     # print(row['playername'])
#     # if player_reader['Player'] == row['palyername']:
#     #     print(player_reader['Win rate'])
#     # result = df[df['Age']==23]
#     # for i in range(1, 10):
#     #     player_name = row['playername'][i]
#     #     player = findPlayer(player_name)
#     #     print(player)
#     # print(row['playername'], player_reader[player_reader['Player'].equals(row['playername'])])
#     # print(player_reader['Player'], row['playername'])
#     for i in range(0, 10):
#         print(row['playername'])
#
import load_data
import csv

match_data = load_data.getMatch()
match_file = csv.writer(open("Model/match.csv", "w", newline=""))
match_player_file = csv.writer(open("Model/match_player.csv", "w", newline=""))
match_file.writerow(
    ("blue_player1_KDA", "blue_player1_CSM", "blue_player1_GPM", "blue_player1_KP%", "blue_player1_DMG%",
     "blue_player1_DPM", "blue_champion1_KDA", "blue_champion1_Avg_BT", "blue_champion1_CSM",
     "blue_champion1_DPM", "blue_champion1_GPM", "blue_player2_KDA", "blue_player2_CSM", "blue_player2_GPM",
     "blue_player2_KP%", "blue_player2_DMG%", "blue_player2_DPM", "blue_champion2_KDA",
     "blue_champion2_Avg_BT", "blue_champion2_CSM", "blue_champion2_DPM", "blue_champion2_GPM",
     "blue_player3_KDA", "blue_player3_CSM", "blue_player3_GPM", "blue_player3_KP%", "blue_player3_DMG%",
     "blue_player3_DPM", "blue_champion3_KDA", "blue_champion3_Avg_BT", "blue_champion3_CSM",
     "blue_champion3_DPM", "blue_champion3_GPM", "blue_player4_KDA", "blue_player4_CSM",
     "blue_player4_GPM", "blue_player4_KP%", "blue_player4_DMG%", "blue_player4_DPM",
     "blue_champion4_KDA", "blue_champion4_Avg_BT", "blue_champion4_CSM", "blue_champion4_DPM",
     "blue_champion4_GPM", "blue_player5_KDA", "blue_player5_CSM", "blue_player5_GPM",
     "blue_player5_KP%", "blue_player5_DMG%", "blue_player5_DPM", "blue_champion5_KDA",
     "blue_champion5_Avg_BT", "blue_champion5_CSM", "blue_champion5_DPM", "blue_champion5_GPM",
     "red_player1_KDA", "red_player1_CSM", "red_player1_GPM", "red_player1_KP%", "red_player1_DMG%",
     "red_player1_DPM", "red_champion1_KDA", "red_champion1_Avg_BT", "red_champion1_CSM", "red_champion1_DPM",
     "red_champion1_GPM",
     "red_player2_KDA", "red_player2_CSM", "red_player2_GPM", "red_player2_KP%", "red_player2_DMG%",
     "red_player2_DPM", "red_champion2_KDA", "red_champion2_Avg_BT", "red_champion2_CSM",
     "red_champion2_DPM", "red_champion2_GPM", "red_player3_KDA", "red_player3_CSM", "red_player3_GPM",
     "red_player3_KP%", "red_player3_DMG%", "red_player3_DPM", "red_champion3_KDA",
     "red_champion3_Avg_BT", "red_champion3_CSM", "red_champion3_DPM", "red_champion3_GPM",
     "red_player4_KDA", "red_player4_CSM", "red_player4_GPM", "red_player4_KP%", "red_player4_DMG%",
     "red_player4_DPM", "red_champion4_KDA", "red_champion4_Avg_BT", "red_champion4_CSM",
     "red_champion4_DPM", "red_champion4_GPM", "red_player5_KDA", "red_player5_CSM", "red_player5_GPM",
     "red_player5_KP%", "red_player5_DMG%", "red_player5_DPM", "red_champion5_KDA",
     "red_champion5_Avg_BT", "red_champion5_CSM", "red_champion5_DPM", "red_champion5_GPM", "result"))
match_player_file.writerow(
    ("blue_player1_KDA", "blue_player1_CSM", "blue_player1_GPM", "blue_player1_KP%", "blue_player1_DMG%",
     "blue_player1_DPM", "blue_player2_KDA", "blue_player2_CSM", "blue_player2_GPM",
     "blue_player2_KP%", "blue_player2_DMG%", "blue_player2_DPM",
     "blue_player3_KDA", "blue_player3_CSM", "blue_player3_GPM", "blue_player3_KP%", "blue_player3_DMG%",
     "blue_player3_DPM", "blue_player4_KDA", "blue_player4_CSM",
     "blue_player4_GPM", "blue_player4_KP%", "blue_player4_DMG%", "blue_player4_DPM",
     "blue_player5_KDA", "blue_player5_CSM", "blue_player5_GPM",
     "blue_player5_KP%", "blue_player5_DMG%", "blue_player5_DPM",
     "red_player1_KDA", "red_player1_CSM", "red_player1_GPM", "red_player1_KP%", "red_player1_DMG%",
     "red_player1_DPM",
     "red_player2_KDA", "red_player2_CSM", "red_player2_GPM", "red_player2_KP%", "red_player2_DMG%",
     "red_player2_DPM", "red_player3_KDA", "red_player3_CSM", "red_player3_GPM",
     "red_player3_KP%", "red_player3_DMG%", "red_player3_DPM",
     "red_player4_KDA", "red_player4_CSM", "red_player4_GPM", "red_player4_KP%", "red_player4_DMG%",
     "red_player4_DPM", "red_player5_KDA", "red_player5_CSM", "red_player5_GPM",
     "red_player5_KP%", "red_player5_DMG%", "red_player5_DPM", "result"))
i = 0  # every 12 row is a match record
match_player_champion_array = []
match_player_array = []
player_exits = True

# convert 12 row match into 1 row and with parameter we need, totally 4339 matches but only 2450 matches we can use
# because some invalid game or "unkown player" or someone's id is like "GL??" we can't find out who he is
try:
    for row in match_data:
        if i < 10:
            player = load_data.getPlayer(row[0])
            champion = load_data.getChampion(row[1])
            # if champion is None:  # check if there's some champions have different names
            #     print(row[1])
            if player is None:
                player_exits = False
            else:
                for row1 in player:
                    match_player_champion_array.append(row1)
                    match_player_array.append(row1)
                for row2 in champion:
                    match_player_champion_array.append(row2)
        elif i == 11:
            i = i - 12
            if player_exits:
                match_player_champion_array.append(row[2])
                match_player_array.append(row[2])
                match_file.writerow(match_player_champion_array)
                match_player_file.writerow(match_player_array)
            player_exits = True
            match_player_champion_array = []
            match_player_array = []
        i = i + 1
        # print(j)
        # j = j + 1
except Exception as e:
    print("unexpected error", type(e), e)

# test part
# j = 0  # get load record
# for row in match_data:
#     if i < 10:
#         player = load_data.getPlayer(row[0])
#         champion = load_data.getChampion(row[1])
#         if champion is None:
#             print(row[1])
#         if player is None:
#             player_exits = False
#         else:
#             for row1 in player:
#                 match_player_champion_array.append(row1)
#             for row2 in champion:
#                 match_player_champion_array.append(row2)
#     elif i == 11:
#         i = i - 12
#         if player_exits:
#             match_player_champion_array.append(row[2])
#             match_file.writerow(match_player_champion_array)
#         player_exits = True
#         match_player_champion_array = []
#     i = i + 1
#     print(j)
#     j = j + 1
