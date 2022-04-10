import sqlite3
import pandas as pd


# get palyer info
def getPlayer(player_name):
    conn = sqlite3.connect('data/database.db')
    c = conn.cursor()
    cursor = c.execute("select KDA, CSM, GPM, [KP%], [DMG%], DPM from player where Player=? COLLATE NOCASE",
                       (player_name,))
    for row in cursor:
        return row
    conn.close()


# get champion info
def getChampion(champion_name):
    conn = sqlite3.connect('data/database.db')
    c = conn.cursor()
    cursor = c.execute("select KDA, [Avg BT], CSM, DPM, GPM from champion where Champion=?", (champion_name,))
    for row in cursor:
        return row
    conn.close()


# get whole match record
def getMatch():
    conn = sqlite3.connect('data/database.db')
    c = conn.cursor()
    cursor = c.execute("select playername, champion, result from match")
    return cursor
    conn.close()

# def closeDB():
#     conn.close()


# test part
# print(getPlayer("GLL"))
# for row in getMatch():
#     print(row)
# print(getPlayer("vital"))
