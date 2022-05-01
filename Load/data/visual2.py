import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix


data = pd.read_csv("match.csv")
df = data[['blue_player1_KDA','blue_player2_KDA','blue_player3_KDA','blue_player4_KDA','blue_player5_KDA','red_player1_KDA','red_player2_KDA','red_player3_KDA','red_player4_KDA','red_player5_KDA']]

# scatter_matrix(df, alpha=0.2, diagonal='kde')
# plt.show()

sm = scatter_matrix(df, alpha=0.2, diagonal='kde')
for subaxis in sm:
    for ax in subaxis:
        # ax.xaxis.set_ticks([])
        # ax.yaxis.set_ticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")
plt.show()
# pic = sm[0][0].get_figure()
# pic.savefig("MyScatter.png")