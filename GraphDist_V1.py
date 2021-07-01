import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Récupération des données et déclaration des axes
# sep spécifie le caractère séparateur de colonnes
# header = 0 : la ligne numéro 0 = aux noms des champs
data = pd.read_csv("DataWebtrack_V2.csv", header = 0)

#énumération des colonnes
print(data.columns)
# Types des données des colonnes de notre dataset
print(data.dtypes)
# Vérifier si valeurs manquantes
print(data.isna().sum())
# retirons les lignes avec des valeurs manquantes
data.dropna(inplace=True)
# vérifions le résultat
data.isna().sum()

y1 = data.ListDist
y2 = data.ListPitch
y3 = data.ListYaw
y4 = data.ListRoll
x = data.listeTime

axes = plt.gca()

# Tracer une courbe représentant Y en fonction de X
plt.grid(True)
axes.set_ylim(-100, 100)
#plt.plot(x,y,"b", linewidth=0.8, marker="*")
plt.scatter(x, y1, label='Distance')
plt.scatter(x, y2, label='Pitch')
plt.scatter(x, y3, label='Yaw')
plt.scatter(x, y4, label='Roll')
# beautify the x-labels
plt.gcf().autofmt_xdate()

# lignes horizontal de criticité
plt.axhline(y=70, color='gray', linestyle='--')
plt.axhline(y=50, color='gray', linestyle='--')


# Pour rajouter un TITRE et un LABEL sur les axes 'x et y'
plt.title('Graphique Temps actif par rapport à la distance oeil-écran', fontsize=8)
plt.xlabel('Temps')
plt.ylabel('Distance')
plt.legend()
# Afficher un graphique dans une fenêtre
plt.savefig("Stats/GraphResult3.png",dpi=300)
plt.show()
# Possibilité de fermer la fenêtre en tappant 'q'
plt.close(ord("q"))
