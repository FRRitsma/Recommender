import json

with open('movie_titles.csv', 'r',  encoding="ISO-8859-1") as f:
    TitleList = f.read().splitlines()
TitleList = [f.split(',')[-1] for f in TitleList]

with open('movie_titles.csv', 'r',  encoding="ISO-8859-1") as f:
    YearList = f.read().splitlines()
YearList = [f.split(',')[-2] for f in YearList]

with open("MovieFrequencies.json", 'r') as f:
    MovieFrequencies = json.load(f)

with open("Lookup.json", 'r') as f:
    Lookup = json.load(f)

