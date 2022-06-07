#%%
# Load data and ease of access:
# Translate movie-numerics to titles and release years:
def GetMovieTitle(i) -> str:
    global TitleList
    i = int(i) - 1
    return TitleList[i]

def GetMovieYear(i) -> str:
    global YearList
    i = int(i) - 1
    return YearList[i]

# Load data in format Movie -> {user:rating, user:rating, etc...}
def GetRatingDict(LookupValues, List) -> dict:
    OutDict = {}
    for f in List[LookupValues[1]:LookupValues[2]]:
        OutDict[f.split(',')[0]] = f.split(',')[1]
    return OutDict

# For a list of movies, get all ratings:
def GetMovieRatings(Movie, Lookup) -> dict:
    if type(Movie) is not list:
        Movie = [Movie]
    FileNames = set(Lookup[m][0] for m in Movie)
    OutDict = {m:None for m in Movie}
    for fileName in FileNames:
        with open(fileName, "r") as f:
            files = f.readlines()
        for m in Movie:
            LookupValues = Lookup[m]
            if LookupValues[0] == fileName:
                OutDict[m] = GetRatingDict(LookupValues, files)
    return OutDict