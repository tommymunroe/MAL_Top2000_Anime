# MAL_Top2000_Anime
Using Kaggle dataset to examine what characteristics make a top 25 anime on MyAnimeList.com

Using the mal_top2000_anime.csv, I dropped columns that weren't necessary for my final regression (i.e. Score Rank will have too high of a correlation with a top 25 dummy variable).

Took qualitative columns (i.e. genre, theme, demographics, platform, and studio) and created dummy variables (examples below):
Genres: divided into Action, Drama, Comedy, and an "Other" category
Themes: divided into Romance, Gag Humor, Military, Historical, Psychological, Gore, and an "Other" category
Demographics: divided into Shonen, Seinen, Shojo, and Josei
Studio: divided into multiple different studios and an "Other" category
Platform: divided into "TV" or "Not TV"

Concatted the dummy variables into a new dataset and then ran a logistical regression using scikitlearn and the coefficients can be found in the ModelOutput.xlsx file.
