import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as web
import yfinance as yf
import statsmodels.formula.api as smf
import HW.HW10.WOE as woe
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
import sklearn.metrics
import seaborn as sb

anime = pd.read_csv('mal_top2000_anime.csv',index_col='Index')
# print(anime)
anime.drop(['Air Date','Popularity Rank','Score'],axis=1,inplace=True)

pd.set_option('display.max_columns',5)



def printOutTheCoefficients(params,coeffecients,intercept):
    tParams = params[np.newaxis].T
    tCoeffs = coeffecients.T
    total = np.concatenate([tParams,tCoeffs],axis=1)
    totalDF = pd.DataFrame(data=total)
    totalDF.to_excel("modelOutput.xlsx")
    print(totalDF)



def Shonen(x):
    if('Shounen' in x): return 'Younger Male Demo'
    if('Shoujo' in x): return 'Younger Female Demo'
    if('Seinen' in x): return 'Older Male Demo'
    if('Josei' in x): return 'Older Female Demo'
    else: return 'Other Demographic'
anime['Demographics'] = anime.apply(lambda row: Shonen(row['Demographic']),axis=1)
anime.drop(['Demographic'],axis=1,inplace=True)
# print(anime.head(25))


def Top25(x):
    if x <= 25:
        return 'Top 25'
    else:
        return 'Not Top 25'
anime['Top 25'] = anime.apply(lambda row: Top25(row['Score Rank']),axis=1)
anime.drop(['Score Rank'],axis=1,inplace=True)


def TVorMovie(x):
    if('TV' in x): return 'TV'
    if not('TV' in x): return 'Not TV'
    return x
anime['TVorNot'] = anime.apply(lambda row: TVorMovie(row['Type']),axis=1)
anime.drop(['Type'],axis=1,inplace=True)
# print(anime.head(25))

# anime = anime.drop(anime[anime['Genres']==' "" '])

def genre(x):
    first_element = x[2:5]
    return first_element

def superGenre(x):
    if('Act' in x): return "Action"
    if('Dra' in x): return 'Drama'
    if('Com' in x): return 'Comedy'
    else: return 'Other Genre'
anime['Genress'] = anime.apply(lambda row: genre(row['Genres']),axis=1)
anime['Genre'] = anime.apply(lambda row: superGenre(row['Genress']),axis=1)
anime.drop(['Genres','Genress'],axis=1,inplace=True)
# print(anime.head(25))


def theme(x):
    element = x[2:11]
    return element

def superTheme(x):
    if('Gag Humor' in x): return 'Gag Humor'
    if('Military' in x): return 'Military'
    if('Childcare' in x): return 'Childcare'
    if('Romantic' in x): return 'Romance'
    if('Gore' in x): return 'Gore'
    if('Vampire' in x): return 'Vampire'
    if('Mecha' in x): return 'Mecha'
    if('Historica' in x): return 'Historical'
    if('Adult Cas' in x): return 'Adult Cast'
    if('Psycholog' in x): return 'Psychological'
    else: return 'Other Theme'
anime['Themess'] = anime.apply(lambda row: theme(row['Theme(s)']),axis=1)
anime['Theme'] = anime.apply(lambda row: superTheme(row['Themess']),axis=1)
anime.drop(['Theme(s)','Themess'],axis=1,inplace=True)
# print(anime.head(25))

def studio(x):
    element = x[2:7]
    return element

def superStudio(x):
    if 'Bones' in x:
        return 'Bones'
    if 'Wit S' in x:
        return 'Wit Studios'
    if 'White' in x:
        return 'White Fox'
    if 'Banda' in x:
        return 'Bandai Namco Pictures'
    if 'Sunri' in x:
        return 'Sunrise'
    if 'Madho' in x:
        return 'Madhouse'
    if 'TMS E' in x:
        return 'TMS Entertainment'
    if 'K-Fac' in x:
        return 'K-Factory'
    if 'A-1 P' in x:
        return 'A-1 Pictures'
    if 'Shaft' in x:
        return 'Shaft'
    if 'Kyoto' in x:
        return 'Kyoto Animation'
    if 'MAPPA' in x:
        return 'Mappa'
    if 'ufota' in x:
        return 'ufotable'
    if 'CoMix' in x:
        return 'CoMix Wave Films'
    else:
        return 'Other Studio'
anime['Studioss'] = anime.apply(lambda row: studio(row['Studio']),axis=1)
anime['Studios'] = anime.apply(lambda row: superStudio(row['Studioss']),axis=1)
anime.drop(['Studio','Studioss'],axis=1,inplace=True)
# print(anime.head(25))

anime.drop(['Name'],axis=1,inplace=True)
print(anime.head())

# anime.to_csv('coolAnimeData.csv')



demographics = pd.get_dummies(anime['Demographics'])
top25 = pd.get_dummies(anime['Top 25'])
tv = pd.get_dummies(anime['TVorNot'])
genres = pd.get_dummies(anime['Genre'])
themes = pd.get_dummies(anime['Theme'])
studios = pd.get_dummies(anime['Studios'])

anime.drop('Demographics',axis=1,inplace=True)
anime.drop('Top 25',axis=1,inplace=True)
anime.drop('TVorNot',axis=1,inplace=True)
anime.drop('Genre',axis=1,inplace=True)
anime.drop('Theme',axis=1,inplace=True)
anime.drop('Studios',axis=1,inplace=True)

animeJoin = pd.concat([demographics,top25,tv,genres,themes,studios],axis=1)
# print(animeJoin)
# animeJoin.to_excel('animeJoinedDF.xlsx')

dfResults = animeJoin['Top 25']
dfInputs = animeJoin.drop(['Top 25','Not Top 25'],axis=1)

inputsTrain,inputsTest,resultTrain,resultTest = train_test_split(dfInputs,
                                    dfResults,test_size=0.3,random_state=1)

LogReg = LogisticRegression()
LogReg.fit(inputsTrain,resultTrain)

resultPred = LogReg.predict(inputsTest)
# print(classification_report(resultTest,resultPred))
# print("Intercept(b):", LogReg.intercept_)

# printOutTheCoefficients(dfInputs.columns.values,LogReg.coef_,LogReg.intercept_)

