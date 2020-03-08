import pandas as pd
import math


#Converts a datetime object to a number on the scale from one to 365.
def datetime_to_day_of_year(date):
    return pd.Period(date, freq='H').dayofyear

#Uses a cosine function to for a better representation of a cyclical value.
def day_of_year_to_period(day, north=True):
    converted = math.cos(day * 2 * math.pi / 365)
    if north:
        return -converted
    else:
        return converted

#Maps season categorical values from a day on a 365 scale.
def determine_season(day, north=True):
    seasons = ["spring", "summer", "fall", "winter"] if north else ["fall", "winter", "spring", "summer"]
    if day < 172 and day >= 80:
        season = seasons[0]
    elif day < 264 and day >= 172:
        season = seasons[1]
    elif day < 355 and day >= 264:
        season = seasons[2]
    else:
        season = seasons[3]
    return season

#Determines how old the well is. The reason why it is incremeented in intervals of .25 is to extract as much information
#left from the date to try to gather more precision while compensating slightly from the information lost with repeating
# values when converting to a period.
def determine_years_old(year_built, day, year):
    day_rounded = round(day/365*4)/4
    return year + day_rounded - year_built



# def date_features(df):
#     df_j["check-day"] = df_j["date_recorded"].map(datetime_to_day_of_year)
#     df_j["check-season"] = df_j["day_of_year"].map(lambda x: determine_season(x, False))
#     df_j["check-period"] = df_j["day_of_year"].map(lambda x: day_of_year_to_period(x, False))

def date_features(line):
    date, year_built = line["date_recorded"], line["construction_year"]
    day, year = datetime_to_day_of_year(date), date.year
    season, period = determine_season(day, False), day_of_year_to_period(day, False)
    years_old = determine_years_old(year_built, day, year)
    line["check-season"], line["check-period"], line["years_old"] = season, period, years_old
    return line

def add_features(df):
    return df.apply(date_features, axis=1)
