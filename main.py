from datetime import time
from os import times
import lmfit
import numpy as np
import pandas as pd

from model import CovidModel


us_county_covid_deaths = pd.read_csv("data/time_series_covid19_deaths_US.csv")
uid = 84017031  # bibb county, alabama

county_data = us_county_covid_deaths.loc[us_county_covid_deaths["UID"] == uid]

dateslist = list(county_data)[12:]

deathslist = []

for date in dateslist:
    deathslist.append(int(county_data[date]))

dateshift = 0

for n in deathslist:
    if n == 0:
        dateshift += 1

deathslist = deathslist[dateshift:]

us_county_population = pd.read_csv("data/co-est2020-alldata.csv")
state = "Illinois"
county = "Cook County"

states = us_county_population.loc[us_county_population["STNAME"] == state]
cty_pop = states.loc[states["CTYNAME"] == county]

# print(deathslist)

# Amount of days in the simulation
days = len(deathslist)
# Total population, N.
# us state pop data from https://www.census.gov/programs-surveys/popest/technical-documentation/research/evaluation-estimates/2020-evaluation-estimates/2010s-counties-total.html
N = int(cty_pop["POPESTIMATE2020"])
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0, D0 = 1, 0, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0 - D0


params = lmfit.Parameters()
# all of these rates can be represented as
# 1/the days it takes to go from one state to the next
# an end state is simply either being recovered or being dead
params.add("beta", min=0, max=1, value=0.2)  # beta is the contact rate
params.add("gamma", min=0, max=1, value=0.1)  # gamma is the infected -> end state rate
params.add("rho", min=0, max=1, value=0.25)  # rho is the death rate


# A grid of time points (in days)
t = np.linspace(0, days, days)


def sigmoid(x, a, b, k):
    return k / (1 + np.exp(a + b * x))


fit_data = np.array(deathslist)  # sigmoid(t, 10, -0.1, 200)

y0 = S0, I0, R0, D0
newModel = CovidModel(N, days, y0, params, fit_data)

newModel.plot()
