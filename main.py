from pathlib import Path
import matplotlib.pyplot as plt
from datetime import time
from os import path, times
import lmfit
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse

from simple_model import SimpleCovidModel
from log_model import LogCovidModel
from complex_model import ComplexCovidModel

def us_county_data():
    us_county_covid_deaths = pd.read_csv("data/time_series_covid19_deaths_US.csv")
    us_county_population = pd.read_csv("data/co-est2020-alldata.csv")

    datalist = []
    countylist = [
        (84017031, "Illinois", "Cook County"),
        (84025003, "Massachusetts", "Berkshire County"),
        (84001007, "Alabama", "Bibb County"),
    ]

    for i in range(len(countylist)):

        uid, state, county = countylist[i]

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

        states = us_county_population.loc[us_county_population["STNAME"] == state]
        cty_pop = states.loc[states["CTYNAME"] == county]

        datalist.append((state, county, deathslist, int(cty_pop["POPESTIMATE2020"])))

    return datalist


def get_params():
    params = lmfit.Parameters()
    # all of these rates can be represented as
    # 1/the days it takes to go from one state to the next
    # an end state is simply either being recovered or being dead

    # beta is calculated through these next three lines is the contact rate
    params.add("beta_a", min=0, max=1, value=0.2)
    params.add("beta_b", min=0, max=1, value=0.2)
    params.add("beta_k", min=0, max=10, value=0.2)

    params.add("lockdown_a", value=0.2)
    params.add("lockdown_b", value=0.2)
    params.add("lockdown_c", value=0.2)

    params.add(
        "gamma", min=0, max=1, value=0.1
    )  # gamma is the infected -> end state rate
    params.add("rho", min=0, max=1, value=0.25)  # rho is the death rate
    return params


usdata = us_county_data()
params = get_params()

# SimpleCovidModel,
models = [LogCovidModel, ComplexCovidModel, SimpleCovidModel]


for m in models:

    outdata = pd.DataFrame(
        {
            "state": [],
            "county": [],
            "accuracy (MSE)": [],
            "total deaths": [],
            "optimized total deaths": [],
            "death difference (MSE)": [],
            "overall lockdown intensity": [],
            "optimized overall lockdown intensity": [],
            "lockdown intensity difference (MSE)": [],
        }
    )

    for i in range(len(usdata)):

        state, county, deaths, pop = usdata[i]

        # Total population, N.
        # us state pop data from https://www.census.gov/programs-surveys/popest/technical-documentation/research/evaluation-estimates/2020-evaluation-estimates/2010s-counties-total.html
        N = pop
        # Amount of days in the simulation
        days = len(deaths)
        # A grid of time points (in days)
        t = np.linspace(0, days, days)
        # fitting data
        fit_data = np.array(deaths)
        # Initial number of infected and recovered individuals, I0 and R0.
        L0, I0, R0, D0 = 0, 1, 0, 0
        # Everyone else, S0, is susceptible to infection initially.
        S0 = N - I0 - R0 - D0

        y0 = L0, S0, I0, R0, D0

        newModel = m(
            N,
            days,
            y0,
            params,
            fit_data,
        )

        retdata = newModel.plot(state, county, display=False)
        outdata.loc[len(outdata.index)] = [
            state,
            county,
            mse(newModel.fit_data, retdata[0][4]),
            retdata[0][4][-1],
            retdata[1][4][-1],
            mse(
                retdata[0][4],
                retdata[1][4],
            ),
            sum(retdata[0][0]),
            sum(retdata[1][0]),
            mse(
                retdata[0][0],
                retdata[1][0],
            ),
        ]

    outdata.to_csv(Path(r"./out-" + m.__name__ + r"-data.csv"))

# print(outdata)
