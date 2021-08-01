import lmfit
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Amount of days in the simulation
days = 155
# Total population, N.
N = 1000
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


def sigmoid(x, h, k, a):
    return 1 / (1 + np.exp(-x))


class CovidModel:
    # params = lmfit.Parameters()

    def __init__(self, params):
        self.params = params

    # The SIR model differential equations.
    def deriv(self, y, t, N):

        # defining variables
        beta = self.params["beta"]
        gamma = self.params["gamma"]
        rho = self.params["rho"]

        S, I, R, D = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = (1 - rho) * gamma * I
        dDdt = (rho) * gamma * I
        return dSdt, dIdt, dRdt, dDdt


# Initial conditions vector
y0 = S0, I0, R0, D0
# Integrate the SIR equations over the time grid, t.
newModel = CovidModel(params)
ret = odeint(newModel.deriv, y0, t, args=(N,))
S, I, R, D = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
def plot():

    fig = plt.figure(facecolor="w")

    ax = fig.add_subplot()

    ax.plot(t, S / 1000, "b", alpha=0.5, lw=2, label="Susceptible")
    ax.plot(t, I / 1000, "r", alpha=0.5, lw=2, label="Infected")
    ax.plot(t, R / 1000, "g", alpha=0.5, lw=2, label="Recovered")
    ax.plot(t, D / 1000, "k", alpha=0.5, lw=2, label="Dead")
    ax.plot(t, sigmoid(t, 100, 0, 1000) / 1000, "y", alpha=0.5, lw=2, label="True Dead")

    ax.set_xlabel("Time /days")
    ax.set_ylabel("Number (1000s)")

    ax.set_ylim(0, 1.2)

    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)

    # ax.grid(b=True, which="major", c="k", lw=2, ls="-")

    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)

    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)

    plt.show()


plot()
