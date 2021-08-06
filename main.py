import lmfit
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Amount of days in the simulation
days = 160
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


def sigmoid(x, a, b, k):
    return k / (1 + np.exp(a + b * x))


fit_data = sigmoid(t, 10, -0.1, 200)


class CovidModel:
    # params = lmfit.Parameters()

    def __init__(self, N, days, y0, params):
        self.params = params  # model parameters
        self.N = N  # number of people
        self.days = days  # number of days
        self.y0 = y0  # initial starting vector
        self.t = np.linspace(0, days, days)  # array of days

    # The SIR model differential equations.
    def deriv(self, y, t, beta, gamma, rho):

        N = self.N

        S, I, R, D = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = (1 - rho) * gamma * I
        dDdt = (rho) * gamma * I
        return dSdt, dIdt, dRdt, dDdt

    def predict(self):
        ret = odeint(
            self.deriv,
            self.y0,
            self.t,
            args=(
                self.params["beta"],
                self.params["gamma"],
                self.params["rho"],
            ),
        )
        return ret.T

    def fitter(self, x, beta, gamma, rho):
        ret = odeint(
            self.deriv,
            self.y0,
            self.t,
            args=(
                beta,
                gamma,
                rho,
            ),
        )
        return ret.T[3]

    def fit(self):

        mod = lmfit.Model(self.fitter)
        params = mod.make_params()

        x_data = self.t
        y_data = fit_data

        result = mod.fit(
            y_data,
            params,
            method="least_squares",
            x=x_data,
            beta=self.params["beta"],
            gamma=self.params["gamma"],
            rho=self.params["rho"],
        )

        print(result.fit_report())
        self.params = result.params

        return

    def plot(self):

        t = self.t

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

        def set_ax(ax, S, I, R, D):
            ax.plot(t, S / 1000, "g", alpha=0.5, lw=2, label="Susceptible")
            ax.plot(t, I / 1000, "r", alpha=0.5, lw=2, label="Infected")
            ax.plot(t, R / 1000, "b", alpha=0.5, lw=2, label="Recovered")
            ax.plot(t, D / 1000, "k", alpha=0.5, lw=2, label="Dead")
            ax.plot(t, fit_data / 1000, "y", alpha=0.5, lw=2, label="True Dead")

            ax.set_xlabel("Time /days")
            ax.set_ylabel("Number (1000s)")

            ax.set_ylim(0, 1.2)

            ax.yaxis.set_tick_params(length=0)
            ax.xaxis.set_tick_params(length=0)

            legend = ax.legend()
            legend.get_frame().set_alpha(0.5)

            for spine in ("top", "right", "bottom", "left"):
                ax.spines[spine].set_visible(False)

        S, I, R, D = self.predict()
        set_ax(ax1, S, I, R, D)
        ax1.set_title("before fitting")

        self.fit()

        S, I, R, D = self.predict()
        set_ax(ax2, S, I, R, D)
        ax2.set_title("after fitting")

        plt.show()


y0 = S0, I0, R0, D0
newModel = CovidModel(N, days, y0, params)

newModel.plot()
