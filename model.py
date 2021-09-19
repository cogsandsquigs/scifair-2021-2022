import lmfit
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class SimpleCovidModel:
    # params = lmfit.Parameters()

    def __init__(self, N, days, y0, params, fit_data):
        self.params = params  # model parameters
        self.N = N  # number of people
        self.days = days  # number of days
        self.y0 = y0  # initial starting vector
        self.t = np.linspace(0, days, days)  # array of days
        self.fit_data = fit_data

    def beta(self, beta_a, beta_b, beta_k, t):
        return beta_k / (1 + np.exp(beta_a + beta_b * t))

    # The SIR model differential equations.
    def deriv(self, y, t, a, b, k, gamma, rho):

        N = self.N

        S, I, R, D = y
        dSdt = -self.beta(a, b, k, t) * S * I / N
        dIdt = self.beta(a, b, k, t) * S * I / N - gamma * I
        dRdt = (1 - rho) * gamma * I
        dDdt = (rho) * gamma * I
        return dSdt, dIdt, dRdt, dDdt

    def predict(self):
        ret = odeint(
            self.deriv,
            self.y0,
            self.t,
            args=(
                self.params["beta_a"],
                self.params["beta_b"],
                self.params["beta_k"],
                self.params["gamma"],
                self.params["rho"],
            ),
        )
        return ret.T

    def fitter(self, x, beta_a, beta_b, beta_k, gamma, rho):
        ret = odeint(
            self.deriv,
            self.y0,
            self.t,
            args=(
                beta_a,
                beta_b,
                beta_k,
                gamma,
                rho,
            ),
        )
        return ret.T[3]

    def fit(self):

        mod = lmfit.Model(self.fitter)
        params = mod.make_params()

        x_data = self.t
        y_data = self.fit_data

        result = mod.fit(
            y_data,
            params,
            method="least_squares",
            x=x_data,
            beta_a=self.params["beta_a"],
            beta_b=self.params["beta_b"],
            beta_k=self.params["beta_k"],
            gamma=self.params["gamma"],
            rho=self.params["rho"],
        )

        print(result.fit_report())
        self.params = result.params

        return

    def optimizer(self, x, beta_a, beta_b):
        ret = odeint(
            self.deriv,
            self.y0,
            self.t,
            args=(
                beta_a,
                beta_b,
                self.params["beta_k"],
                self.params["gamma"],
                self.params["rho"],
            ),
        )

        totaldeaths = ret.T[3][-1]

        return totaldeaths

    def optimize(self):
        mod = lmfit.Model(self.optimizer)
        # params = mod.make_params()

        params = lmfit.Parameters()

        params.add("beta_a", min=0, max=1, value=0.2)
        params.add("beta_b", min=0, max=1, value=0.2)

        x_data = self.t
        y_data = self.fit_data

        result = mod.fit(
            y_data,
            params,
            method="least_squares",
            x=x_data,
            beta_a=self.params["beta_a"],
            beta_b=self.params["beta_b"],
        )

        print(result.fit_report())

        self.params["beta_a"] = params["beta_a"]
        self.params["beta_b"] = params["beta_b"]

    def plot(self):

        t = self.t

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 5))

        def set_ax(ax, S, I, R, D):
            # ax.plot(t, S, "g", alpha=0.5, lw=2, label="Susceptible")
            ax.plot(t, I, "r", alpha=0.5, lw=2, label="Infected")
            # ax.plot(t, R, "b", alpha=0.5, lw=2, label="Recovered")
            ax.plot(t, D, "k", alpha=0.5, lw=2, label="Dead")
            ax.plot(t, self.fit_data, "y", alpha=0.5, lw=2, label="True Dead")

            ax.set_xlabel("Time in days")
            ax.set_ylabel("Number of people per category")

            ax.yaxis.set_tick_params(length=0)
            ax.xaxis.set_tick_params(length=0)

            legend = ax.legend()
            legend.get_frame().set_alpha(0.5)

            for spine in ("top", "right", "bottom", "left"):
                ax.spines[spine].set_visible(False)

        def set_ax_no_truedead(ax, S, I, R, D):
            # ax.plot(t, S, "g", alpha=0.5, lw=2, label="Susceptible")
            ax.plot(t, I, "r", alpha=0.5, lw=2, label="Infected")
            # ax.plot(t, R, "b", alpha=0.5, lw=2, label="Recovered")
            ax.plot(t, D, "k", alpha=0.5, lw=2, label="Dead")
            # ax.plot(t, self.fit_data, "y", alpha=0.5, lw=2, label="True Dead")

            ax.set_xlabel("Time in days")
            ax.set_ylabel("Number of people per category")

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

        self.optimize()

        S, I, R, D = self.predict()
        set_ax_no_truedead(ax3, S, I, R, D)
        ax3.set_title("after optimizing")

        plt.show()
