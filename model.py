import lmfit
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class SimpleCovidModel:
    def __init__(self, N, days, y0, params, fit_data):
        self.params = params  # model parameters
        self.N = N  # number of people
        self.days = days  # number of days
        self.y0 = y0  # initial starting vector
        self.t = np.linspace(0, days, days)  # array of days
        self.fit_data = fit_data

    def beta(self, beta_a, beta_b, beta_k, t):
        return beta_k / (1 + np.exp(beta_a + beta_b * t))

    def lockdown(self, a, b, t):
        return 1 / (1 + np.exp(a + b * t))

    # The SIR model differential equations.
    def deriv(
        self,
        y,
        t,
        beta_a,
        beta_b,
        beta_k,
        slipthrough,
        lockdown_a,
        lockdown_b,
        gamma,
        rho,
    ):

        N = self.N

        L, S, I, R, D = y
        dLdt = slipthrough - self.lockdown(lockdown_a, lockdown_b, t)
        dSdt = -self.beta(beta_a, beta_b, beta_k, t) * L * S * I / N
        dIdt = self.beta(beta_a, beta_b, beta_k, t) * (1 + L) * S * I / N - gamma * I
        dRdt = (1 - rho) * gamma * I
        dDdt = (rho) * gamma * I
        return dLdt, dSdt, dIdt, dRdt, dDdt

    def predict(self):
        ret = odeint(
            self.deriv,
            self.y0,
            self.t,
            args=(
                self.params["beta_a"],
                self.params["beta_b"],
                self.params["beta_k"],
                self.params["slipthrough"],
                self.params["lockdown_a"],
                self.params["lockdown_b"],
                self.params["gamma"],
                self.params["rho"],
            ),
        )
        return ret.T

    def fitter(
        self, x, beta_a, beta_b, beta_k, slipthrough, lockdown_a, lockdown_b, gamma, rho
    ):
        ret = odeint(
            self.deriv,
            self.y0,
            self.t,
            args=(
                beta_a,
                beta_b,
                beta_k,
                slipthrough,
                lockdown_a,
                lockdown_b,
                gamma,
                rho,
            ),
        )
        return ret.T[4]

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
            slipthrough=self.params["slipthrough"],
            lockdown_a=self.params["lockdown_a"],
            lockdown_b=self.params["lockdown_b"],
            gamma=self.params["gamma"],
            rho=self.params["rho"],
        )

        print(result.fit_report())
        self.params = result.params

        return

    def optimizer(self, params):
        ret = odeint(
            self.deriv,
            self.y0,
            self.t,
            args=(
                self.params["beta_a"],
                self.params["beta_b"],
                self.params["beta_k"],
                self.params["slipthrough"],
                params["lockdown_a"],
                params["lockdown_b"],
                self.params["gamma"],
                self.params["rho"],
            ),
        )

        totaldeaths = ret.T[4][-1]
        lockdownintensity = sum(ret.T[0])

        return totaldeaths, lockdownintensity

    def optimize(self):
        """
        mod = lmfit.Model(self.optimizer)
        # params = mod.make_params()



        x_data = self.t
        y_data = self.fit_data

        result = mod.fit(
            y_data,
            params,
            method="least_squares",
            x=x_data,
            lockdown_a=self.params["lockdown_a"],
            lockdown_b=self.params["lockdown_b"],
        )
        """

        params = lmfit.Parameters()

        params.add("lockdown_a", min=0, max=1, value=0.2)
        params.add("lockdown_b", min=0, max=1, value=0.2)

        # print(result.fit_report())
        lmfit.minimize(self.optimizer, params, method="leastsq")

        self.params["lockdown_a"] = params["lockdown_a"]
        self.params["lockdown_b"] = params["lockdown_b"]

    def plot(self):

        t = self.t

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 5))

        def set_ax(ax, L, S, I, R, D):
            # ax.plot(t, S, "g", alpha=0.5, lw=2, label="Susceptible")
            l1 = ax.plot(t, I, "r", alpha=0.5, lw=2, label="Infected")
            # ax.plot(t, R, "b", alpha=0.5, lw=2, label="Recovered")
            l2 = ax.plot(t, D, "k", alpha=0.5, lw=2, label="Dead")
            l3 = ax.plot(t, self.fit_data, "y", alpha=0.5, lw=2, label="True Dead")

            ax.set_xlabel("Time in days")
            ax.set_ylabel("Number of people per category")

            ax.yaxis.set_tick_params(length=0)
            ax.xaxis.set_tick_params(length=0)

            for spine in ("top", "right", "bottom", "left"):
                ax.spines[spine].set_visible(False)

            ax2 = ax.twinx()
            l4 = ax2.plot(
                t, L, "g", alpha=0.5, lw=2, label="lockdown intensity over time"
            )
            ax.set_ylabel("Lockdown intensity")

            lns = l1 + l2 + l3 + l4
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc=0)

        """
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
        """

        L, S, I, R, D = self.predict()
        set_ax(ax1, L, S, I, R, D)
        ax1.set_title("before fitting")

        self.fit()

        L, S, I, R, D = self.predict()
        set_ax(ax2, L, S, I, R, D)
        ax2.set_title("after fitting")

        self.optimize()

        L, S, I, R, D = self.predict()
        set_ax(ax3, L, S, I, R, D)
        ax3.set_title("after optimizing")

        fig.savefig(
            "covid-model-simple.jpg",
            format="jpeg",
            dpi=100,
            bbox_inches="tight",
        )

        plt.show()
