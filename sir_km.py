# Modeling an SIR model based on the Kermack-McKendrick differential equations
# Based on https://towardsdatascience.com/how-quickly-does-an-influenza-epidemic-grow-7e95786115b3
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, interpolate
from scipy import optimize
from scipy.odr import Model, Data, ODR
from scipy.integrate import odeint
    
init_S = 999
init_I = 1
init_R = 0


def create_model_manual(beta, gamma, t, init_conds):
    S, I, R = [init_conds[0]], [init_conds[1]], [init_conds[2]]
    dt = t[2] - t[1]
    for step in t[:-1]:
        S.append(S[-1] - (beta * S[-1] * I[-1]) * dt)
        I2 = I[-1] + (beta * S[-1] * I[-1] - gamma * I[-1]) * dt
        I.append(I2)
        R.append(R[-1] + (gamma * I2) * dt)
    return np.stack([S, I, R]).T

def simulate_model(duration, dt, beta, gamma, init_conds):
    t = np.linspace(0, duration, int(duration/dt))

    model_manual = create_model_manual(beta, gamma, t, init_conds)

    plot_model(model_manual)

    model = odeint(derivative, init_conds, t, args=(beta, gamma))
    return model, model_manual

def derivative(y, t, beta, gamma):
    S, I, R = y
    dSdt = - beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def plot_model(model):
    plt.figure(facecolor='w')
    plt.plot(model)
    plt.title("SIR Model")
    plt.legend(['Susceptible', 'Infected', 'Recovered'])
    plt.xlabel('Time')
    plt.ylabel('Population Size')
    plt.show()

class ParameterRecoverer():
    def __init__(self, model_data, init_conds):
        self.model_data = model_data # t, S, I, R
        self.init_conds = init_conds
    
    def ode(self, y, t, beta, gamma):
        return derivative(y, t, beta, gamma)

    def model(self, t, p):
        return odeint(self.ode, self.init_conds, t, args=(p,))

    def f_resid(self, p):
        return self.model_data - self.model(self.model_data, p)

    # def best_fit_params(self, guess):
    #     return optimize.curve_fit(self.ode, )

def main():
    print("Enter beta.")
    beta = 0.001#float(input())
    print("Enter gamma.")
    gamma = 0.1#float(input())
    print("Enter population size (recommendation is 1000)")
    pop_size = 1000#int(input())
    if pop_size < 2:
        pop_size = 1000

    print("Simulating the SIR model for an initial population of 1000.")
    print("Branching Factor = {}".format(beta/gamma))

    init_conds = [pop_size - 1, 1, 0]
    model_data, manual = simulate_model(100, 0.1, beta, gamma, init_conds)

    # param_recoverer = ParameterRecoverer(model_data.ravel(), init_conds)

    # print("Estimated (beta, gamma) is: ")
    # print(param_recoverer.best_fit_params([0.002, 0.2]))

    # def SIR_ODE(params, t):
    #     beta, gamma = params
    
    #     def deriv(y, t):
    #         S, I, R = y
    #         dSdt = - beta * S * I
    #         dIdt = beta * S * I - gamma * I
    #         dRdt = gamma * I
    #         return [dSdt, dIdt, dRdt]

    #     y = odeint(deriv, init_conds, t)
    #     return y[:,1]

    # print(np.shape(model_data))
    # t = np.linspace(0, 100, int(100/0.1))
    # # data = Data(np.repeat(t, 3), model_data)
    # data = Data(t, model_data)
    # model = Model(SIR_ODE)
    # guess = [0.002, 0.2]
    # odr = ODR(data, model, guess)
    # odr.set_job(2)
    # out = odr.run()
    # f = plt.figure()
    # p = f.add_subplot(111)
    # p.plot(t, model_data)
    # print(out.beta)
    # p.plot(t, SIR_ODE(out.beta, t))
    # plt.show()

if __name__ == "__main__":
    main()