# Modeling an SIR model based on the Kermack-McKendrick differential equations
# Based on https://towardsdatascience.com/how-quickly-does-an-influenza-epidemic-grow-7e95786115b3
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from symfit import variables, parameters, Fit, D, ODEModel, Parameter

def create_model(beta, gamma, delta, t, init_conds):
    S, I, R, D = [init_conds[0]], [init_conds[1]], [init_conds[2]], [init_conds[3]]
    dt = t[2] - t[1]
    for step in t[:-1]:
        S.append(S[-1] - (beta * S[-1] * I[-1]) * dt)
        I2 = I[-1] + (beta * S[-1] * I[-1] - gamma * I[-1]) * dt - delta * I[-1] * dt
        I.append(I2)
        R.append(R[-1] + (gamma * I2) * dt)
        D.append(D[-1] + (delta * I2) * dt)
    return np.stack([S, I, R, D]).T

def simulate_model(duration, dt, beta, gamma, delta, init_conds):
    t = np.linspace(0, duration, int(duration/dt))

    model = create_model(beta, gamma, delta, t, init_conds)

    # model = odeint(derivative, init_conds, t, args=(beta, gamma, delta))
    return model

def derivative(y, t, beta, gamma, delta):
    S, I, R, D = y
    dSdt = - beta * S * I
    dIdt = beta * S * I - gamma * I - delta * I
    dRdt = gamma * I
    dDdt = delta * I
    return dSdt, dIdt, dRdt, dDdt

def plot_model(model):
    plt.figure(facecolor='w')
    plt.plot(model)
    plt.title("SIR Model")
    plt.legend(['Susceptible', 'Infected', 'Recovered', 'Dead'])
    plt.xlabel('Time')
    plt.ylabel('Population Size')
    plt.show()

def find_best_fit_params(data, init_conds):
    S, I, R, De, t = variables('S, I, R, De, t')
    
    beta = Parameter('beta', value = 0.005, min = 0, max = 0.01)
    gamma = Parameter('gamma', value = 0.05, min = 0.01, max = 0.4)
    delta = Parameter('delta', value = 0.05, min = 0.0, max = 0.4)

    model_dict = {
        D(S, t): -beta * S * I,
        D(I, t): beta * S * I - gamma * I - delta * I,
        D(R, t): gamma * I,
        D(De, t): delta * I
    }
    ode_model = ODEModel(model_dict, initial={t: 0, S: init_conds[0], I: init_conds[1], R: init_conds[2], De: init_conds[3]})
    fit = Fit(ode_model, t=np.linspace(0, 100, int(100/0.1)), I=data[1], S=data[0], R=data[2], De=data[3])
    fit_result = fit.execute()

    return fit_result.params

def calc_error(exp, real):
    return np.abs((exp-real)/real)

def print_model_errors(exp, real):
    for param in exp:
        print("Error in {}: {}".format(param, calc_error(exp[param], real[param])))

def input_params():
    params = {}
    print("Enter beta.")
    params["beta"] = float(input())
    print("Enter gamma.")
    params["gamma"] = float(input())
    print("Enter delta.")
    params["delta"] = float(input())
    return params

def input_population():
    print("Enter population size (recommendation is 1000)")
    pop_size = int(input())
    if pop_size < 2:
        pop_size = 1000

    return pop_size

def main():
    params = input_params()
    
    pop_size = input_population()

    print("Simulating the SIR model for an initial population of 1000.")
    print("Branching Factor = {}".format(params["beta"]/params["gamma"]))

    init_conds = [pop_size - 1, 1, 0, 0]
    model = simulate_model(100, 0.1, params["beta"], params["gamma"], params["delta"], init_conds)
    plot_model(model)

    exp_params = find_best_fit_params(np.ndarray.transpose(model), init_conds)
    print(exp_params)
    print_model_errors(exp_params, params)

if __name__ == "__main__":
    main()