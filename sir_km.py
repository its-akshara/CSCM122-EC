# Modeling an SIR model based on the Kermack-McKendrick differential equations
# Based on https://towardsdatascience.com/how-quickly-does-an-influenza-epidemic-grow-7e95786115b3
import numpy as np
import matplotlib.pyplot as plt

def create_model(beta, gamma, t):
    init_S = 999
    init_I = 1
    init_R = 0
    S, I, R = [init_S], [init_I], [init_R]
    dt = t[2] - t[1]
    for step in t[:-1]:
        S.append(S[-1] - (beta * S[-1] * I[-1]) * dt)
        I2 = I[-1] + (beta * S[-1] * I[-1] - gamma * I[-1]) * dt
        I.append(I2)
        R.append(R[-1] + (gamma * I2) * dt)
    return np.stack([S, I, R]).T

def simulate_model(duration, dt, beta, gamma):
    t = np.linspace(0, duration, int(duration/dt))

    model = create_model(beta, gamma, t)

    plot_model(model)

def plot_model(model):
    plt.figure(facecolor='w')
    plt.plot(model)
    plt.title("SIR Model")
    plt.legend(['Susceptible', 'Infected', 'Recovered'])
    plt.xlabel('Time')
    plt.ylabel('Population Size')
    plt.show()

def main():
    print("Enter beta.")
    beta = float(input())
    print("Enter gamma.")
    gamma = float(input())

    print("Simulating the SIR model for an initial population of 1000.")
    print("Branching Factor = {}".format(beta/gamma))

    simulate_model(100, 0.1, beta, gamma)

if __name__ == "__main__":
    main()