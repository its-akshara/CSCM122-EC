import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import tkinter as tk

# Create entries for the input box
def makeform(root, fields):
    entries = []
    for field in fields:
        row = tk.Frame(root)
        lab = tk.Label(row, width=15, text=field, anchor='w')
        ent = tk.Entry(row)
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries.append((field, ent))
    return entries

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Plot the SIR model graph
def plot(entries):
    # N, beta, gamma, respectively
    args_list = [0, 0, 0]

    i = 0
    for entry in entries:
        field = entry[0]
        input  = entry[1].get()
        args_list[i] = float(input)
        i += 1

    # Total population
    N = args_list[0]
    # Contact rate
    beta = args_list[1]
    # Mean recovery rate (in 1/days)
    gamma = args_list[2]
    
    # Initial number of infected individuals
    I0 = 1
    # Initial number of recovered individuals
    R0 = 0
    # Everyone else, S0, is susceptible to infection initially
    S0 = N - I0 - R0
    
    # A grid of time points (in days)
    t = np.linspace(0, 200, 200)
    
    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()

def main():
    root = tk.Tk()
    root.title('SIR Model')
    fields = 'Total Population', 'Beta', 'Gamma'
    ents = makeform(root, fields)
    b1 = tk.Button(root, text='Plot', command=(lambda e=ents: plot(e)))
    b1.pack(side=tk.RIGHT, padx=5, pady=5)

    def cleanup():
        root.destroy()
        plt.close('all')
    b2 = tk.Button(root, text='Quit', command=cleanup)
    b2.pack(side=tk.RIGHT, padx=5, pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    main()
