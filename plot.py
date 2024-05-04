import matplotlib.pyplot as plt

def plot(costs1, times1, costs2, times2, param, DS):
    
    plt.figure(figsize=(10, 5))


    # Plotting the cost vs time
    plt.plot(times1, costs1, label=f"GLocalKD")
    plt.plot(times2, costs2, label=f"w/ DiffPool")
    plt.xlabel('Compute Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title(label=f"Accuracy over time, dataset = {DS}")
    plt.legend()
    plt.grid(True)
    # Save the plot to a file

    path = f"plots/plot150_{DS}"

    plt.savefig(path)