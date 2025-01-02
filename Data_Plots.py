from matplotlib import pyplot as plt

def scatter_plot(x, y, xlabel, ylabel, title, directory):
    plt.scatter(x, y, alpha = 0.5, s = 70)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.savefig(directory)

def scree_plot(pc_values, explained_variance, xlabel, ylabel, title, directory, color):
    plt.plot(pc_values, explained_variance, 'o-', linewidth = 2, color = color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()
    plt.savefig(directory)

def plot_of_actual_and_predicted(X, y, predictions, xlabel, ylabel, title, directory):
    plt.scatter(X, y, color = 'blue', label = 'Actual')
    plt.scatter(X, predictions, color = 'red', label = 'Predicted')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(directory)