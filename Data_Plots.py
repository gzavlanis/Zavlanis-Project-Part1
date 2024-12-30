from matplotlib import pyplot as plt

def scatter_plot(width, height, x, y, xlabel, ylabel, title, directory):
    plt.figure(figsize = (width, height))
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
    plt.show()
    plt.savefig(directory)
