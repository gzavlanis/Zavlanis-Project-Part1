from matplotlib import pyplot as plt

def scatter_plot(width, height, x, y, xlabel, ylabel, title):
    plt.figure(figsize = (width, height))
    plt.scatter(x, y, alpha = 0.5, s = 70)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def scree_plot():
    pass