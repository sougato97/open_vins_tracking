import numpy as np

def plot_trajectories_2d(title, *trajectories):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)

    for ii, trajectory in enumerate(trajectories):
        xs, ys = trajectory[:,1].tolist(), trajectory[:,2].tolist()
        line, = ax.plot(xs, ys, label=str(ii+1))
        ax.legend()

    plt.show()

def plot_bar_3d_flat(values, xlabels, ylabels, xdist, ydist, axis_labels=['X', 'Y', 'Z'], rotate=(30,-45)):
    dx = 4.2
    xlabels_c = []

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    distances = range(0, len(xlabels)*xdist, xdist)
    xs = []
    for ii, distance in enumerate(distances):
        xs.append( distance )
        xs.append( distance+dx )
        xlabels_c.append(xlabels[ii])
        xlabels_c.append('')

    ys = range(0, len(ylabels)*ydist, ydist)

    for ii, y in enumerate(ys):
        zs = values[ii]
        ax.bar(xs, zs, zs=y, zdir='y', color=colors[:2], width=4, alpha=0.6, tick_label=xlabels_c)
        # ax.bar(xs, zs, zs=y, zdir='y', color=colors[ii*2:ii*2+2], width=4, alpha=0.6, tick_label=xlabels_c)

    ax.view_init(50, -45)
    ax.set_yticks(ys)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    plt.show()

def plot_bar_3d(values, xlabels, ylabels, xdist, ydist, axis_labels=['X', 'Y', 'Z'], rotate=(50,-45), colortile=(12,1)):
    ddx = 2
    xlabels_c = []

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    distances = range(0, len(xlabels)*xdist, xdist)
    xs = []
    for ii, distance in enumerate(distances):
        xs.append( distance )
        xs.append( distance+ddx )
        xlabels_c.append(xlabels[ii])
        xlabels_c.append('')

    ys = range(0, len(ylabels)*ydist, ydist)

    xs_, ys_ = np.meshgrid(xs, ys)
    xs_, ys_ = xs_.ravel(), ys_.ravel()

    dx = 1
    dy = 1
    values = np.asarray(values).ravel()
    zs = np.zeros_like(values)

    colors = cm.tab10(range(2))
    colors = np.tile(np.asarray(colors), colortile)

    ax.bar3d(xs_, ys_, zs, dx, dy, values, shade=True, alpha=0.8,color=colors)
    # ax.bar3d(xs_, ys_, zs, dx, dy, values, shade=True, alpha=0.8)

    ax.view_init(*rotate)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels_c)
    ax.set_yticks(ys)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    plt.show()
