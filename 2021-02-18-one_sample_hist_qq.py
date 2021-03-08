import random
import time
import scipy.stats as st
from scipy.stats import norm
import numpy as np
from matplotlib import gridspec
import gc
import matplotlib.pyplot as plt


plt.ion()
i = 0

figure = plt.figure()
figure.set_figheight(9)
figure.set_figwidth(18)

spec = gridspec.GridSpec(ncols=2, nrows=3,
                         height_ratios=[10, 1, 1],
                         hspace=0.6)

population = list(np.random.normal(15, 3.5, 1000))
population_b = list(np.random.normal(26, 3.2, 240))
population_c = list(np.random.normal(29, 3.1, 280))
population_d = list(np.random.normal(13, 1.5, 800))

# Try including different kinds of populations here
population = population #+ population_b + population_c + population_d

x = []
x.append(population[random.randint(1, len(population)-1)])


def move_figure(f, x_loc, y_loc):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = plt.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x_loc, y_loc))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x_loc, y_loc))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x_loc, y_loc)


for i in range(1, len(population)):
    x.append(population[random.randint(0, len(population))])

    # bins = round (np.log(len(x), None)) + 1 # Sturge's rule
    bins = round(np.cbrt(len(x)) * 2)  # Rice's rule

    ax0 = figure.add_subplot(spec[0])
    plt.hist(x, bins=bins,
             align="mid",
             alpha=0.5,
             color="g",
             lw=1,
             ec="black",
             # density=True
             )

    mu, std = st.norm.fit(x)
    xmin, xmax = plt.xlim()

    plt.xlabel("Value of variable x")
    plt.ylabel("Number of observations")

    y = np.linspace(xmin, xmax, len(x))

    p = norm.pdf(y, mu, std)
    p = (p * np.cbrt(len(x)**2)) * (np.std(x)*2.5)

    plt.plot(y, p, "k", linewidth=1)

    plt.title("Sample A: AVG = {:05.2f} |"
              " SD = {:05.2f}".format(np.average(x), np.std(x)))

    plt.xlim(-10, 40)

    ax2 = figure.add_subplot(spec[2])
    plt.boxplot(x, vert=False, widths=0.5)
    plt.xlim(-10, 40)

    ax1 = figure.add_subplot(spec[1])
    res = st.probplot(x, plot=plt)
    ax1.get_lines()[0].set_markeredgecolor('g')
    ax1.get_lines()[0].set_markerfacecolor('g')
    ax1.get_lines()[0].set_markersize(3)
    ax1.get_lines()[1].set_linewidth(1.0)
    ax1.get_lines()[1].set_color("b")
    # plt.xlim(-3, 3)
    # plt.ylim(-10, 40)
    plt.title("Q-Q plot")

    ax4 = figure.add_subplot(spec[4])
    ax4.axis("off")
    props = dict(boxstyle="round", facecolor="grey", alpha=0.1)
    ks_test = [0, 0]
    sw_test = [0, 0]

    if len(x) > 2:
        ks_test = st.kstest(x, np.random.normal(np.average(x),
                                                np.std(x),
                                                len(x)))
        sw_test = st.shapiro(x)

    textbox_string_a = "Sample size (n) = {:05d} \n" \
                       "Kolmogorov-Smirnov test: D = {:05.2f}" \
                       " p = {:05.3f} \n" \
                       "Shapiro-Wilk test: W = {:05.2f}" \
                       " p = {:05.3f} \n" \
                       "Skewness = {:06.2f} \n" \
                       "Kurtosis = {:06.2f}" \
                        .format(len(x), ks_test[0], ks_test[1], sw_test[0],
                                sw_test[1], st.skew(x, None),
                                st.kurtosis(x, None))

    ax4.text(0.00, 0.00, textbox_string_a, bbox=props)

    ax5 = plt.subplot2grid(shape=(3, 3), loc=(2, 2), rowspan=2)
    # ax5 = figure.add_subplot(spec[3])

    bins = round(np.cbrt(len(population)) * 2)  # Rice's rule

    plt.hist(population,
             bins=bins,
             color="g",
             align="mid",
             alpha=0.3,
             lw=0.1,
             ec="black")

    plt.xlim(-10, 40)

    ax5.set_title(("Population histogram, N = " + str(len(population)) +
                   ", AVG = {:05.2f}, SD = {:05.2f}"
                   .format(np.average(population), np.std(population))),
                  fontdict={"fontsize": 8})

    ax5.tick_params(axis='both', which='both', labelsize=6)

    figure.canvas.draw()
    figure.canvas.flush_events()
    plt.clf()
    if i <= 3:
        move_figure(figure, 45, 20)
    time.sleep(0.01)
    gc.collect()
    del gc.garbage[:]
