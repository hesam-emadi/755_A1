from matplotlib import pyplot as plt
import numpy as np


def graph(title, labels, results, metrics, x_param):
    plt.figure()
    plt.title(title)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.grid()
    axes = plt.axes()

    x_axis = np.array(results['param_' + x_param].data, dtype=float)

    for scorer, color in zip(sorted(metrics), ['b', 'g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):

            sample_score_mean = results['mean_{}_{}'.format(sample, scorer)]
            axes.plot(x_axis, sample_score_mean, style, color=color,
                      label="{} {}".format(scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_{}'.format(scorer)][best_index]

        axes.annotate("{0:.2f}".format(best_score), (x_axis[best_index], best_score))

    plt.legend(loc="best")
    # plt.grid('off')
    plt.savefig(title + '.png')
    plt.show()
