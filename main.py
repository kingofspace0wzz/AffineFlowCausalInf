### Affine Flow-based causal discovery & inference experiments 
#
#
#

import argparse
import os

from runners.cause_effect_pairs_runner import RunCauseEffectPairs
from runners.counterfactual_trials import counterfactuals
from runners.intervention_trials import intervention
from runners.simulation_runner import RunSimulations


def parse_input():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Dataset to run synthetic CD experiments on. Should be either linear, '
                             'hoyer2009 or nueralnet_l1 or all to run all')
    parser.add_argument('--nSims', type=int, default=250, help='Number of synthetic simulations to run')
    parser.add_argument('--resultsDir', type=str, default='results/', help='Path for saving results.')

    parser.add_argument('-s', '--simulation', action='store_true', help='run the CD exp on synthetic data')
    parser.add_argument('-p', '--pairs', action='store_true', help='Run Cause Effect Pairs experiments')
    parser.add_argument('-i', '--intervention', action='store_true', help='run intervention exp on toy example')
    parser.add_argument('-c', '--counterfactual', action='store_true', help='run counterfactual exp on toy example')

    return parser.parse_args()


if __name__ == '__main__':
    # parse command line arguments
    args = parse_input()

    # create results directory
    os.makedirs(args.resultsDir, exist_ok=True)

    if args.simulation:
        # run proposed method as well as baseline methods on simulated data
        # and save the results as pickle files which can be used later to plot Fig 1.
        import pickle
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        print('Running {} synthetic experiments. Will run {} simulations'.format(args.dataset, args.nSims))
        algos = ['FlowCD', 'LRHyv', 'notears', 'RECI', 'ANM']

        # chose the form of f (equation 11)
        if args.dataset == 'all':
            exp_list = ['linear', 'hoyer2009', 'nueralnet_l1']
        elif args.dataset in ['linear', 'hoyer2009', 'nueralnet_l1']:
            exp_list = [args.dataset]
        else:
            raise ValueError('Unknown dataset: {}'.format(args.dataset))

        for exp in exp_list:
            nvals = [25, 50, 75, 100, 150, 250, 500]
            results = []
            causal_mechanism = exp
            nsims = args.nSims
            print('Mechanism: {}'.format(causal_mechanism))
            for n in nvals:
                print('### {} ###'.format(n))
                results.append(
                    RunSimulations(nSims=nsims, nPoints=n, causal_mechanism=causal_mechanism, algolist=algos))

            # save results
            pickle.dump(results, open(args.resultsDir + causal_mechanism + "_results.p", 'wb'))

        # if args.plot and not args.pairs and not args.intervention and not args.counterfactuals:
        # produce a plot of synthetic results

        title_dic = {'nueralnet_l1': "Neural network" + "\n" + r"$x_2 = \sigma \left ( \sigma ( x_1) + n_2 \right)$",
                     'linear': "Linear SEM\n" + r"$x_2 = x_1 + n_2 $",
                     'hoyer2009': "Nonlinear SEM\n" + r"$x_2 = x_1 + \frac{1}{2} x_1^3 + n_2 $"}

        label_dict = {'FlowCD': 'Affine flow LR',
                      'LRHyv': 'Linear LR',
                      'RECI': 'RECI',
                      'ANM': 'ANM',
                      'notears': 'NO-TEARS'}

        # define some parameters
        nvals = [25, 50, 75, 100, 150, 250, 500]
        algos = ['FlowCD', 'LRHyv', 'notears', 'RECI', 'ANM']
        sim_type = ['linear', 'hoyer2009', 'nueralnet_l1']

        res_all = {s: {a: [] for a in algos} for s in sim_type}

        for s in sim_type:
            results = pickle.load(open(args.resultsDir + s + '_results.p', 'rb'))
            for a in algos:
                for n in range(len(nvals)):
                    res_all[s][a].append(np.mean(results[n][a] == results[n]['true']))

        # prepare plot
        sns.set_style("whitegrid")
        sns.set_palette('deep')

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

        for a in algos:
            ax1.plot(nvals, res_all['linear'][a], marker='o')
            ax2.plot(nvals, res_all['hoyer2009'][a], marker='o')
            ax3.plot(nvals, res_all['nueralnet_l1'][a], marker='o', label=label_dict[a])

        fontsize = 12
        font_xlab = 10

        ax1.set_title(title_dic['linear'], fontsize=fontsize)
        ax2.set_title(title_dic['hoyer2009'], fontsize=fontsize)
        ax3.set_title(title_dic['nueralnet_l1'], fontsize=fontsize)

        ax1.set_xlabel('Sample size', fontsize=font_xlab)
        ax2.set_xlabel('Sample size', fontsize=font_xlab)
        ax3.set_xlabel('Sample size', fontsize=font_xlab)

        ax1.set_ylabel('Proportion correct', fontsize=font_xlab)
        ax2.set_ylabel('Proportion correct', fontsize=font_xlab)
        ax3.set_ylabel('Proportion correct', fontsize=font_xlab)

        fig.legend(  # The labels for each line
            loc="center right",  # Position of legend
            borderaxespad=0.2,  # Small spacing around legend box
            title="Algorithm"  # Title for the legend
        )

        plt.tight_layout()
        plt.subplots_adjust(right=0.87)
        plt.savefig(os.path.join(args.resultsDir, 'CausalDiscSims.pdf'), dpi=300)

    if args.pairs:
        # Run proposed method on CauseEffectPair dataset
        # Percentage of correct causal direction is printed to standard output,
        # and updated online after each new pair.
        # The values for baseline methods were taken from their respective papers.
        print('running cause effect pairs experiments ')
        RunCauseEffectPairs()

    if args.intervention:
        # Run proposed method to perform interventions on the toy example described in the manuscript
        print('running interventions on toy example')
        intervention(dim=4, results_dir=args.resultsDir)

    if args.counterfactual:
        # Run proposed method to perform counterfactuals on the toy example described in the manuscript
        print('running counterfactuals on toy example')
        counterfactuals(results_dir=args.resultsDir)
