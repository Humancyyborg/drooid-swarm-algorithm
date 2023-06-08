import argparse
import os
import pickle
import sys
from os.path import join
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from sample_factory.utils.utils import ensure_dir_exists
from matplotlib.ticker import FuncFormatter

PAGE_WIDTH_INCHES = 8.2
FULL_PAGE_WIDTH = 1.4 * PAGE_WIDTH_INCHES
HALF_PAGE_WIDTH = FULL_PAGE_WIDTH / 2

plt.rcParams['figure.figsize'] = (FULL_PAGE_WIDTH, 2.5)  # (2.5, 2.0) 7.5， 4

NUM_AGENTS = 8
EPISODE_DURATION = 16  # seconds
TIME_METRIC_COLLISION = 60  # ONE MINUTE
COLLISIONS_SCALE = ((
                            TIME_METRIC_COLLISION / EPISODE_DURATION) / NUM_AGENTS) * 2  # times two because 1 collision = 2 drones collided
COLLISIONS_OBST_SCALE = ((
                                 TIME_METRIC_COLLISION / EPISODE_DURATION) / NUM_AGENTS)  # Not times two because 1 collision = 1 drone collide with 1 obstacle, and we only talk about drones here

CRASH_GROUND_SCALE = (-1.0 / EPISODE_DURATION)

PLOTS = [
    dict(key='metric/agent_success_rate', name='Agent success rate', label='Average rate'),
    # dict(key='policy_stats/avg_distance_to_goal_1s', name='Avg. distance to the goal', label='Avg. distance(m)', clip_min=0.1, y_scale_formater=[0.1, 0.5, 1.0, 2.0]),
    # dict(key='metric/agent_deadlock_rate', name='Agent deadlock rate', label='Deadlock Rate'),
    dict(key='metric/agent_col_rate', name='Agent collision rate', label='Average rate'),
    # dict(key='metric/agent_obst_col_rate', name='Agent collision w/ obstacles rate'),
    dict(key='o_random/distance_to_goal_1s', name='Distance to goal (random)', label='Distance (m)'),
    dict(key='o_static_same_goal/distance_to_goal_1s', name='Distance to goal (same goal)', label='Distance (m)'),

]

PLOT_STEP = int(5e6)
TOTAL_STEP = int(1e9 + 10000)

# 'blue': '#1F77B4', 'orange': '#FF7F0E', 'green': '#2CA02C', 'red': '#d70000'
# COLOR = ['#1F77B4', '#FF7F0E', '#2CA02C', '#d70000']
COLOR = ['#4477AA', '#EE6677', '#54B345', '#CCBB44', '#AA3377']


def extract(experiments):
    scalar_accumulators = [EventAccumulator(experiment_dir).Reload().scalars for experiment_dir in experiments]

    # Filter non event files
    scalar_accumulators = [scalar_accumulator for scalar_accumulator in scalar_accumulators if
                           scalar_accumulator.Keys()]

    # Get and validate all scalar keys
    all_keys = [tuple(sorted(scalar_accumulator.Keys())) for scalar_accumulator in scalar_accumulators]
    # assert len(set(all_keys)) == 1, \
    #     "All runs need to have the same scalar keys. There are mismatches in {}".format(all_keys)

    keys = all_keys[0]
    all_scalar_events_per_key = [[scalar_accumulator.Items(key)
                                  for scalar_accumulator in scalar_accumulators] for key in keys]

    # Get and validate all steps per key
    x_per_key = [[tuple(scalar_event.step
                        for scalar_event in scalar_events) for scalar_events in all_scalar_events]
                 for all_scalar_events in all_scalar_events_per_key]

    plot_step = PLOT_STEP
    all_steps_per_key = [[tuple(int(step_id) for step_id in range(0, TOTAL_STEP, plot_step))
                          for _ in all_scalar_events]
                         for all_scalar_events in all_scalar_events_per_key]

    for i, all_steps in enumerate(all_steps_per_key):
        assert len(set(
            all_steps)) == 1, "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
            keys[i], [len(steps) for steps in all_steps])

    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

    # Get values per step per key
    values_per_key = [[[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
                      for all_scalar_events in all_scalar_events_per_key]

    interpolated_keys = dict()
    for tmp_id in range(len(PLOTS)):
        key_idx = keys.index(PLOTS[tmp_id]['key'])
        values = values_per_key[key_idx]

        x = steps_per_key[key_idx]
        x_steps = x_per_key[key_idx]

        interpolated_y = [[] for _ in values]

        for i in range(len(values)):
            idx = 0

            tmp_min_step = min(len(x_steps[i]), len(values[i]))
            values[i] = values[i][2: tmp_min_step]
            x_steps[i] = x_steps[i][2: tmp_min_step]

            assert len(x_steps[i]) == len(values[i])
            for x_idx in x:
                while idx < len(x_steps[i]) - 1 and x_steps[i][idx] < x_idx:
                    idx += 1

                if x_idx == 0:
                    interpolated_value = values[i][idx]
                elif idx < len(values[i]) - 1:
                    interpolated_value = (values[i][idx] + values[i][idx + 1]) / 2
                else:
                    interpolated_value = values[i][idx]

                interpolated_y[i].append(interpolated_value)
            assert len(interpolated_y[i]) == len(x)

        print(PLOTS[tmp_id]['key'], interpolated_y[0][:30])

        interpolated_keys[PLOTS[tmp_id]['key']] = (x, interpolated_y)

    return interpolated_keys


def aggregate(path, subpath, experiments, ax, legend_name, group_id):
    print("Started aggregation {}".format(path))

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = join(curr_dir, 'cache')
    cache_env = join(cache_dir, subpath)

    if os.path.isdir(cache_env):
        try:
            with open(join(cache_env, f'{subpath}.pickle'), 'rb') as fobj:
                interpolated_keys = pickle.load(fobj)

        except FileNotFoundError:
            interpolated_keys = extract(experiments=experiments)

    else:
        cache_env = ensure_dir_exists(cache_env)
        interpolated_keys = extract(experiments=experiments)
        with open(join(cache_env, f'{subpath}.pickle'), 'wb') as fobj:
            pickle.dump(interpolated_keys, fobj)

    for i, key in enumerate(interpolated_keys.keys()):
        # if i == (len(interpolated_keys.keys()) - 1) // 2:
        #     set_xlabel = True
        # else:
        #     set_xlabel = False
        set_xlabel = True
        plot(i, interpolated_keys[key], ax[i], set_xlabel, legend_name, group_id)


def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]

    valid_only: put nan in entries where the full-sized window is not available

    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out


def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    '''
    perform one-sided (causal) EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments:

    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds

    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]

    n: int                - number of points in new x grid

    decay_steps: float    - EMA decay factor, expressed in new x grid steps.

    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN

    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid

    '''

    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
    assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
    assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))

    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')

    luoi = 0  # last unused old index
    sum_y = 0.
    count_y = 0.
    xnews = np.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = np.exp(- 1. / decay_steps)
    sum_ys = np.zeros_like(xnews)
    count_ys = np.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sum_y *= interstep_decay
        count_y *= interstep_decay
        while True:
            if luoi >= len(xolds):
                break
            xold = xolds[luoi]
            if xold <= xnew:
                decay = np.exp(- (xnew - xold) / decay_period)
                sum_y += decay * yolds[luoi]
                count_y += decay
                luoi += 1
            else:
                break
        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = sum_ys / count_ys
    ys[count_ys < low_counts_threshold] = np.nan

    return xnews, ys, count_ys


def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    """
    perform symmetric EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments:

    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds

    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]

    n: int                - number of points in new x grid

    decay_steps: float    - EMA decay factor, expressed in new x grid steps.

    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN

    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid

    """
    xs, ys, count_ys = one_sided_ema(xolds, yolds, low, high, n, decay_steps, low_counts_threshold=0)
    # _, ys2, count_ys2 = one_sided_ema(-xolds[::-1], yolds[::-1], -high, -low, n, decay_steps, low_counts_threshold=0)
    # ys2 = ys2[::-1]
    # count_ys2 = count_ys2[::-1]
    # count_ys = count_ys1 + count_ys2
    # ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
    # ys[count_ys < low_counts_threshold] = np.nan
    return xs, ys, count_ys


# def plot(env, key, interpolated_key, ax, count):
def plot(index, interpolated_key, ax, set_xlabel, legend_name, group_id):
    params = PLOTS[index]

    # set title
    title_text = params['name']
    ax.set_title(title_text, fontsize=10)
    if set_xlabel:
        ax.set_xlabel('Total environment steps')

    x, y = interpolated_key
    x = np.array(x)
    y_np = [np.array(yi) for yi in y]
    for i, yi in enumerate(y_np):
        xnew, y_np[i], _ = symmetric_ema(x, yi, x[0], x[-1], n=200, decay_steps=1.0)

    x = xnew

    y_np = [smooth(yi, 1) for yi in y_np]
    y_np = np.stack(y_np)

    logscale = params.get('logscale', False)
    if logscale:
        ax.set_yscale('log', base=2)
        ax.yaxis.set_minor_locator(ticker.NullLocator())  # no minor ticks

    def scientific(x, pos):
        # x:  tick value - ie. what you currently see in yticks
        # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
        return '%.1f' % x

    scientific_formatter = FuncFormatter(scientific)
    ax.yaxis.set_major_formatter(scientific_formatter)
    # TO CHANGE
    ax.set_ylim(-0.02, 1.0 + 0.02)
    # if np.max(y_np) < 0.6:
    #     ax.set_ylim(-0.02, 0.62)
    # elif np.max(y_np) < 0.8:
    #     ax.set_ylim(-0.02, 0.82)

    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter())  # set regular formatting

    coeff = params.get('coeff', 1.0)
    y_np *= coeff

    mutate = params.get('mutate', None)
    if mutate:
        for i in range(y_np.shape[1]):
            y_np[:, i] = mutate(y_np[:, i])

    y_mean = np.mean(y_np, axis=0)
    y_std = np.std(y_np, axis=0)
    y_plus_std = y_mean + y_std
    y_minus_std = y_mean - y_std

    clip_max = params.get('clip_max', None)
    if clip_max:
        y_mean = np.minimum(y_mean, clip_max)
        y_plus_std = np.minimum(y_plus_std, clip_max)
        y_minus_std = np.minimum(y_minus_std, clip_max)

    clip_min = params.get('clip_min', None)
    if clip_min:
        y_mean = np.maximum(y_mean, clip_min)
        y_plus_std = np.maximum(y_plus_std, clip_min)
        y_minus_std = np.maximum(y_minus_std, clip_min)

    def mkfunc(x, pos):
        if x >= 1e9:
            return '%dB' % int(x * 1e-9)
        elif x >= 1e6:
            return '%dM' % int(x * 1e-6)
        elif x >= 1e3:
            return '%dK' % int(x * 1e-3)
        else:
            return '%d' % int(x)

    mkformatter = matplotlib.ticker.FuncFormatter(mkfunc)
    ax.xaxis.set_major_formatter(mkformatter)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    # ax.spines['top'].set_linewidth(0.5)
    # ax.spines['right'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)

    label = params.get('label')
    if label:
        ax.set_ylabel(label, fontsize=10)

        # hide tick of axis
        ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.tick_params(which='major', length=0)

    ax.grid(color='#B3B3B3', linestyle='-', linewidth=0.25, alpha=0.2)
    # ax.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))

    lw = 1.4

    ax.plot(x, y_mean, color=COLOR[group_id], label=legend_name, linewidth=lw, antialiased=True)
    ax.fill_between(x, y_minus_std, y_plus_std, color=COLOR[group_id], alpha=0.25)
    # ax.legend(prop={'size': 6}, loc='lower right')


def hide_tick_spine(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='main path for tensorboard files', default=os.getcwd())
    parser.add_argument('--output', type=str,
                        help='aggregation can be saved as tensorboard file (summary) or as table (csv)', default='csv')

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        raise argparse.ArgumentTypeError('Parameter {} is not a valid path'.format(path))

    # TO CHANGE
    subpaths = sorted(os.listdir(path))
    # subpaths = ['20%', '40%', '60%', '80%']
    legend_name = sorted(['20%', '40%', '60%', '80%'])
    # legend_name = sorted(['1', '2', '6', '16', '31'])
    all_experiment_dirs = {}
    for subpath in subpaths:
        if subpath not in all_experiment_dirs:
            all_experiment_dirs[subpath] = []

        for filename in Path(args.path + "/" + subpath).rglob('*.tfevents.*'):
            experiment_dir = os.path.dirname(filename)
            all_experiment_dirs[subpath].append(experiment_dir)

    if args.output not in ['summary', 'csv']:
        raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(args.output))

    # TO CHANGE
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    # ax = (ax1, ax2, ax3, ax4)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax = (ax1, ax2, ax3, ax4)


    for i in range(len(all_experiment_dirs)):
        aggregate(path, subpaths[i], all_experiment_dirs[subpaths[i]], ax, legend_name[i], i)

    handles, labels = ax[0].get_legend_handles_labels()
    # TO CHANGE
    lgd = fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.07), fontsize=10)
    lgd.set_in_layout(True)

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(wspace=0.15, hspace=0.25)
    # plt.margins(0, 0)

    save_dir = os.path.join(os.getcwd(), 'final_plots')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    figname = 'scale_obst_density.pdf'
    plt.savefig(os.path.join(save_dir, figname), format='pdf', bbox_inches='tight', pad_inches=0.01)

    return 0


if __name__ == '__main__':
    sys.exit(main())
