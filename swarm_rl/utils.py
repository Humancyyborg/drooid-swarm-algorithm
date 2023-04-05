import datetime
import random


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S-{fname}'):
    # This creates a timestamped filename so we don't overwrite our good work
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def generate_seeds(num_seeds):
    return [random.randrange(0, 9999) for _ in range(num_seeds)]
