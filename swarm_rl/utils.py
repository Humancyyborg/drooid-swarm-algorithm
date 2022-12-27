import datetime


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S-{fname}'):
    # This creates a timestamped filename so we don't overwrite our good work
    return datetime.datetime.now().strftime(fmt).format(fname=fname)
