import os
import sys
import glob
import datetime
import matplotlib.pyplot as plt


def time_to_timestr():
    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    return timestr
 