import signal
from . import config

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
def debug_print(message):
    if config.DEBUG:
        print(message)