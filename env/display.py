import os
from pyvirtualdisplay import Display

try:
    ORIGIN_DISP = os.environ['DISPLAY']
except:
    ORIGIN_DISP = None # Docker in case
DISP = Display(visible=False, size=(1400, 900))
DISP.start()
VIRTUAL_DISP = os.environ['DISPLAY']
os.environ['DISPLAY'] = ORIGIN_DISP


def virtual_display(func):
    def wrapper(*args, **kwargs):
        os.environ['DISPLAY'] = VIRTUAL_DISP
        func(*args, **kwargs)
        os.environ['DISPLAY'] = ORIGIN_DISP
    return wrapper