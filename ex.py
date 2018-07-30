import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', 
        call_pdb=1)

for i in range(-10,10):
    a = 1/i
    print("a is :", a)

