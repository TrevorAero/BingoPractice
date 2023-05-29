import numpy as np
import matplotlib.pyplot as plt
import glob

from bingo.stats.pareto_front import ParetoFront as pf

FILES = glob.glob("./*.pkl")
pickles = [pf(FILE) for FILE in FILES]
upd = [pickle.update for pickle in pickles]




print(" FITNESS  COMPLEXITY  EQUATION")
for member in pickles:
    print("%.3e  " % member.fitness, member.get_complexity(),
          "     f(X_0) =", member)


    
