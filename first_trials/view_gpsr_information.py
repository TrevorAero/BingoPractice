import numpy as np
import glob

from bingo.evolutionary_optimizers.evolutionary_optimizer import \
            load_evolutionary_optimizer_from_file as leoff


FILES = glob.glob("./*.pkl")
pickles = [leoff(FILE) for FILE in FILES]
gens = [pickle.generational_age for pickle in pickles]
final_pickle = pickles[np.argmax(gens)]
import pdb;pdb.set_trace()

