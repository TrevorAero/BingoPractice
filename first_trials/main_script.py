
# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness \
    import LocalOptFitnessFunction

from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      AGraphCrossover, \
                                      AGraphMutation, \
                                      ExplicitRegression, \
                                      ExplicitTrainingData
POP_SIZE = 100
STACK_SIZE = 16


def execute_generational_steps():
    data = np.load("training_data.npy")
    x = data[:,0].reshape((-1,1))
    y = data[:,1].reshape((-1,1))

    training_data = ExplicitTrainingData(x, y)

    component_generator = ComponentGenerator(x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator,
                                       use_simplification=True
                                       )

    fitness = ExplicitRegression(training_data=training_data)
    optimizer = ScipyOptimizer(fitness, method='lm')
    local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
    evaluator = Evaluation(local_opt_fitness)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    archipelago = SerialArchipelago(island)

    opt_result = archipelago.evolve_until_convergence(max_generations=1000,
                                                      fitness_threshold=1.0e-4,
                                                      convergence_check_frequency=50,
                                                      checkpoint_base_name="checkpoint")
    if opt_result.success:
        print(archipelago.get_best_individual().get_formatted_string("console"))
    else:
        print("Failed.")

    print(opt_result.ea_diagnostics)


def main():
    execute_generational_steps()


if __name__ == '__main__':
    import random
    random.seed(7)
    np.random.seed(7)
    main()