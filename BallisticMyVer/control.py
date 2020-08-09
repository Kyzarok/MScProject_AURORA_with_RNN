# This is the control program
# Running this runs the ballistic task

import numpy as np
import torch
import individual
import random


def ae():
    

def AURORA_ballistic_task():
    # Randomly generate some controllers
    batch_size = 200                                # Initial size of population
    pop = []                                        # Population
    for b in range(batch_size):
        new_indiv = individual.indiv()
        pop.append(new_indiv)

    # Collect sensory data of the generated controllers. In the case of the ballistic task this is the trajectories but any sensory data can be considered
    for member in pop:
        genotype = [random.uniform(0, 1), random.uniform(0, 1)]
        member.eval(genotype)

    # The collected sensory data makes up the first dataset

    # Train the dimension reduction algorithm (the Autoencoder) on the dataset
    # Sensory data is then projected into the latent space, this is used as the behavioural descriptor

    
    # Randomly intialize the QD algorithm
    # QD loop
    # For each iteration: 1. Randomly select a controller
    #                     2. Mutate it
    #                     3. Evaluate the new controller and try to put it into the collection

    # Retrain Autoencoder after a number of QD iterations


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-control', '--CONTROL', type=str, default='ballistic', help = "Which simulation you want to run")
    args = parser.parse_args()
    if args.control == "ballistic":
        AURORA_ballistic_task()