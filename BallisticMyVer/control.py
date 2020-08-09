# This is the control program
# Running this runs the ballistic task

import numpy as np
import torch
import individual
import random
import my_ae
from tqdm.autonotebook import tqdm
from itertools import chain
import math

POPULATION_INITIAL_SIZE = 200
POPULATION_LIMIT = 10000

NUM_EPOCH = 25000
BATCH_SIZE = 20000

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, pop_data):
        self.data = []
        # Create image dataset from trajectory data
        for member in pop_data:
            tmp = []
            traj = member.get_traj()
            for t in traj:
                tmp.append(traj[0])
                tmp.append(traj[1])
            self.data.append(tmp)
        print("Dataset constructed")
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def train_ae(encoder, decoder, population):
    # Create dataset for loading
    train_data = MyDataset(population)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
    optimizer = torch.optim.Adagrad(chain(encoder.parameters(), decoder.parameters()), lr=0.1, lr_decay=0.9)
    print("Beginning Training of Autoencoder")
    for epoch in range(NUM_EPOCH):
        if epoch % 1000 == 0:
            print("At epoch " + str(epoch))
        train_loader = tqdm(train_loader)
        for data in (train_loader):
            optimizer.zero_grad()
            z = encoder(data)
            outputs = decoder(z)

            diffs = (data - outputs)**2
            loss = sum(diffs)/len(diffs)

            loss.backward()
            optimizer.step()
    print("Training Complete")
    return encoder, decoder
    

def AURORA_ballistic_task():
    # Randomly generate some controllers
    batch_size = POPULATION_INITIAL_SIZE            # Initial size of population
    pop = []                                        # Container for population
    print("Creatin population container")
    for b in range(batch_size):
        new_indiv = individual.indiv()
        pop.append(new_indiv)

    # Collect sensory data of the generated controllers. In the case of the ballistic task this is the trajectories but any sensory data can be considered
    # The collected sensory data makes up the first dataset
    print("Evaluating current population")
    for member in pop:
        genotype = [random.uniform(0, 1), random.uniform(0, 1)]
        member.eval(genotype)

    # Create the dimension reduction algorithm (the Autoencoder)
    enc = my_ae.Encoder()
    dec = my_ae.Decoder()

    # Train the dimension reduction algorithm (the Autoencoder) on the dataset
    enc, dec = train_ae(enc, dec, pop)

    # Sensory data is then projected into the latent space, this is used as the behavioural descriptor
    # Use the now trained Autoencoder to get the behavioural descriptors
    for member in pop:
        tmp = []
        traj = member.get_traj()
        for t in traj:
            tmp.append(traj[0])
            tmp.append(traj[1])
        member_bd = enc(tmp)
        member.set_bd(member_bd)
    
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