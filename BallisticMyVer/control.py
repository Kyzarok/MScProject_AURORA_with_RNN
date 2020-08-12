# This is the control program
# Running this runs the ballistic task

import numpy as np
# import torch
import individual
import random
# import my_ae
# from tqdm.autonotebook import tqdm
# from itertools import chain
import math
from original_ae import AE
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import matplotlib.pyplot as plt

POPULATION_INITIAL_SIZE = 200
POPULATION_LIMIT = 10000

NUM_EPOCH = 25000
BATCH_SIZE = 20000

    
def make_batches(population):
    pop_left = len(population)
    num_of_full_batches = pop_left % BATCH_SIZE
    batches = [ [] for i in range(num_of_full_batches + 1) ]
    for i in range(num_of_full_batches + 1):
        if pop_left >= BATCH_SIZE:
            batches[i] = population[ i * BATCH_SIZE : (i+1) * BATCH_SIZE ]
        else:
            batches[i] = population[ i * BATCH_SIZE : i * BATCH_SIZE + pop_left ]
        pop_left -= BATCH_SIZE
    return batches

def train_ae(autoencoder, population):
    autoencoder.reset_optimizer_op
    loss_plot = []
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as session:
        init_all_vars_op = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
        session.run(init_all_vars_op) 
        batch_list = make_batches(population)
        print("Beginning Training of Autoencoder")
        for epoch in range(NUM_EPOCH):
            if epoch % 10 == 0:
                print("At epoch " + str(epoch))
            for batch in batch_list:
                for member in batch:
                    image = member.get_traj()
                    _, loss, _ = session.run((autoencoder.decoded, autoencoder.loss, autoencoder.learning_rate), feed_dict={autoencoder.x : image, autoencoder.keep_prob : 0, autoencoder.global_step : epoch})
                    autoencoder.step()
                    avg_loss = np.mean(loss)
                    loss_plot.append(avg_loss)
    print("Training Complete")
    plt.plot(loss_plot)
    plt.show()
    return autoencoder
    

def AURORA_ballistic_task():
    # Randomly generate some controllers
    init_size = POPULATION_INITIAL_SIZE            # Initial size of population
    pop = []                                        # Container for population
    print("Creating population container")
    for b in range(init_size):
        new_indiv = individual.indiv()
        pop.append(new_indiv)
    print("Complete")

    # Collect sensory data of the generated controllers. In the case of the ballistic task this is the trajectories but any sensory data can be considered
    # The collected sensory data makes up the first dataset
    print("Evaluating current population")
    for member in pop:
        genotype = [random.uniform(0, 1), random.uniform(0, 1)]
        member.eval(genotype)
    print("Complete")

    # Create the dimension reduction algorithm (the Autoencoder)
    my_ae = AE()

    # # Initialise
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as sess:
    #     init_all_vars_op = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
    #     sess.run(init_all_vars_op)   

    # Train the dimension reduction algorithm (the Autoencoder) on the dataset
    my_ae = train_ae(my_ae, pop)
    exit()

    # Use the now trained Autoencoder to get the behavioural descriptors
    with tf.Session() as sess:
        for member in pop:
            traj = member.get_traj()
            # Sensory data is then projected into the latent space, this is used as the behavioural descriptor
            member_bd = sess.run(my_ae.latent, feed_dict={"input_x" : traj, "keep_prob" : 0, "step_id" : 1})
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
    parser.add_argument('--control', type=str, default='ballistic', help = "Which simulation you want to run")
    args = parser.parse_args()
    if args.control == "ballistic":
        AURORA_ballistic_task()





# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, pop_data):
#         self.data = torch.zeros(len(pop_data), 100)
#         # Create image dataset from trajectory data
#         print("Constructing Dataset")
#         index = 0
#         for member in pop_data:
#             tmp = torch.zeros(100, 1)
#             traj = member.get_traj()
#             for t in range(50):
#                 tmp[2*t] = traj[t][0]
#                 tmp[2*t+1] = traj[t][1]
#             self.data[index] = tmp.t()
#             index += 1
#         print("Complete")
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         return self.data[index]

# def train_ae(encoder, decoder, population):
#     # Create dataset for loading
#     train_data = MyDataset(population)
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
#     optimizer = torch.optim.Adagrad(chain(encoder.parameters(), decoder.parameters()), lr=0.1, lr_decay=0.9)
#     print("Beginning Training of Autoencoder")
#     for epoch in range(NUM_EPOCH):
#         if epoch % 1000 == 0:
#             print("At epoch " + str(epoch))
#         train_loader = tqdm(train_loader)
#         for data in (train_loader):
#             optimizer.zero_grad()
#             z = encoder(data)
#             outputs = decoder(z)
#             diffs = (data - outputs)**2
#             loss = sum(diffs)/len(diffs)

#             loss.backward()
#             optimizer.step()
#     print("Training Complete")
#     return encoder, decoder