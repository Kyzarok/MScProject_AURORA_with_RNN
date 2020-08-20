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
# NUM_EPOCH = 250
BATCH_SIZE = 20000

RETRAIN_ITER = [50, 150, 350, 750, 1550, 3150]
NB_QD_ITERATIONS = 200
NB_QD_BATCHES = 5000

MUTATION_RATE = 0.1
ETA = 10
EPSILON = 0.1

K_NEAREST_NEIGHBOURS = 15
INITIAL_NOVLETY = 0.01

FIT_MIN = 0.001
FIT_MAX = 0.999

def get_scaling_vars(population):
    dims = population[0].get_traj().shape
    max_vals = [ 0.0 for i in range(dims[0]) ] 
    min_vals = [ 99999.9 for i in range(dims[0]) ]
    for member in population:
        this_traj = member.get_traj()
        for row in range(len(this_traj)):
            get_max = np.amax(this_traj[row])
            get_min = np.amin(this_traj[row])
            if get_max > max_vals[row]:
                max_vals[row] = get_max
            if get_min < min_vals[row]:
                min_vals[row] = get_min

    max_vals = np.array(max_vals)
    min_vals = np.array(min_vals)
    return max_vals, min_vals


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

def split_dataset(pop_size):
    val_size = int(pop_size/4)
    train_size = pop_size - val_size
    indices = [ i for i in range(pop_size)]

    t_v_list = []
    for j in range(5):
        this_list = indices.copy()
        random.shuffle(this_list)
        training_indices = this_list[0 : train_size]
        val_indices = this_list[train_size : train_size + val_size]
        t_v_list.append([training_indices, val_indices])
    return t_v_list

def train_ae(autoencoder, population, trained_this_many):
    _max, _min = get_scaling_vars(population)

    t_loss_record = []
    v_loss_record = []
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as session:
        init_all_vars_op = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
        session.run(init_all_vars_op)   

        # Reset the optimizer
        autoencoder.reset_optimizer_op
        # if(trained_this_many != 1):
        #     autoencoder.saver.restore(session, "MY_MODEL")

        # Make batches
        # batch_list = make_batches(population)
        
        # Get training and validation datasets
        ref_dataset = split_dataset(len(population))

        print("Beginning Training of Autoencoder")
        # loss = 0

        for training_data, validation_data in ref_dataset:
            condition = True
            epoch = 0
            last_val_loss = 999999999999

            # This condition controls the training session
            while(condition):
                
                # Reset epoch losses
                t_loss = 0.0
                v_loss = 0.0

                if epoch % 250 == 0:
                    print("At training epoch " + str(epoch) + ", we're " + str(epoch/NUM_EPOCH * 100) + "% of the way there!")

                # Actual training
                for t in training_data:
                    member = population[t]
                    image = member.get_scaled_image(_max, _min)
                    _, _, loss, _, _ = session.run((autoencoder.latent, autoencoder.decoded, autoencoder.loss, autoencoder.learning_rate, autoencoder.train_step), feed_dict={autoencoder.x : image, autoencoder.keep_prob : 0, autoencoder.global_step : epoch})
                    # print("New boi  " + str(member.get_key()) + " got the bd " + str(bd))
                    # autoencoder.train_step

                    t_loss += np.mean(loss)

                t_loss_record.append(t_loss)

                # Calcualte epoch validation loss
                for v in validation_data:
                    member = population[v]
                    image = member.get_scaled_image(_max, _min)
                    loss = session.run((autoencoder.loss), feed_dict={autoencoder.x : image, autoencoder.keep_prob : 0, autoencoder.global_step : 25000})
                    v_loss += np.mean(loss)

                v_loss_record.append(v_loss)

                if len(v_loss_record) >= 500:
                    # Get the last 500 values of validation loss
                    calc_avg_loss = v_loss_record[-500:]
                    # Calculate the average
                    calc_avg_loss = np.mean(calc_avg_loss)

                    # If validation loss has increased OR if max epoch count has been reached, stop the training session
                    if (calc_avg_loss > last_val_loss):
                        condition = False
                    
                    else:
                        # Record new value to use as previous validation loss
                        last_val_loss = calc_avg_loss

                if (epoch == NUM_EPOCH):
                        condition = False
                # Increase counter
                else:
                    epoch += 1



        # Save the current autoencoder
        autoencoder.saver.save(session, "MY_MODEL")

    print("Training Complete")
    plt.clf()
    plt.plot(t_loss_record, label="Training Loss")
    plt.plot(v_loss_record, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.legend()
    title = None
    if trained_this_many == 1:
        title = "Loss when Autoencoder Trained " + str(trained_this_many) + " time"
    else:
        title = "Loss when Autoencoder Trained " + str(trained_this_many) + " times"
    plt.title(title)
    save_name = "myplots/Loss_AE_Trained_"+str(trained_this_many)
    plt.savefig(save_name)
    return autoencoder

def KLC(population):
    nb_bins = 30

    # Get the "ground truth", aka the manually defined behavioural descriptor, and the current encoder latent space
    ground_truth_0 = []
    ground_truth_1 = []
    latent_space_0 = []
    latent_space_1 = []
    for member in population:
        gt = member.get_gt()
        ground_truth_0.append(gt[0])
        ground_truth_1.append(gt[1])
        bd = member.get_bd()[0]
        latent_space_0.append(bd[0])
        latent_space_1.append(bd[1])

    # Get normalized histograms of the data
    g_norm_0, _, _ = plt.hist(ground_truth_0, bins=nb_bins, density=True)
    g_norm_1, _, _ = plt.hist(ground_truth_1, bins=nb_bins, density=True)

    l_norm_0, _, _ = plt.hist(latent_space_0, bins=nb_bins, density=True)
    l_norm_1, _, _ = plt.hist(latent_space_1, bins=nb_bins, density=True)

    D_KL_0 = 0.0
    D_KL_1 = 0.0
    e_i = 0.0
    a_i = 0.0
    for i in range(nb_bins):
        # Case control. These histograms exist to act as pseudo probability distributions
        # In histograms, a bin may have 0 values
        # In a probability distribution, there is always a minor chance something can happend
        # So if any values are explicit 0s, we set them to a very small number to avoid log(0)

        if g_norm_0[i] == 0.0:
            e_i = 0.00000001
        else:
            e_i = g_norm_0[i]

        if l_norm_0[i] == 0.0:
            a_i = 0.00000001
        else:
            a_i = l_norm_0[i]
        D_KL_0 += e_i * math.log(e_i / a_i)


        if g_norm_1[i] == 0.0:
            e_i = 0.00000001
        else:
            e_i = g_norm_1[i]

        if l_norm_1[i] == 0.0:
            a_i = 0.00000001
        else:
            a_i = l_norm_1[i]
        D_KL_1 += e_i * math.log(e_i / a_i)

    D_KL = (D_KL_0 + D_KL_1) / 2
    return D_KL

def calculate_novelty_threshold(latent_space):

    rows = len(latent_space)
    cols = len(latent_space[0][0])
    X = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            X[i][j] = latent_space[i][0][j]

    XX = np.zeros((rows, 1))

    for i in range(rows):
        sigma = 0
        for j in range(cols):
            sigma += X[i][j]**2
        XX[i][0] = sigma

    XY = (2*X) @ X.T

    dist = XX @ np.ones((1, rows))
    dist += np.ones((rows, 1)) @ XX.T
    dist -= XY

    maxdist = np.sqrt(np.max(dist))

    #  arbitrary value to have a specific "resolution"
    K = 60000
    
    new_novelty = maxdist/np.sqrt(K)
    return new_novelty


def mut_eval(indiv_params):
    # Implements polynomial mutation
    new_indiv = individual.indiv()
    new_params = indiv_params.copy()
    for p in range(len(indiv_params)):
        mutate = random.uniform(0, 1)
        if mutate < MUTATION_RATE:
            u = random.uniform(0, 1)
            if u <= 0.5:
                # Calculate delta_L
                delta_L = (2 * u)**(1 / (1 + ETA)) - 1
                new_params[p] = indiv_params[p] + delta_L * (indiv_params[p] - FIT_MIN)
            else:
                # Calculate delta_R
                delta_R = 1 - (2 * ( (1-u))**(1/ (1 + ETA) ) )
                new_params[p] = indiv_params[p] + delta_R * (FIT_MAX - indiv_params[p])

    new_indiv.eval(new_params)
    return new_indiv

def does_dominate(curr_threshold, k_n_n, dist_from_k_n_n, population, init_novelty):
    dominated_indiv = -1
    x_1_novelty = init_novelty
    # If novelty threshold is greater than the nearest neighbour but less than the second nearest neighbour
    if (curr_threshold > dist_from_k_n_n[0]) and (curr_threshold < dist_from_k_n_n[1]):

        pop_without_x_2 = population.copy()
        del pop_without_x_2[k_n_n[0]]
        # print(dist_from_k_n_n)
        x_1_novelty = dist_from_k_n_n[1]
        x_2_novelty, _ = calculate_novelty(population[k_n_n[0]].get_bd(), pop_without_x_2, curr_threshold, False)

        # Find if exclusive epsilon dominance is met according to Cully QD Framework
        # First Condition
        if x_1_novelty >= (1 - EPSILON) * x_2_novelty:
            # The Second Condition measures Quality, i.e. a fitness function
            # In AURORA this does not exist as having such a defeats the purpose of autonomous discovery
            # I have included because why the hell not
            # if Q(new_indiv) >= (1 - ETA) * Q(nearest_neighbour)

            # The Third Condition is another bound that measures the combination of Novelty and Quality
            #             Quality
            #               |      .
            #               |      .    Idea is that this area 
            #               |      .    dominates
            #               |      ........
            #               |
            #               |
            #                -------------- Novelty

            dominated_indiv = k_n_n[0]

    return dominated_indiv, x_1_novelty

    
def calculate_novelty(this_bd, population, curr_threshold, check_dominate):
    # If the population is still too small
    if len(population) < K_NEAREST_NEIGHBOURS:
        novelty = 99999
        return novelty, -1
    else:
        k_n_n = []
        dist_from_k_n_n = []
        for member in range(len(population)):
            coord = population[member].get_bd()
            dist = np.linalg.norm(this_bd - coord)
            if len(dist_from_k_n_n) < K_NEAREST_NEIGHBOURS:
                dist_from_k_n_n.append(dist)
                k_n_n.append(member)
            else:
                # Sort lists so that minimum novelty is at the start
                k_n_n = [x for _,x in sorted(zip(dist_from_k_n_n, k_n_n))]
                dist_from_k_n_n.sort()
                # Is new novelty larger?
                if dist < dist_from_k_n_n[-1]:
                    dist_from_k_n_n[-1] = dist
                    k_n_n[-1] = member

        # Sort lists so that minimum novelty is at the start
        k_n_n = [x for _,x in sorted(zip(dist_from_k_n_n, k_n_n))]
        dist_from_k_n_n.sort()
        novelty = dist_from_k_n_n[0]

        dominated_indiv = -1

        if check_dominate:
            dominated_indiv, novelty = does_dominate(curr_threshold, k_n_n, dist_from_k_n_n, population, novelty)

        return novelty, dominated_indiv

def plot_latent(population, trained_this_many):
    x = []
    y = []
    for member in population:
        this_x, this_y = member.get_bd()[0]
        x.append(this_x)
        y.append(this_y)
    plt.clf()
    plt.scatter(x, y, c='b')
    plt.xlabel("Encoded dimension 1")
    plt.ylabel("Encoded dimension 2")
    title = None
    if trained_this_many == 1:
        title = "Latent Space when Autoencoder Trained " + str(trained_this_many) + " time"
    else:
        title = "Latent Space when Autoencoder Trained " + str(trained_this_many) + " times"
    plt.title(title)
    # plt.show()
    save_name = "myplots/Latent_Space_AE_Trained_"+str(trained_this_many)
    plt.savefig(save_name)

def plot_gt(population, trained_this_many):
    x = []
    y = []
    for member in population:
        this_x, this_y = member.get_gt()
        x.append(this_x)
        y.append(this_y)
    plt.clf()
    plt.scatter(x, y, c='b')
    plt.xlabel("X position at Hax Height")
    plt.ylabel("Max Height Achieved")
    title = None
    if trained_this_many == 1:
        title = "Ground Truth when Autoencoder Trained " + str(trained_this_many) + " time"
    else:
        title = "Ground Truth when Autoencoder Trained " + str(trained_this_many) + " times"
    plt.title(title)
    # plt.show()
    save_name = "myplots/Ground_Truth_AE_Trained_"+str(trained_this_many)
    plt.savefig(save_name)

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

    # Initialize variables
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as session:
    #     init_all_vars_op = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
    #     session.run(init_all_vars_op) 
        # tf.train.write_graph(session.graph_def, '../resources', 'graph.pb', as_text=False)
        # save_path_init = my_ae.saver.save(session, "../resources/model_init.ckpt")

    # Train the dimension reduction algorithm (the Autoencoder) on the dataset
    my_ae = train_ae(my_ae, pop, 1)

    # Create container for laten space representation
    latent_space = []

    _max, _min = get_scaling_vars(pop)

    # Use the now trained Autoencoder to get the behavioural descriptors
    with tf.Session() as sess:
        my_ae.saver.restore(sess, "MY_MODEL")
        for member in pop:
            image = member.get_scaled_image(_max, _min)
            # Sensory data is then projected into the latent space, this is used as the behavioural descriptor
            member_bd = sess.run(my_ae.latent, feed_dict={my_ae.x : image, my_ae.keep_prob : 0, my_ae.global_step : 25000})
            member.set_bd(member_bd)
            latent_space.append(member_bd)
    plot_latent(pop, 1)

    # Randomly intialize the QD algorithm
    # Calculate starting novelty threshold
    threshold = INITIAL_NOVLETY

    # Main algorithm

    # Loop for controlling number of times Autoencoder is retrained

    # Begin AURORA
    network_activation = 0
    klc_log = []

    # For 5000 generations, run 200 evaluations, and retrain the network a set numebr of times
    for i in range(NB_QD_BATCHES):
        _max, _min = get_scaling_vars(pop)
        # Begin Quality Diversity iterations
        print("Beginning QD iterations")

        with tf.Session() as sess:
            my_ae.saver.restore(sess, "MY_MODEL")
            print("Current size of population " + str(len(pop)))
            for j in range(NB_QD_ITERATIONS):
                # if j%20 == 0:
                #     print("At QD iteration " + str(j) + ", we're " + str(j/NB_QD_ITERATIONS * 100) + "% of the way there!")

                # 1. Randomly select a controller from the population
                this_indiv = random.choice(pop)
                controller = this_indiv.get_key()

                # 2. Mutate and evaluate the chosen controller
                new_indiv = mut_eval(controller)

                # 3. Get the Behavioural Descriptor for the new individual
                image = new_indiv.get_scaled_image(_max, _min)
                new_bd = sess.run(my_ae.latent, feed_dict={my_ae.x : image, my_ae.keep_prob : 0, my_ae.global_step : 25000})
                # print("New boi  " + str(new_indiv.get_key()) + " got the bd " + str(new_bd))

                # 4. See if the new Behavioural Descriptor is novel enough
                novelty, dominated = calculate_novelty(new_bd, pop, threshold, True)

                # 5. If the new individual has novel behaviour, add it to the population and the BD to the latent space
                if dominated == -1:                           #    If the individual did not dominate another individual
                    if novelty >= threshold:                  #    If the individual is novel
                        new_indiv.set_bd(new_bd)
                        new_indiv.set_novelty(novelty)
                        pop.append(new_indiv)
                else:                                         #    If the individual dominated another individual
                    new_indiv.set_bd(new_bd)
                    new_indiv.set_novelty(novelty)
                    pop[dominated] = new_indiv
        print("Finished QD iterations")

        if i == RETRAIN_ITER[network_activation]:
            # 6. Retrain Autoencoder after a number of QD iterations
            print("Calling Autoencoder retrain")
            my_ae = train_ae(my_ae, pop, network_activation + 2)
            print("Completed retraining")

            # 7. Clear latent space
            latent_space = []

            # 8. Assign the members of the population the new Behavioural Descriptors
            #    and refill the latent space
            _max, _min = get_scaling_vars(pop)
            with tf.Session() as sess:
                my_ae.saver.restore(sess, "MY_MODEL")
                for member in pop:
                    image = member.get_scaled_image(_max, _min)
                    member_bd = sess.run(my_ae.latent, feed_dict={my_ae.x : image, my_ae.keep_prob : 0, my_ae.global_step : 25000})
                    member.set_bd(member_bd)
                    latent_space.append(member_bd)

            # 9. Calculate new novelty threshold to ensure population size less than 10000
            threshold = calculate_novelty_threshold(latent_space)
            print("New novelty threshold is " + str(threshold))

            # 10. Update population so that only members with novel bds are allowed
            print("Add viable members back to population")
            new_pop = []
            for member in pop:
                this_bd = member.get_bd()
                novelty, dominated = calculate_novelty(this_bd, new_pop, threshold, True)
                if dominated == -1:                           #    If the individual did not dominate another individual
                    if novelty >= threshold:                  #    If the individual is novel
                        member.set_novelty(novelty)
                        new_pop.append(member)
                else:                                         #    If the individual dominated another individual
                    member.set_novelty(novelty)
                    new_pop[dominated] = member

            pop = new_pop

            # 11. Get current Latent Space and Ground Truth plots
            plot_latent(pop, network_activation + 2)
            plot_gt(pop, network_activation + 2)

            # 12. Prepare to check next retrain period
            network_activation += 1
        
        # 13. For each batch/generation, get the Kullback Liebler Coverage value
        current_klc = KLC(pop)
        klc_log.append(current_klc)

    plt.clf()
    plt.plot(klc_log, label="Training Loss")
    plt.xlabel("Generation")
    plt.ylabel("Kullback-LieblerDivergence")
    title = "Kullback-Libler Coverage, KL Divergence (Ground Truth || Generated BD)"
    plt.title(title)
    save_name = "myplots/KLC"
    plt.savefig(save_name)
    


        

 

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