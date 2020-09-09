# This is the control program
# Running this runs the ballistic task

import numpy as np
import individual
import random
import math
from original_ae import AE
from sklearn.metrics import mutual_info_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import matplotlib.pyplot as plt

from datetime import datetime

now = datetime.now()

POPULATION_INITIAL_SIZE = 200
POPULATION_LIMIT = 10000

NUM_EPOCH = 300
BATCH_SIZE = 20000

RETRAIN_ITER = [50, 150, 350, 750, 1550, 3150]

NB_QD_ITERATIONS = 200
NB_QD_BATCHES = 5000


# Params much lower for testing purposes
# NB_QD_BATCHES = 40
# RETRAIN_ITER = [10, 20, 30]
# POPULATION_INITIAL_SIZE = 5000
# NUM_EPOCH = 1

MUTATION_RATE = 0.1
ETA = 10
EPSILON = 0.1

K_NEAREST_NEIGHBOURS = 2
INITIAL_NOVLETY = 0.01

FIT_MIN = 0.001
FIT_MAX = 0.999

CURIOSITY_OFFSET = 0.01

FRAC = 0.0075

def translate_image(image):
    # print(image.shape)

    decoded = np.zeros((1, 100))
    mult = int(2/FRAC)
    for i in range(50):
        encoded = np.argmax(image[i])
        # print(image[i])
        a_count = 0
        while encoded > mult:
            encoded -= ( mult + 1 )
            a_count += 1
        x = a_count
        y = encoded.copy()
        decoded[0][i] = (x * FRAC) - 1
        decoded[0][i + 50] = (y * FRAC) - 1

    return decoded

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

def dummy_split(pop_size):
    indices = [ i for i in range(pop_size)]
    t_v_list = []
    this_list = indices.copy()
    random.shuffle(this_list)
    training_indices = this_list
    val_indices = [0 for i in range(pop_size)]
    t_v_list.append([training_indices, val_indices])
    return t_v_list

def train_ae(autoencoder, population, when_trained, with_rnn):

    _max, _min = get_scaling_vars(population)
    is_pretrained = False
    if len(population) == 10000:
        is_pretrained = True
        print("Confirmed pretrained version")

    t_loss_record = []
    v_loss_record = []
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as session:
        init_all_vars_op = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
        session.run(init_all_vars_op)
        if when_trained != 0:
            print("Loading model for retraining")
            autoencoder.saver.restore(session, "MY_MODEL")
        else:
            print("Training Autoencoder, this is training session: 1/" + str(len(RETRAIN_ITER) + 1))

        # Reset the optimizer
        autoencoder.reset_optimizer_op
        
        # Get training and validation datasets
        ref_dataset = None
        if is_pretrained == False:
            ref_dataset = split_dataset(len(population))
        else:
            ref_dataset = dummy_split(len(population))

        print("Beginning Training of Autoencoder")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)            
        
        check_state = 50
        total_epochs = 0

        condition = True
        epoch = 0
        last_val_loss = 999999999999

        for training_data, validation_data in ref_dataset:
            if total_epochs <= 600:
                condition = True
                epoch = 0
                last_val_loss = 999999999999

            # This condition controls the training session
            while (condition == True):
                
                # Reset epoch losses
                t_loss = 0.0
                v_loss = 0.0

                if (epoch % 3 == 0):
                    print("At training epoch " + str(epoch) + ", we're " + str(int(epoch/NUM_EPOCH * 100)) + "% of the way there!")

                # Actual training
                for t in training_data:
                    member = population[t]
                    if with_rnn == True:
                        true_image = member.get_scaled_image(_max, _min)
                        rnn_image = member.get_lstm_embed_traj()
                        rnn_output = session.run((autoencoder.rnn_output_image),\
                             feed_dict={autoencoder.x : true_image, autoencoder.new_rnn_input : rnn_image, autoencoder.true_x : true_image, autoencoder.keep_prob : 0, autoencoder.global_step : 300})
                        network_input_image = translate_image(rnn_output)
                        _, _, loss, _, _ = session.run((autoencoder.latent, autoencoder.decoded, autoencoder.loss, autoencoder.learning_rate, autoencoder.train_step), \
                            feed_dict={autoencoder.x : network_input_image, autoencoder.new_rnn_input : rnn_image, autoencoder.true_x : true_image, autoencoder.keep_prob : 0, autoencoder.global_step : epoch})
                        t_loss += np.mean(loss)
                    else:
                        image = member.get_scaled_image(_max, _min)
                        _, _, loss, _, _ = session.run((autoencoder.latent, autoencoder.decoded, autoencoder.loss, autoencoder.learning_rate, autoencoder.train_step), \
                            feed_dict={autoencoder.x : image, autoencoder.keep_prob : 0, autoencoder.global_step : epoch})
                        t_loss += np.mean(loss)

                t_loss_record.append(t_loss)


                # Calcualte epoch validation loss
                if is_pretrained == False:
                    for v in validation_data:
                        member = population[v]
                        if with_rnn == True:
                            true_image = member.get_scaled_image(_max, _min)
                            rnn_image = member.get_lstm_embed_traj() 
                            rnn_output = session.run((autoencoder.rnn_output_image),\
                                feed_dict={autoencoder.x : true_image, autoencoder.new_rnn_input : rnn_image, autoencoder.true_x : true_image, autoencoder.keep_prob : 0, autoencoder.global_step : 300})
                            network_input_image = translate_image(rnn_output)
                            loss = session.run((autoencoder.loss), \
                                feed_dict={autoencoder.x : network_input_image, autoencoder.new_rnn_input : rnn_image, autoencoder.true_x : true_image, autoencoder.keep_prob : 0, autoencoder.global_step : epoch})
                            
                            v_loss += np.mean(loss)
                        else:
                            image = member.get_scaled_image(_max, _min)
                            loss = session.run((autoencoder.loss), feed_dict={autoencoder.x : image, autoencoder.keep_prob : 0, autoencoder.global_step : 300})
                            v_loss += np.mean(loss)

                    v_loss_record.append(v_loss)

                    if len(v_loss_record) >= check_state:
                        # Get the last 500 values of validation loss
                        calc_avg_loss = v_loss_record[-check_state:]
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
                # Increase epoch counter
                else:
                    epoch += 1
                    total_epochs += 1

        # Save the current autoencoder
        autoencoder.saver.save(session, "MY_MODEL")

    print("Training Complete")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    plt.clf()
    plt.plot(t_loss_record, label="Training Loss")
    plt.plot(v_loss_record, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.legend()
    title = None
    if when_trained == 0:
        title = "Loss when Autoencoder Initially Trained"
    else:
        title = "Loss when Autoencoder Retrained at Generation/Batch " + str(when_trained)
    plt.title(title)
    save_name = "myplots/Loss_AE_Trained_"+str(when_trained)
    plt.savefig(save_name)
    return autoencoder, t_loss_record, v_loss_record

def KLC(population, true_gt):
    nb_bins = 30

    # Get the "ground truth", aka the manually defined behavioural descriptor, and the current encoder latent space
    my_gt_0 = []
    my_gt_1 = []
    true_gt_0 = true_gt[0]
    true_gt_1 = true_gt[1]
    for member in population:
        gt = member.get_gt()
        my_gt_0.append(gt[0])
        my_gt_1.append(gt[1])

    # Get normalized histograms of the data
    l_norm_0, _, _ = plt.hist(my_gt_0, bins=nb_bins)#, density=True, stacked=True)
    l_norm_1, _, _ = plt.hist(my_gt_1, bins=nb_bins)#, density=True, stacked=True)

    g_norm_0, _, _ = plt.hist(true_gt_0, bins=nb_bins)#, density=True, stacked=True)
    g_norm_1, _, _ = plt.hist(true_gt_1, bins=nb_bins)#, density=True, stacked=True)


    g_norm_0 /= sum(g_norm_0)
    g_norm_1 /= sum(g_norm_1)
    l_norm_0 /= sum(l_norm_0)
    l_norm_1 /= sum(l_norm_1)

    for i in range(nb_bins):
        # Case controlis necessary as these histograms exist to act as pseudo probability distributions
        # In histograms, a bin may have 0 values
        # In a probability distribution, there is always a minor chance something can happend
        # So if any values are explicit 0s, we set them to a very small number to avoid log(0)

        if g_norm_0[i] == 0.0:
            g_norm_0[i] = 0.00000001

        if l_norm_0[i] == 0.0:
            l_norm_0[i] = 0.00000001
        
        if g_norm_1[i] == 0.0:
            g_norm_1[i]= 0.00000001

        if l_norm_1[i] == 0.0:
            l_norm_1[i] = 0.00000001

    # D_KL_0 = mutual_info_score(g_norm_0, l_norm_0)
    # D_KL_1 = mutual_info_score(g_norm_1, l_norm_1)
    # sk_D_KL = (D_KL_0 + D_KL_1) / 2

    D_KL_0 = 0.0
    D_KL_1 = 0.0

    for i in range(nb_bins):
        D_KL_0 += g_norm_0[i] * math.log(g_norm_0[i]/l_norm_0[i])
        D_KL_1 += g_norm_1[i] * math.log(g_norm_1[i]/l_norm_1[i])

    D_KL = (D_KL_0 + D_KL_1) / 2
    return D_KL

def calculate_novelty_threshold(latent_space, with_rnn):

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
    # if with_rnn == True:
    #     K = 1000000
    
    new_novelty = maxdist/np.sqrt(K)
    return new_novelty

def make_novelty_params(population):

    rows = len(population)
    cols = 2
    X = np.zeros((rows, cols))
    for i in range(rows):
        this_bd = population[i].get_bd()[0]
        X[i][0] = this_bd[0]
        X[i][1] = this_bd[1]

    x_squared = np.zeros((rows, 1))
    two_x = np.zeros((rows, 1))
    y_squared = np.zeros((rows, 1))
    two_y = np.zeros((rows, 1))

    for i in range(rows):
        x_squared[i] = X[i][0]**2
        two_x[i] = 2 * X[i][0]
        y_squared[i] = X[i][1]**2
        two_y[i] = 2 * X[i][1]

    return x_squared, two_x, y_squared, two_y


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

def is_indiv_legal(new_guy, old_guy):
    get_key = new_guy.get_key()
    get_parent_key = old_guy.get_key()
    if FIT_MIN <= get_key[0] <= FIT_MAX and FIT_MIN <= get_key[1] <= FIT_MAX:
        if get_key[0] != get_parent_key[0] and get_key[1] != get_parent_key[1]:
            return True
    else:
        return False

def grow_pop_does_dominate(curr_threshold, k_n_n, dist_from_k_n_n, population, init_novelty):
    dominated_indiv = -1
    x_1_novelty = init_novelty
    # If novelty threshold is greater than the nearest neighbour but less than the second nearest neighbour
    if (curr_threshold > dist_from_k_n_n[0]) and (curr_threshold < dist_from_k_n_n[1]):

        pop_without_x_2 = population.copy()
        del pop_without_x_2[k_n_n[0]]
        x_1_novelty = dist_from_k_n_n[1]
        x_2_novelty, _ = grow_pop_calculate_novelty(population[k_n_n[0]].get_bd(), pop_without_x_2, curr_threshold, False)

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

    
def grow_pop_calculate_novelty(this_bd, population, curr_threshold, check_dominate):
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
            dominated_indiv, novelty = grow_pop_does_dominate(curr_threshold, k_n_n, dist_from_k_n_n, population, novelty)

        return novelty, dominated_indiv

def does_dominate(curr_threshold, k_n_n, dist_from_k_n_n, x_squared, two_x, y_squared, two_y, population):
    dominated_indiv = -1
    x_1_novelty = dist_from_k_n_n[0]
    # print(dist_from_k_n_n)
    if (curr_threshold > dist_from_k_n_n[0]) and (curr_threshold < dist_from_k_n_n[1]):
        x_1_novelty = dist_from_k_n_n[1]
        x_2_novelty, _ = calculate_novelty(population[k_n_n[0]].get_bd(), curr_threshold, False, x_squared, two_x, y_squared, two_y, population)
        if x_1_novelty >= (1 - EPSILON) * x_2_novelty:
            dominated_indiv = k_n_n[0]

    return dominated_indiv, x_1_novelty

def calculate_novelty(this_bd, curr_threshold, check_dominate, x_squared, two_x, y_squared, two_y, population):
    this_bd = this_bd[0]
    two_x_new_x = two_x * this_bd[0]
    two_y_new_y = two_y * this_bd[1]
    size = len(x_squared)
    new_x_squared = np.full((size, 1), this_bd[0]**2)
    new_y_squared = np.full((size, 1), this_bd[1]**2)

    sq_distances = x_squared - two_x_new_x + new_x_squared + y_squared - two_y_new_y + new_y_squared

    min_dist = np.sqrt(np.min(sq_distances))
    nn = np.argmin(sq_distances)

    sq_distances[nn] = 99999999

    second_min_dist = np.sqrt(np.min(sq_distances))
    second_nn = np.argmin(sq_distances)

    novelty = min_dist

    dominated_indiv = -1

    if check_dominate:
        dominated_indiv, novelty = does_dominate(curr_threshold, [nn, second_nn], [min_dist, second_min_dist], x_squared, two_x, y_squared, two_y, population)
    else:
        novelty = second_min_dist

    return novelty, dominated_indiv

def plot_latent_gt(population, when_trained):
    l_x = []
    l_y = []
    g_x = []
    g_y = []

    for member in population:
        this_x, this_y = member.get_bd()[0]
        l_x.append(this_x)
        l_y.append(this_y)
        this_x, this_y = member.get_gt()
        g_x.append(this_x)
        g_y.append(this_y)

    t = np.arange(len(population))

    euclidean_from_zero_gt = []
    for i in range(len(g_x)):
        distance = g_x[i]**2 + g_y[i]**2
        euclidean_from_zero_gt.append(distance)
    

    g_x = [x for _,x in sorted(zip(euclidean_from_zero_gt, g_x))]
    g_y = [y for _,y in sorted(zip(euclidean_from_zero_gt, g_y))]
    l_x = [x for _,x in sorted(zip(euclidean_from_zero_gt, l_x))]
    l_y = [y for _,y in sorted(zip(euclidean_from_zero_gt, l_y))]

    plt.clf()
    plt.scatter(l_x, l_y, c=t, cmap="rainbow")
    plt.xlabel("Encoded dimension 1")
    plt.ylabel("Encoded dimension 2")
    title = None
    if when_trained == 0:
        title = "Latent Space when Autoencoder Initially Trained"
    elif when_trained == -1:
        title = "Latent Space at Final Generation"
    else:
        title = "Latent Space when Autoencoder Retrained at Generation/Batch " + str(when_trained)
    
    plt.title(title)
    save_name = "myplots/Latent_Space_AE_Trained_"+str(when_trained)
    plt.savefig(save_name)

    plt.clf()
    plt.scatter(g_x, g_y, c=t, cmap="rainbow")
    plt.xlabel("X position at Max Height")
    plt.ylabel("Max Height Achieved")
    title = None
    if when_trained == 0:
        title = "Ground Truth when Autoencoder Initially Trained"
    elif when_trained == -1:
        title = "Ground Truth at Final Generation"
    else:
        title = "Ground Truth when Autoencoder Retrained at Generation/Batch " + str(when_trained)
    plt.title(title)
    save_name = "myplots/Ground_Truth_AE_Trained_"+str(when_trained)
    plt.savefig(save_name)

def plot_gt(population, when_trained):
    g_x = []
    g_y = []

    for member in population:
        this_x, this_y = member.get_gt()
        g_x.append(this_x)
        g_y.append(this_y)

    t = np.arange(len(population))

    euclidean_from_zero_gt = []
    for i in range(len(g_x)):
        distance = g_x[i]**2 + g_y[i]**2
        euclidean_from_zero_gt.append(distance)

    g_x = [x for _,x in sorted(zip(euclidean_from_zero_gt, g_x))]
    g_y = [y for _,y in sorted(zip(euclidean_from_zero_gt, g_y))]

    plt.clf()
    plt.scatter(g_x, g_y, c=t, cmap="rainbow")
    plt.xlabel("X position at Max Height")
    plt.ylabel("Max Height Achieved")
    title = None
    if when_trained == 0:
        title = "Ground Truth at Initialisation"
    elif when_trained == -1:
        title = "Ground Truth at Final Generation"
    else:
        title = "Ground Truth at Generation " + str(when_trained)
    plt.title(title)
    save_name = "myplots/Ground_Truth_"+str(when_trained)
    plt.savefig(save_name)

# Function to make the curiosity proportionate roulette wheel
def make_wheel(population):
    literal_roulette_curiosities = []
    min_cur = 999999999.9
    for member in population:                  
        cur = member.get_curiosity()
        literal_roulette_curiosities.append(cur)
        if cur < min_cur:
            min_cur = cur
    if min_cur < 0:
        min_cur = -1 * min_cur

    # Shift all curiosity values by the minimum curiosity in the population
    # Curiosity offset is necessary otherwise at initialisation nothing will work AND no new member of the population will be selected
    offset_roulette_curiosities = [ cur + min_cur + CURIOSITY_OFFSET for cur in literal_roulette_curiosities ]

    # Calculate proportionate wheel by dividing wheel by sum
    roulette_sum = sum(offset_roulette_curiosities)
    roulette_wheel = [ cur / roulette_sum for cur in offset_roulette_curiosities]
    return roulette_wheel


def AURORA_incremental_ballistic_task(with_rnn):
    comparison_gt = np.load("GROUND_TRUTH.npy")
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
    print("Creating network, printing autoencoder layers")
    my_ae = AE(with_rnn)

    # Train the dimension reduction algorithm (the Autoencoder) on the dataset
    my_ae, t_error, v_error = train_ae(my_ae, pop, 0, with_rnn)

    # Create container for laten space representation
    latent_space = []

    _max, _min = get_scaling_vars(pop)

    # Use the now trained Autoencoder to get the behavioural descriptors
    with tf.Session() as sess:
        my_ae.saver.restore(sess, "MY_MODEL")
        for member in pop:
            if with_rnn == False:
                image = member.get_scaled_image(_max, _min)
                # Sensory data is then projected into the latent space, this is used as the behavioural descriptor
                member_bd = sess.run(my_ae.latent, feed_dict={my_ae.x : image, my_ae.keep_prob : 0, my_ae.global_step : 300})
                member.set_bd(member_bd)
                latent_space.append(member_bd)
            else:
                true_image = member.get_scaled_image(_max, _min)
                # Sensory data is then projected into the latent space, this is used as the behavioural descriptor
                rnn_image = member.get_lstm_embed_traj()
                rnn_output = sess.run((my_ae.rnn_output_image),\
                        feed_dict={my_ae.x : true_image, my_ae.new_rnn_input : rnn_image, my_ae.true_x : true_image, my_ae.keep_prob : 0, my_ae.global_step : 300})
                network_input_image = translate_image(rnn_output)
                member_bd = sess.run((my_ae.latent), \
                    feed_dict={my_ae.x : network_input_image, my_ae.new_rnn_input : rnn_image, my_ae.true_x : true_image, my_ae.keep_prob : 0, my_ae.global_step : 300})
                member.set_bd(member_bd)
                latent_space.append(member_bd)
    
    # Record current latent space
    plot_latent_gt(pop, 0)

    # Calculate starting novelty threshold
    threshold = INITIAL_NOVLETY
    # if with_rnn == True:
    #     threshold = calculate_novelty_threshold(latent_space, with_rnn)

    # These are needed for the main algorithm
    network_activation = 0
    klc_log = []                             # Record my ver and the sklearn ver
    just_finished_training = True
    roulette_wheel = []
    repertoire_size = []
    big_error_log = [ [] for i in range(2) ]
    big_error_log[0] = t_error
    big_error_log[1] = v_error

    rmse_log = []

    last_run = False
    
    # Main AURORA algorithm, for 5000 generations, run 200 evaluations, and retrain the network at specific generations
    for generation in range(NB_QD_BATCHES):
        _max, _min = get_scaling_vars(pop)
        if just_finished_training:
            print("Beginning QD iterations, next Autoencoder retraining at generation " + str(RETRAIN_ITER[network_activation]))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print("Reinitialising curiosity proportionate roulette wheel")
        if last_run:
            print("Beginning QD iterations, no more Autoencoder training")
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print("Reinitialising curiosity proportionate roulette wheel")
            last_run = False


        roulette_wheel = make_wheel(pop)
        x_squared, two_x, y_squared, two_y = make_novelty_params(pop)

        # Begin Quality Diversity iterations
        with tf.Session() as sess:
            my_ae.saver.restore(sess, "MY_MODEL")
            print("Generation " + str(generation) + ", current size of population is " + str(len(pop)))
            gen_rmse_log = []
            for j in range(NB_QD_ITERATIONS):

                # 1. Select controller from population using curiosity proportionate selection ( AKA Spin wheel! )
                selector = random.uniform(0, 1)
                index = 0
                cumulative = roulette_wheel[index]
                while (selector >= cumulative ) and (index != len(roulette_wheel)-1):
                    index += 1
                    cumulative += roulette_wheel[index]
                this_indiv = pop[index]
                controller = this_indiv.get_key()

                # 2. Mutate and evaluate the chosen controller
                new_indiv = mut_eval(controller)
                if is_indiv_legal(new_indiv, this_indiv) == True:
                    new_bd = None
                    out = None
                    image = None

                    # 3. Get the Behavioural Descriptor for the new individual
                    if with_rnn == False:
                        image = new_indiv.get_scaled_image(_max, _min)
                        new_bd, out = sess.run((my_ae.latent, my_ae.decoded), feed_dict={my_ae.x : image, my_ae.keep_prob : 0, my_ae.global_step : 300})
                    else:
                        true_image = new_indiv.get_scaled_image(_max, _min)
                        rnn_image = new_indiv.get_lstm_embed_traj()
                        rnn_output = sess.run((my_ae.rnn_output_image),\
                                feed_dict={my_ae.x : true_image, my_ae.new_rnn_input : rnn_image, my_ae.true_x : true_image, my_ae.keep_prob : 0, my_ae.global_step : 300})
                        network_input_image = translate_image(rnn_output)
                        new_bd, out = sess.run((my_ae.latent, my_ae.decoded), \
                            feed_dict={my_ae.x : network_input_image, my_ae.new_rnn_input : rnn_image, my_ae.true_x : true_image, my_ae.keep_prob : 0, my_ae.global_step : 300})
                        image = true_image

                    gen_rmse_log.append(np.sqrt(np.mean((image - out)**2)))

                    # 4. See if the new Behavioural Descriptor is novel enough
                    # novelty, dominated = calculate_novelty(new_bd, pop, threshold, True)

                    novelty, dominated = calculate_novelty(new_bd, threshold, True, x_squared, two_x, y_squared, two_y, pop)
                    # 5. If the new individual has novel behaviour, add it to the population and the BD to the latent space
                    if dominated == -1:                           #    If the individual did not dominate another individual
                        # print("no_domination")
                        if novelty >= threshold:                  #    If the individual is novel
                            new_indiv.set_bd(new_bd)
                            new_indiv.set_novelty(novelty)
                            pop.append(new_indiv)

                            # Increase curiosity score of individual
                            pop[index].increase_curiosity()

                        else:                                    #    If the individual is NOT novel
                            # Decrease curiosity score of individual
                            pop[index].decrease_curiosity()
                    else:                                         #    If the individual dominated another individual
                        new_indiv.set_bd(new_bd)
                        new_indiv.set_novelty(novelty)
                        pop[dominated] = new_indiv

                        # Increase curiosity score of individual
                        pop[index].increase_curiosity()


        just_finished_training = False

        # Check if this generation is before the last retraining session
        if network_activation < len(RETRAIN_ITER):
            if generation == RETRAIN_ITER[network_activation]:
                # plot_latent_gt(pop, generation-1)

                print("Finished QD iterations")
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("Current Time =", current_time)
                print("Current size of population is " + str(len(pop)))

                # 6. Retrain Autoencoder after a number of QD iterations
                print("Training Autoencoder, this is training session: " + str(network_activation + 2) + "/" + str(len(RETRAIN_ITER) + 1))
                my_ae, t_error, v_error = train_ae(my_ae, pop, generation, with_rnn)
                print("Completed retraining")

                tmp = t_error.copy()
                big_error_log[0] = big_error_log[0] + tmp
                tmp = v_error.copy()
                big_error_log[1] = big_error_log[1] + tmp

                # 7. Clear latent space
                latent_space = []

                # 8. Assign the members of the population the new Behavioural Descriptors
                #    and refill the latent space
                _max, _min = get_scaling_vars(pop)
                with tf.Session() as sess:
                    my_ae.saver.restore(sess, "MY_MODEL")
                    for member in pop:
                        this_bd = None
                        if with_rnn == False:
                            image = member.get_scaled_image(_max, _min)
                            member_bd = sess.run(my_ae.latent, feed_dict={my_ae.x : image, my_ae.keep_prob : 0, my_ae.global_step : 300})
                            this_bd = member_bd.copy()
                        else:
                            true_image = member.get_scaled_image(_max, _min)
                            rnn_image = member.get_lstm_embed_traj()
                            rnn_output = sess.run((my_ae.rnn_output_image),\
                                    feed_dict={my_ae.x : true_image, my_ae.new_rnn_input : rnn_image, my_ae.true_x : true_image, my_ae.keep_prob : 0, my_ae.global_step : 300})
                            network_input_image = translate_image(rnn_output)
                            member_bd = sess.run((my_ae.latent), \
                                feed_dict={my_ae.x : network_input_image, my_ae.new_rnn_input : rnn_image, my_ae.true_x : true_image, my_ae.keep_prob : 0, my_ae.global_step : 300})
                            this_bd = member_bd.copy()

                        member.set_bd(this_bd)
                        latent_space.append(this_bd)

                # 9. Calculate new novelty threshold to ensure population size less than 10000
                threshold = calculate_novelty_threshold(latent_space, with_rnn)
                print("New novelty threshold is " + str(threshold))

                # 10. Update population so that only members with novel bds are allowed
                print("Add viable members back to population")
                new_pop = []
                for member in pop:
                    this_bd = member.get_bd()
                    novelty, dominated = grow_pop_calculate_novelty(this_bd, new_pop, threshold, True)
                    if dominated == -1:                           #    If the individual did not dominate another individual
                        if novelty >= threshold:                  #    If the individual is novel
                            member.set_novelty(novelty)
                            new_pop.append(member)
                    else:                                         #    If the individual dominated another individual
                        member.set_novelty(novelty)
                        new_pop[dominated] = member

                pop = new_pop
                print("After re-training, size of population is " + str(len(pop)))

                # 11. Get current Latent Space and Ground Truth plots
                plot_latent_gt(pop, generation)

                # 12. Prepare to check next retrain period
                network_activation += 1
                if network_activation == len(RETRAIN_ITER):
                    last_run = True
                else:
                    just_finished_training = True
            
        # 13. For each batch/generation, record various metrics
        current_klc = KLC(pop, comparison_gt)
        klc_log.append(current_klc)
        repertoire_size.append(len(pop))
        rmse_log.append(np.mean(gen_rmse_log))

    plt.clf()
    plt.plot(klc_log, label="KLC value per generation")
    plt.xlabel("Generation")
    plt.ylabel("Kullback-Leibler Divergence")
    title = "Kullback-Leibler Coverage, KL Divergence (Ground Truth || Generated BD)"
    plt.title(title)
    save_name = "myplots/KLC"
    plt.savefig(save_name)
    np.save("mydata/inc_KLC.npy", klc_log)

    plt.clf()
    plt.plot(repertoire_size, label="Repertoire Size")
    plt.xlabel("Generation")
    plt.ylabel("Number of controllers")
    title = "Repertoire Size"
    plt.title(title)
    save_name = "myplots/RepSize"
    plt.savefig(save_name)
    np.save("mydata/inc_rep_size.npy", repertoire_size)
    
    plot_latent_gt(pop, -1)

    plt.clf()
    plt.plot(big_error_log[0], label="Training Loss")
    plt.plot(big_error_log[1], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.legend()
    title = "Reconstruction Loss"
    plt.title(title)
    save_name = "myplots/Full_loss_plot"
    plt.savefig(save_name)
    np.save("mydata/inc_error_log.npy", big_error_log)

    plt.clf()
    plt.plot(rmse_log)
    plt.xlabel("Generation")
    plt.ylabel("RMSE Loss")
    title = "Average Root Mean Squared Error per Generation"
    plt.title(title)
    save_name = "myplots/inc_rmse_plot"
    plt.savefig(save_name)
    np.save("mydata/inc_rmse.npy", rmse_log)
    

# The only difference between the other basic ballistic task is that this is only trained once and with more samples    
def AURORA_pretrained_ballistic_task(with_rnn):
    comparison_gt = np.load("GROUND_TRUTH.npy")
    init_size = 100 * 100           # Initial size of population
    training_pop = []                                  # Container for training population

    # Collect sensory data of the generated controllers. In the case of the ballistic task this is the trajectories but any sensory data can be considered
    # The collected sensory data makes up the first dataset
    # print("Creating and Evaluating training population")
    # step_size = (FIT_MAX - FIT_MIN)/100
    # genotype = [0.0, 0.0]


    # for i in range(100):
    #     for j in range(100):
    #         new_indiv = individual.indiv()
    #         new_indiv.eval(genotype)
    #         training_pop.append(new_indiv)
    #         genotype[1] += step_size
    #     genotype[0] += step_size

    # for member in training_pop:
    #     genotype = [random.uniform(0, 1), random.uniform(0, 1)]
    #     member.eval(genotype)
    # print("Complete")
    # print("Creating training population")
    # for i in range(init_size):
    #     new_indiv = individual.indiv()
    #     training_pop.append(new_indiv)

    # print("Evaluating training population")
    # step_size = (FIT_MAX - FIT_MIN)/100
    # genotype = [FIT_MIN, FIT_MIN]
    # for member in training_pop:
    #     if genotype[1] > FIT_MAX:
    #         genotype[1] = FIT_MIN
    #         genotype[0] += step_size
    #     member.eval(genotype)
    #     genotype[1] += step_size
    # print("Complete")
    dim = 100
    init_size = dim * dim           # Initial size of population
    pop = []                        # Container for population
    print("Creating population")
    for i in range(init_size):
        new_indiv = individual.indiv()
        pop.append(new_indiv)
    print("Complete")
    print("Evaluating population")
    step_size = (FIT_MAX - FIT_MIN)/dim
    genotype = [FIT_MIN, FIT_MIN]
    for member in pop:
        if genotype[1] > FIT_MAX:
            genotype[1] = FIT_MIN
            genotype[0] += step_size
        member.eval(genotype)
        genotype[1] += step_size
    print("Complete")

    # Create the dimension reduction algorithm (the Autoencoder)
    print("Creating network, printing autoencoder layers")
    my_ae = AE(with_rnn)

    # Train the dimension reduction algorithm (the Autoencoder) on the dataset
    my_ae, t_error, v_error = train_ae(my_ae, training_pop, 0, with_rnn)

    # Create container for laten space representation
    latent_space = []

    # Create actual population
    init_size = POPULATION_INITIAL_SIZE 
    pop = []
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


    _max, _min = get_scaling_vars(pop)

    # Use the now trained Autoencoder to get the behavioural descriptors
    with tf.Session() as sess:
        my_ae.saver.restore(sess, "MY_MODEL")
        for member in pop:
            if with_rnn == False:
                image = member.get_scaled_image(_max, _min)
                # Sensory data is then projected into the latent space, this is used as the behavioural descriptor
                member_bd = sess.run(my_ae.latent, feed_dict={my_ae.x : image, my_ae.keep_prob : 0, my_ae.global_step : 300})
                member.set_bd(member_bd)
                latent_space.append(member_bd)
            else:
                true_image = member.get_scaled_image(_max, _min)
                rnn_image = member.get_lstm_embed_traj()
                # Sensory data is then projected into the latent space, this is used as the behavioural descriptor
                member_bd = sess.run(my_ae.latent, feed_dict={my_ae.x : rnn_image, my_ae.true_x : true_image, my_ae.keep_prob : 0, my_ae.global_step : 300})
                member.set_bd(member_bd)
                latent_space.append(member_bd)
    
    # Record current latent space
    plot_latent_gt(pop, 0)

    # Calculate starting novelty threshold
    threshold = INITIAL_NOVLETY

    # These are needed for the main algorithm
    klc_log = []                          # Record my ver and the sklearn ver
    roulette_wheel = []
    repertoire_size = []
    big_error_log = [ [] for i in range(2) ]
    big_error_log[0] = t_error
    big_error_log[1] = v_error
    rmse_log = []
    
    # Main AURORA algorithm, for 5000 generations, run 200 evaluations, and retrain the network at specific generations
    for generation in range(NB_QD_BATCHES):
        _max, _min = get_scaling_vars(pop)

        roulette_wheel = make_wheel(pop)
        x_squared, two_x, y_squared, two_y = make_novelty_params(pop)

        if generation % 1000 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            plot_latent_gt(pop, generation)

        # Begin Quality Diversity iterations
        with tf.Session() as sess:
            gen_rmse_log = []
            my_ae.saver.restore(sess, "MY_MODEL")
            print("Generation " + str(generation) + ", current size of population is " + str(len(pop)))

            for j in range(NB_QD_ITERATIONS):

                # 1. Select controller from population using curiosity proportionate selection
                selector = random.uniform(0, 1)
                index = 0
                cumulative = roulette_wheel[index]
                while (selector >= cumulative ) and (index != len(roulette_wheel)-1):
                    index += 1
                    cumulative += roulette_wheel[index]
                this_indiv = pop[index]

                controller = this_indiv.get_key()

                # 2. Mutate and evaluate the chosen controller
                new_indiv = mut_eval(controller)

                if is_indiv_legal(new_indiv, this_indiv) == True:
                    new_bd = None
                    out = None
                    image = None

                    # 3. Get the Behavioural Descriptor for the new individual
                    if with_rnn == False:
                        image = new_indiv.get_scaled_image(_max, _min)
                        new_bd, out = sess.run((my_ae.latent, my_ae.decoded), feed_dict={my_ae.x : image, my_ae.keep_prob : 0, my_ae.global_step : 300})
                    else:
                        true_image = new_indiv.get_scaled_image(_max, _min)
                        rnn_image = new_indiv.get_lstm_embed_traj()
                        rnn_output = sess.run((my_ae.rnn_output_image),\
                                feed_dict={my_ae.x : true_image, my_ae.new_rnn_input : rnn_image, my_ae.true_x : true_image, my_ae.keep_prob : 0, my_ae.global_step : 300})
                        network_input_image = translate_image(rnn_output)
                        new_bd, out = sess.run((my_ae.latent, my_ae.decoded), \
                            feed_dict={my_ae.x : network_input_image, my_ae.new_rnn_input : rnn_image, my_ae.true_x : true_image, my_ae.keep_prob : 0, my_ae.global_step : 300})
                        image = true_image

                    gen_rmse_log.append(np.sqrt(np.mean((image - out)**2)))

                    # 4. See if the new Behavioural Descriptor is novel enough

                    novelty, dominated = calculate_novelty(new_bd, threshold, True, x_squared, two_x, y_squared, two_y, pop)
                    # 5. If the new individual has novel behaviour, add it to the population and the BD to the latent space
                    if dominated == -1:                           #    If the individual did not dominate another individual
                        if novelty >= threshold:                  #    If the individual is novel
                            new_indiv.set_bd(new_bd)
                            new_indiv.set_novelty(novelty)
                            pop.append(new_indiv)

                            # Increase curiosity score of individual
                            pop[index].increase_curiosity()

                        else:                                    #    If the individual is NOT novel
                            # Decrease curiosity score of individual
                            pop[index].decrease_curiosity()
                    else:                                         #    If the individual dominated another individual
                        new_indiv.set_bd(new_bd)
                        new_indiv.set_novelty(novelty)
                        pop[dominated] = new_indiv

                        # Increase curiosity score of individual
                        pop[index].increase_curiosity()

        # 6. For each batch/generation, record various metrics
        current_klc = KLC(pop, comparison_gt)
        klc_log.append(current_klc)
        repertoire_size.append(len(pop))
        rmse_log.append(np.mean(gen_rmse_log))


    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


    plt.clf()
    plt.plot(klc_log, label="KLC value per generation")
    plt.xlabel("Generation")
    plt.ylabel("Kullback-Leibler Divergence")
    title = "Kullback-Leibler Coverage, KL Divergence (Ground Truth || Generated BD)"
    plt.title(title)
    save_name = "myplots/KLC"
    plt.savefig(save_name)
    np.save("mydata/pre_KLC.npy", klc_log)

    plt.clf()
    plt.plot(repertoire_size, label="Repertoire Size")
    plt.xlabel("Generation")
    plt.ylabel("Number of controllers")
    title = "Repertoire Size"
    plt.title(title)
    save_name = "myplots/RepSize"
    plt.savefig(save_name)
    np.save("mydata/pre_KLC.npy", repertoire_size)
    
    plot_latent_gt(pop, -1)

    plt.clf()
    plt.plot(big_error_log[0], label="Training Loss")
    plt.plot(big_error_log[1], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.legend()
    title = "Reconstruction Loss"
    plt.title(title)
    save_name = "myplots/Full_loss_plot"
    plt.savefig(save_name)
    np.save("mydata/pre_error_log.npy", big_error_log)  

    
    plt.clf()
    plt.plot(rmse_log)
    plt.xlabel("Generation")
    plt.ylabel("RMSE Loss")
    title = "Average Root Mean Squared Error per Generation"
    plt.title(title)
    save_name = "myplots/pre_rmse_plot"
    plt.savefig(save_name)
    np.save("mydata/pre_rmse.npy", rmse_log) 


def Handcoded_Genotype(hand_ver):
    # Get starting novelty threshold
    # x20 : 0.14, x25 : 0.3 or somethinig , x22 : 0.17-0.15, x18 : 0.3
    comparison_gt = np.load("GROUND_TRUTH.npy")

    # Create actual population
    init_size = POPULATION_INITIAL_SIZE * 4
    pop = []
    new_bd = np.zeros((1, 2))
    # latent_space = [ np.zeros((1,2)) for i in range(init_size)]
    print("Creating population container")
    for b in range(init_size):
        new_indiv = individual.indiv()
        pop.append(new_indiv)
    print("Complete")
    print("Evaluating population container")
    for m in range(len(pop)):
        genotype = [random.uniform(0, 1), random.uniform(0, 1)]
        pop[m].eval(genotype)
        if hand_ver == True:
            new_bd[0] = pop[m].get_gt()
        else:
            new_bd[0] = pop[m].get_key()
        pop[m].set_bd(new_bd)
        # if hand_ver == True:
        #     latent_space[m][0] = pop[m].get_gt()
        # else:
        #     latent_space[m][0] = pop[m].get_key()
    print("Complete")


    # threshold = calculate_novelty_threshold(latent_space, False)
    # print(threshold)
    # threshold = INITIAL_NOVLETY * 80

    # RETRAIN_ITER = [150, 350, 750, 1550, 3150]
    # # nov_mods = [70, 50, 25, 10, 0.5]

    # # RETRAIN_ITER = [150, 350, 750, 1550, 3150]
    # nov_mods = [70, 50, 40, 30, 20]

    # RETRAIN_ITER = [350, 750, 1550, 3150]
    # nov_mods = [60, 40, 30, 10]
    # RETRAIN_ITER = [350, 750, 2050]
    # nov_mods = [60, 40, 10]

    # threshold = INITIAL_NOVLETY * 80
    # # RETRAIN_ITER = [350, 1050, 2550, 3150]
    # RETRAIN_ITER = [350, 750, 1550, 3150]
    # nov_mods = [60, 40, 20, 10]

    # RETRAIN_ITER = [350, 750, 1550, 3150]
    # nov_mods = [60, 40, 10, 5]

    # threshold = INITIAL_NOVLETY * 70
    # RETRAIN_ITER = [300, 1000, 2550]
    # nov_mods = [45, 25, 15]

    # threshold = INITIAL_NOVLETY * 100
    # RETRAIN_ITER = [750, 1550, 3150]
    # nov_mods = [50, 30, 20]

    # threshold = INITIAL_NOVLETY * 50
    # nov_mods = [40, 30, 25, 20, 5]
    # These are needed for the main algorithm
    klc_log = []                           # Record my ver and the sklearn ver
    roulette_wheel = []
    repertoire_size = []

    threshold = INITIAL_NOVLETY * 55
    
    # nov_index = 0
    # Main AURORA algorithm, for 5000 generations, run 200 evaluations, and retrain the network at specific generations
    for generation in range(NB_QD_BATCHES):
        # if generation == 500:
        #     threshold = INITIAL_NOVLETY * 200
        
        # if generation == 1000:
        #     threshold = INITIAL_NOVLETY *

        # if nov_index < len(RETRAIN_ITER):
        #     if generation == RETRAIN_ITER[nov_index]:
        #         threshold = INITIAL_NOVLETY * nov_mods[nov_index]
        #         # latent_space = [ np.zeros((1,2)) for i in range(len(pop)) ]
        #         # for m in range(len(pop)):
        #         #     if hand_ver == True:
        #         #         latent_space[m][0] = pop[m].get_gt()
        #         #     else:
        #         #         latent_space[m][0] = pop[m].get_key()
        #         # threshold = calculate_novelty_threshold(latent_space, False)
        #         print("New threshold is " + str(threshold))
        #         nov_index+=1

        roulette_wheel = make_wheel(pop)
        x_squared, two_x, y_squared, two_y = make_novelty_params(pop)

        if generation % 1000 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
        if generation % 500 == 0:
            plot_gt(pop, generation)
        
        if generation != 0:
            print(klc_log[-1])

        # Begin Quality Diversity iterations
        print("Generation " + str(generation) + ", current size of population is " + str(len(pop)))

        for j in range(NB_QD_ITERATIONS*3):
            index = 0
            # 1. Select controller from population using curiosity proportionate selection
            selector = random.uniform(0, 1)
            cumulative = roulette_wheel[index]
            while (selector >= cumulative ) and (index != len(roulette_wheel)-1):
                index += 1
                cumulative += roulette_wheel[index]
            this_indiv = pop[index]

            controller = this_indiv.get_key()

            # 2. Mutate and evaluate the chosen controller
            new_indiv = mut_eval(controller)

            if is_indiv_legal(new_indiv, this_indiv) == True:
                # 3. Get new Behavioural Descriptor
                if hand_ver == True:
                    new_bd[0] = new_indiv.get_gt()
                else:
                    new_bd[0] = new_indiv.get_key()

                # 4. See if the new Behavioural Descriptor is novel enough
                novelty, dominated = calculate_novelty(new_bd, threshold, True, x_squared, two_x, y_squared, two_y, pop)
                # 5. If the new individual has novel behaviour, add it to the population and the BD to the latent space
                if dominated == -1:                           #    If the individual did not dominate another individual
                    if novelty >= threshold:                  #    If the individual is novel
                        new_indiv.set_bd(new_bd)
                        new_indiv.set_novelty(novelty)
                        pop.append(new_indiv)

                        # Increase curiosity score of individual
                        pop[index].increase_curiosity()

                    else:                                    #    If the individual is NOT novel
                        # Decrease curiosity score of individual
                        pop[index].decrease_curiosity()
                else:                                         #    If the individual dominated another individual
                    new_indiv.set_bd(new_bd)
                    new_indiv.set_novelty(novelty)
                    pop[dominated] = new_indiv

                    # Increase curiosity score of individual
                    pop[index].increase_curiosity()

        # 6. For each batch/generation, record various metrics
        current_klc = KLC(pop, comparison_gt)
        # print(current_klc)
        klc_log.append(current_klc)
        repertoire_size.append(len(pop))


    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


    plt.clf()
    plt.plot(klc_log, label="KLC value per generation")
    plt.xlabel("Generation")
    plt.ylabel("Kullback-Leibler Divergence")
    title = "Kullback-Leibler Coverage, KL Divergence (Ground Truth || Generated BD)"
    plt.title(title)
    save_name = "myplots/KLC"
    plt.savefig(save_name)
    if hand_ver == True:
        np.save("mydata/hand_KLC.npy", klc_log)
    else:
        np.save("mydata/geno_KLC.npy", klc_log)

    plt.clf()
    plt.plot(repertoire_size, label="Repertoire Size")
    plt.xlabel("Generation")
    plt.ylabel("Number of controllers")
    title = "Repertoire Size"
    plt.title(title)
    save_name = "myplots/RepSize"
    plt.savefig(save_name)
    np.save("mydata/hand_repSize.npy", repertoire_size)
    if hand_ver == True:
        np.save("mydata/hand_repSize.npy", repertoire_size)
    else:
        np.save("mydata/geno_repSize.npy", repertoire_size)

    plot_gt(pop, -1)

def get_GROUND_TRUTH():
    dim = 125
    init_size = dim * dim           # Initial size of population
    pop = []                        # Container for population
    print("Creating population")
    for i in range(init_size):
        new_indiv = individual.indiv()
        pop.append(new_indiv)
    print("Complete")
    print("Evaluating population")
    step_size = (FIT_MAX - FIT_MIN)/dim
    genotype = [FIT_MIN, FIT_MIN]
    # count = 0
    for member in pop:
        if genotype[1] > FIT_MAX:
            # count += 1
            # print("now at " + str(count))
            genotype[1] = FIT_MIN
            genotype[0] += step_size
        member.eval(genotype)
        genotype[1] += step_size
    print("Complete")

    for member in pop:
        member.set_bd(np.array(member.get_gt()))
    
    print("Add viable members back to population")
    new_pop = []
    count = 0
    random.shuffle(pop)
    for member in pop:
        if count % 1000 == 0:
            print("At " + str(int(count/len(pop) * 100)) + "%, population is of size " + str(len(new_pop)))
        this_bd = member.get_bd()
        novelty, dominated = grow_pop_calculate_novelty(np.array(this_bd), new_pop, INITIAL_NOVLETY, True)
        if dominated == -1:                           #    If the individual did not dominate another individual
            if novelty >= INITIAL_NOVLETY:            #    If the individual is novel
                member.set_novelty(novelty)
                new_pop.append(member)
        else:                                         #    If the individual dominated another individual
            member.set_novelty(novelty)
            new_pop[dominated] = member
        count += 1

    pop = new_pop
    print("Size of distribution")
    print(len(pop))

    g_x = []
    g_y = []

    for member in pop:
        this_x, this_y = member.get_gt()
        g_x.append(this_x)
        g_y.append(this_y)
    
    np.save("GROUND_TRUTH.npy", np.array([g_x, g_y]))
    
    t = np.arange(len(pop))

    euclidean_from_zero_gt = []
    for i in range(len(g_x)):
        distance = g_x[i]**2 + g_y[i]**2
        euclidean_from_zero_gt.append(distance)


    g_x = [x for _,x in sorted(zip(euclidean_from_zero_gt, g_x))]
    g_y = [y for _,y in sorted(zip(euclidean_from_zero_gt, g_y))]

    plt.clf()
    plt.scatter(g_x, g_y, c=t, cmap="rainbow")
    plt.xlabel("X position at Max Height")
    plt.ylabel("Max Height Achieved")
    title = "The sampled ground truth distribution"
    plt.title(title)
    save_name = "myplots/Ground_Truth"
    plt.savefig(save_name)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='incremental', help = "Which version of AURORA-AE do you want to run? 'incremental' or 'pretrained'?")
    parser.add_argument('--with_RNN', type=bool, default=False, help = "Do you want to run the RNN version? 'True' or 'False'")
    args = parser.parse_args()
    if args.version == "pretrained":
        AURORA_pretrained_ballistic_task(args.with_RNN)
    elif args.version == "incremental":
        AURORA_incremental_ballistic_task(args.with_RNN)
    elif args.version == "handcoded":
        Handcoded_Genotype(True)
    elif args.version == "genotype":
        Handcoded_Genotype(False)
    elif args.version == "GT":
        get_GROUND_TRUTH()
    

