#Author: Kyle Norland
#Date: 8-17-22, modified 11/23/22
#Purpose: Implement GA for a box environment and compare with SOTA, testing epsilon existence and settings, as well as population size.


#-------------------Imports-----------------------
#General
import numpy as np
rng = np.random.default_rng(12345)


import os
import time
import random
import time
from datetime import datetime
import json
import copy

#OpenAI Gym and Environments
import gym
from gym import wrappers
import vector_grid_goal

#Matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

#from moviepy.editor import ImageSequenceClip

#deap
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

#Scoop for distributed tasking
from scoop import futures

#Local
import ga_eps_grapher



#--------------------------------------------------
#---------------------Functions--------------------
#--------------------------------------------------
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
            
def get_discrete_state(self, state, bins, obsSpaceSize):
    #https://github.com/JackFurby/CartPole-v0
    stateIndex = []
    for i in range(obsSpaceSize):
        stateIndex.append(np.digitize(state[i], bins[i]) - 1) # -1 will turn bin into index
    return tuple(stateIndex)
   
def init_environment(env_config):
    env_name = env_config['env_name']
    
    if env_name == 'FrozenLake-v1':
        env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=env_config['is_slippery'])
        #env = gym.make("FrozenLake-v1", is_slippery=False)
    
    if env_name == 'vector_grid_goal':
        grid_dims = (7,7)
        player_location = (0,0)
        goal_location = (6,6)
        custom_map = np.array([[0,1,1,1,0,0,0],
                                [0,0,1,0,0,1,0],
                                [0,0,0,0,0,0,1],
                                [0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,1],
                                [0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0]])
    
        env = vector_grid_goal.CustomEnv(grid_dims=grid_dims, player_location=player_location, goal_location=goal_location, map=custom_map)
    
    
    return env

def eval_individual(ind, eval_settings):
    env = eval_settings['env']
    episode_len = eval_settings['episode_len']

    #eval_settings['epsilon'] = max(eval_settings['epsilon_min'], np.exp(-eval_settings['epsilon_decay']*e))
 
    render=False
    total_reward = 0
   
    if eval_settings['episode_counter'] % 500 == 0:
        print("Evaluation number #", eval_settings['episode_counter'])
        print("Epsilon is: ", eval_settings['epsilon'])
    eval_settings['episode_counter'] += 1
    
    
    #Handle the epsilon switching period, if present.
    if eval_settings['epsilon_switching'] == True:
        if eval_settings['episode_counter'] % eval_settings['epsilon_switching_period'] == 0 and eval_settings['episode_counter'] != 0:
            #Toggle the switching flag.
            eval_settings['switching_flag'] = not eval_settings['switching_flag']
            #print("Switching epsilon flag to: ", eval_settings['switching_flag'])
    
    #Initialize environment
    obs = env.reset()
    
    #print("evaluating individual: ", ind)
    
    t = 0
    while t < episode_len:
    
        #Rendering
        if render:
            env.render()
            
        #action_index = ind[obs]
        #action_sequence = action_table[action_index]
        
        #Epsilon random choice with handling for epsilon switching.
        if eval_settings['epsilon_switching']:
            if eval_settings['switching_flag']:
                modified_epsilon = eval_settings['epsilon']
            else:
                modified_epsilon = 0
        else:
            #Not switching
            modified_epsilon = eval_settings['epsilon']
            
        if np.random.uniform(0,1) < modified_epsilon:
            action = random.randrange(env.action_space.n)
            #print("Taking random action: ", action)
        else:
            action = ind[obs]   #Keep action from GA
            #print("Taking preplanned action: ", action)
                   
        #Take the action
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        
        #Increment time step
        t += 1
    
        if done:
            break
    
    #Adjust epsilon
    #if eval_settings['epsilon'] > eval_settings['epsilon_min']:
    #    eval_settings['epsilon'] *= (1-eval_settings['epsilon_decay'])
    
    #eval_settings['epsilon'] = max(eval_settings['epsilon_min, np.exp(-eval_settings[epsilon_decay]*e))
    #print("Check")
    #print(eval_settings['epsilon'] - eval_settings['epsilon_min'])
    #print(np.exp(-eval_settings['epsilon_decay'] * eval_settings['episode_counter'])) 
    
    #pre_epsilon = eval_settings['epsilon']
    eval_settings['epsilon'] = eval_settings['epsilon_min'] + (eval_settings['epsilon'] - eval_settings['epsilon_min']) * np.exp(-eval_settings['epsilon_decay'] * eval_settings['episode_counter'])            
    #print("Post Epsilon", eval_settings['epsilon'])
    #print("Difference: ", (pre_epsilon - eval_settings['epsilon'])) 
    return total_reward, 

    
#-------------------Experiments---------------------------
def gen_epsilon_exist(default_run):
    #Description: Test epsilon setting 0 vs 50%
    #Try with two epsilon settings and 10 different random seeds.
    #epsilon_maxes = [0, 0.05, 0.1, 0.25, 0.5]
    epsilon_maxes = [0, 0.25, 0.5, 0.75]
    random_seeds = rng.integers(low=0, high=9999, size=4)
    
    new_experiment = {'runs':[]}
    new_experiment['generation_time']= time.time()
    new_experiment['variables'] = ['epsilon_max', 'np_seed']
    
    color_list = ['green', 'blue', 'red', 'yellow', 'orange', 'brown']
    for i, epsilon_max in enumerate(epsilon_maxes):
        for seed in random_seeds:
            new_run = copy.deepcopy(default_run)
            new_run['epsilon_max'] = epsilon_max
            new_run['np_seed'] = seed
            new_run['env_seed'] = seed
            new_run['python_seed'] = seed
            new_run['color'] = color_list[i]
            new_run['label'] = "eps_max: " + str(epsilon_max) + " seed: " + str(seed)
            
            print("Settings: ", new_run['epsilon_max'], ": ", seed)
            
            #Add run to experiment
            new_experiment['runs'].append(copy.deepcopy(new_run))
            
    print("Returning new experiment")
    return new_experiment

def mutation_vs_epsilon(default_run):
    #Description: Test epsilon setting 0 vs 50%
    #Try with two epsilon settings and 10 different random seeds.
    #epsilon_maxes = [0, 0.05, 0.1, 0.25, 0.5]
    epsilon_maxes = [0, 0.25]
    mutation_probs = [0, 0.25, 0.5]
    random_seeds = rng.integers(low=0, high=9999, size=3)
    
    new_experiment = {'runs':[]}
    new_experiment['generation_time']= time.time()
    new_experiment['variables'] = ['epsilon_max', 'mutation_prob', 'np_seed']
    
    color_list = ['green', 'blue', 'red', 'yellow', 'orange', 'brown']
    color_counter = 0
    for epsilon_max in epsilon_maxes:
        for mutation_prob in mutation_probs:
            for seed in random_seeds:
                new_run = copy.deepcopy(default_run)
                new_run['epsilon_max'] = epsilon_max
                new_run['mutation_prob'] = mutation_prob
                new_run['np_seed'] = seed
                new_run['env_seed'] = seed
                new_run['python_seed'] = seed
                new_run['color'] = color_list[color_counter]
                new_run['label'] = "eps_max: " + str(epsilon_max) + " mutation_prob: "+ str(mutation_prob) +  " seed: " + str(seed)
                
                print("Settings: ", new_run['epsilon_max'], ": ", seed)
                
                #Add run to experiment
                new_experiment['runs'].append(copy.deepcopy(new_run))
            
            color_counter += 1   
    print("Returning new experiment")
    return new_experiment        
    
def gen_pop_size(default_run):
    #Description: Test different population sizes and different epsilons
    
    #Try with two epsilon settings and two population sizes.
    #epsilon_maxes = [0, 0.05, 0.1, 0.25, 0.5]
    epsilon_maxes = [0.25]
    pop_sizes = [25, 50, 100]
    random_seeds = rng.integers(low=0, high=9999, size=3)
    
    new_experiment = {'runs':[]}
    new_experiment['generation_time']= time.time()
    new_experiment['variables'] = ['epsilon_max', 'pop_size', 'np_seed']
    
    color_list = ['green', 'blue', 'red', 'yellow', 'orange', 'brown']
    color_counter = 0
    for epsilon_max in epsilon_maxes:
        for pop_size in pop_sizes:
            for seed in random_seeds:
                new_run = copy.deepcopy(default_run)
                new_run['epsilon_max'] = epsilon_max
                new_run['pop_size'] = pop_size
                new_run['np_seed'] = seed
                new_run['env_seed'] = seed
                new_run['python_seed'] = seed
                new_run['color'] = color_list[color_counter]
                new_run['label'] = "eps_max: " + str(epsilon_max) + " pop_size: "+ str(pop_size) +  " seed: " + str(seed)
                
                print("Settings: ", new_run['epsilon_max'], ": ", seed)
                
                #Add run to experiment
                new_experiment['runs'].append(copy.deepcopy(new_run))
            
            color_counter += 1   
    print("Returning new experiment")
    return new_experiment 
    
def gen_slippery(default_run):
    #Description: Test Performance if Slippery or Not
    #Try with slippery or not standard GA and 10 different random seeds.
    #epsilon_maxes = [0, 0.05, 0.1, 0.25, 0.5]
    slippery_vals = [True, False]
    random_seeds = rng.integers(low=0, high=9999, size=5)
    
    new_experiment = {'runs':[]}
    new_experiment['generation_time']= time.time()
    new_experiment['variables'] = ['is_slippery']
    
    color_list = ['green', 'blue', 'red', 'yellow', 'orange', 'brown']
    color_counter = 0
    for slippery_val in slippery_vals:
        for seed in random_seeds:
            new_run = copy.deepcopy(default_run)
            new_run['env_config']['is_slippery'] = slippery_val
            new_run['np_seed'] = seed
            new_run['env_seed'] = seed
            new_run['python_seed'] = seed
            new_run['color'] = color_list[color_counter]
            new_run['label'] = "is_slippery: " + str(slippery_val) +  " seed: " + str(seed)
            
            print("Settings: ", new_run['epsilon_max'], ": ", seed)
            
            #Add run to experiment
            new_experiment['runs'].append(copy.deepcopy(new_run))
        
        color_counter += 1   
    print("Returning new experiment")
    return new_experiment 
    
def epsilon_switching(default_run):
    #Description: Test if switching on and off the epsilon has an effect.
    #Try with two epsilon settings and different random seeds and the switching.
    epsilon_maxes = [0.5]
    epsilon_switching_settings = [True, False]
    random_seeds = rng.integers(low=0, high=9999, size=5)
    
    new_experiment = {'runs':[]}
    new_experiment['generation_time']= time.time()
    new_experiment['variables'] = ['epsilon_max', 'epsilon_switching', 'np_seed']
    
    color_list = ['green', 'blue', 'red', 'yellow']
    color_counter = 0
    for epsilon_max in epsilon_maxes:
        for epsilon_switching in epsilon_switching_settings:
            for seed in random_seeds:       #Trials per setting
                new_run = copy.deepcopy(default_run)
                new_run['epsilon_max'] = epsilon_max
                new_run['np_seed'] = seed
                new_run['env_seed'] = seed
                new_run['python_seed'] = seed
                
                #Epsilon Switching
                new_run['epsilon_switching'] = epsilon_switching
                new_run['switching_flag'] = epsilon_switching #Flag is on if switching, off if not
                
                #Plotting details
                new_run['color'] = color_list[color_counter]
                new_run['label'] = 'eps_max: ' +  str(epsilon_max) + ' switching: ' + str(int(epsilon_switching))
            
            
                
                print("Settings: " + 'eps_max: ' +  str(epsilon_max) + ' switching: ' + str(int(epsilon_switching)))
                
                #Add run to experiment
                new_experiment['runs'].append(copy.deepcopy(new_run))
                
            #Increment Color Counter
            color_counter += 1 
    print("Returning new experiment")
    return new_experiment        

def gen_mutation_rates(default_run):
    #Description: Test performance across different mutation rates
    #Try with several mutation values and different random seeds.
    mutation_probs = [0, 0.25, 0.5, 0.75, 1]
    random_seeds = rng.integers(low=0, high=9999, size=3)
    
    new_experiment = {'runs':[]}
    new_experiment['generation_time']= time.time()
    new_experiment['variables'] = ['mutation_prob']
    
    color_list = ['green', 'blue', 'red', 'yellow', 'orange', 'brown']
    color_counter = 0
    for mutation_prob in mutation_probs:
        for seed in random_seeds:
            new_run = copy.deepcopy(default_run)
            new_run['mutation_prob'] = mutation_prob
            new_run['np_seed'] = seed
            new_run['env_seed'] = seed
            new_run['python_seed'] = seed
            new_run['color'] = color_list[color_counter]
            new_run['label'] = "Mutation Prob: " + str(mutation_prob) +  " seed: " + str(seed)
            
            print("Settings: ", new_run['mutation_prob'], ": ", seed)
            
            #Add run to experiment
            new_experiment['runs'].append(copy.deepcopy(new_run))
        
        color_counter += 1   
    print("Returning new experiment")
    return new_experiment 
#----------------------------------------------------
#-----------------MAIN-------------------------------
#----------------------------------------------------

if __name__ == "__main__":

    #------------------------------------------
    #--------------Generate Experiment---------
    #------------------------------------------
    #Default Run Settings    
    default_run = {}    
    default_run['generation_date'] = time.time()
    default_run['pop_size'] = 50
    default_run['ngen'] = 150
    default_run['episode_len'] = 100
    default_run['np_seed'] = 345
    default_run['env_seed'] = 345
    default_run['python_seed'] = 345
    default_run['epsilon_max'] = 0#0.5
    default_run['epsilon_min'] = 0
    default_run['epsilon_decrease_rule'] = 'exponential'
    default_run['epsilon_decay'] = 0.000001 #0.0000001 
    default_run['epsilon_switching'] = False
    default_run['epsilon_switching_period'] = 500
    default_run['switching_flag'] = False
    default_run['results_dict'] = {}
    default_run['selection_method'] = 'tournament'
    default_run['tournament_size'] = 3
    default_run['mutation_method'] = 'mutUniformInt'
    default_run['crossover_prob'] = 0.5
    default_run['mutation_prob'] = 0.2
    
    
    default_run['output_dict'] = {}
    
    #Env Config
    default_run['env_config'] = {}
    default_run['env_config']['env_name'] = 'FrozenLake-v1'
    default_run['env_config']['is_slippery'] = False
    #default_run['env_config']['env_name'] = 'vector_grid_goal'
    
    
    #Output Path
    default_run['output_path'] = 'macro_GA_output'
    if not os.path.exists(default_run['output_path']):
        os.makedirs(default_run['output_path'], exist_ok = True)    

    
    #Generate or Load Experiment
    folder_mode = False
    generate_mode = True
    
    experiment = {'runs': []}
    
    if generate_mode:
        #Generate experiment
        #experiment = copy.deepcopy(gen_epsilon_exist(default_run))
        experiment = copy.deepcopy(gen_pop_size(default_run))
        #experiment = copy.deepcopy(mutation_vs_epsilon(default_run))
        #experiment = copy.deepcopy(gen_slippery(default_run))
        #experiment = copy.deepcopy(gen_mutation_rates(default_run))
        #experiment = copy.deepcopy(epsilon_switching(default_run))
        #Save experiment
        experiment_name = str(experiment['generation_time']) + '.json'
        with open(os.path.join('saved_experiments', experiment_name), 'w') as f:
            json.dump(experiment, f, cls=MyEncoder)
    
    
    
    
    #----------------------------------------------------
    #---------------Run the Experiment-------------------
    #----------------------------------------------------
    for run in experiment['runs']:
        #Start Timer
        print("Run")
        print(run)
        run['run_start_time'] = time.time()

        #Set random seeds
        random.seed(run['python_seed'])
        np.random.seed(run['np_seed'])
        
        #Initialize environment
        env = init_environment(run['env_config'])
        
        #Env seed
        env.action_space.np_random.seed(run['env_seed'])


        #---------------------------------
        #-------Initialize Genome---------
        #---------------------------------
        genome_size = env.observation_space.n

        #Options for each gene
        gene_options = list(range(0, env.action_space.n))
        
        print("Genome Size: ", genome_size)
        print("Gene options: ", gene_options)


        #Deap initialization
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        #Scoop enabling
        toolbox.register("map", futures.map)
        
        #Structure Initializers
        toolbox.register("attr_int", random.randint, 0, env.action_space.n-1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, env.observation_space.n)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        #Set up eval settings
        eval_settings = {}
        eval_settings['env'] = env
        eval_settings['episode_len'] = run['episode_len']
        eval_settings['episode_counter'] = 0
        eval_settings['epsilon_switching'] = run['epsilon_switching']
        eval_settings['epsilon_switching_period'] = run['epsilon_switching_period']
        eval_settings['switching_flag'] = run['switching_flag']
        
        for key in ['epsilon_max', 'epsilon_min', 'epsilon_decrease_rule', 'epsilon_decay']:
            eval_settings[key] = run[key]
        eval_settings['epsilon'] = eval_settings['epsilon_max']
        
        #Define operations
        toolbox.register("evaluate", eval_individual, eval_settings=eval_settings)
        toolbox.register("mate", tools.cxTwoPoint)
        if run['mutation_method'] == 'mutUniformInt':
            toolbox.register("mutate", tools.mutUniformInt, low=0, up=env.action_space.n-1, indpb=0.05)
        if run['selection_method'] == 'tournament':
            toolbox.register("select", tools.selTournament, tournsize=run['tournament_size'])
        
        #Create the population and run the evolution
        pop = toolbox.population(n=run['pop_size'])
        
        
        #Register stats and other records
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        #Run the algorithm using a pre-set training algorithm.
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=run['crossover_prob'], mutpb=run['mutation_prob'], ngen=run['ngen'], 
                                       stats=stats, halloffame=hof, verbose=False)

        run['time_taken'] = time.time() - run['run_start_time']
        print("Time taken: ", run['time_taken'])
        
        #Get best agent and try it
        #print(pop)
        #print(stats)
        print("Best agent: ", hof, '\n')
        #print(log)
        #print(log[0])
        #print(stats.compile(pop))
        
        
        #-----------------------Output_Data--------------------------
        output = run['output_dict']
        output['stats'] = [x for x in log]
            
        #Save the JSON outputs

        
    
    #--------------------------------------------------------
    #--------------End of Experiment Processing--------------
    #--------------------------------------------------------
    
    #Save
    #save_name = str(round(time.time())) + "_json_output.json"
    
    out_folder_name = str(experiment['generation_time']) + '_experiment'
    os.makedirs(os.path.join('results', out_folder_name), exist_ok=True)
    
    save_name = "json_output.json"
    with open(os.path.join('results', out_folder_name, save_name ), 'w') as f:
        json.dump(experiment, f, cls=MyEncoder)    
    
    #Save a text file with the changed variables
    save_name = '+'.join(experiment['variables'])
    file_path = os.path.join('results', out_folder_name, save_name)
    with open(file_path, 'w') as f:
        pass
    
    print("Processing Visuals")
    ga_eps_grapher.Analyze_Experiment(experiment)
    
    
    '''
    #--------------------------------------------------
    #-----------------Post Processing------------------
    #--------------------------------------------------
    
    #Plot all of the results together.   
    fig = plt.figure()
    ax = plt.subplot(111)
    
    
    #Prepare data set
    for iterator, run in enumerate(output_dict['runs']):
        avg_reward = [x['avg'] for x in run['stats']]
        
        #Make label
        set_name = ""
        for entry in run['action_table']:
            set_name += str(entry) + "-"
        
        #Generate equivalent number of timesteps for x axis
        timesteps = [(x * pop_size * episode_len) for x in range(0,len(avg_reward))]
        
        ax.plot(avg_reward, label= str(iterator))
    
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.4,
                 box.width, box.height * 0.6])
        
    plt.title("Tester")
    plt.ylabel("Reward")
    plt.xlabel("# Generations")
    plt.ylim(-0.1, 1.1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, fancybox=True, shadow=True)
    save_name = str(round(time.time())) + "_avg_reward.jpg"
    plt.savefig(os.path.join(output_path, save_name))
    #plt.show()  
    '''
    
    
    
    
    
    #Run eval
    '''
    print("Running with best")
    best = pop[0]
    episode_len = 100
    total_reward = 0
    
    obs = env.reset()
    for t in range(episode_len):
        env.render()
        action = best[obs]
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print("Total reward: ", total_reward) 
    '''    
    
    '''   
    #Initialize structures
    #Check for too large of an observation space
    env_type="Discrete"
    try:
        obs_size = len(env.observation_space.high)
        if len(env.observation_space.high) > 6:
            print("Observation space too large with current binning")  
        env_type = "Box"
    except:
        pass
    
    
        
    num_bins = 10
    size = ([num_bins * self.obs_space_size + [self.act_size])
    self.Q = np.random.uniform(low=-1, high=1, size=size)
    #self.Q = np.zeros((size)) 
    '''


