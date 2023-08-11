#Author: Kyle Norland
#Date: 9-17-22
#Purpose: Secondary code to analyze agents.


#-------------------Imports-----------------------
#General
import numpy as np
import os
import time
import random
import time
from datetime import datetime
import json
import re
import matplotlib.pyplot as plt
import copy

def Analyze_Experiment(experiment):
    #Set up the figure   
    fig = plt.figure()
    ax = plt.subplot(111)

    #For output in folder, if meets conditions, graph all.
    total_run_rewards = {}
    num_runs = 1    #Special adjustment, averaging actually not necessary in this case.
    plot_names = []

    for iterator, entry in enumerate(experiment['runs']):    
        run = entry['output_dict']
        avg_reward = [x['avg'] for x in run['stats']]
        
        #Make label
        set_name = ""
        set_name += "eps_max: " + str(entry['epsilon_max']) + " seed: " + str(entry['np_seed'])
        plot_names.append(set_name)
        
        #Generate equivalent number of timesteps for x axis
        #timesteps = [(x * pop_size * episode_len) for x in range(0,len(avg_reward))]
        
        
        ax.plot(avg_reward, label=entry['label'], c=entry['color'])
        '''
        if iterator in total_run_rewards:
            temp_list = []
            temp_list = [a + b for a,b in zip(total_run_rewards[iterator], avg_reward)]  
            total_run_rewards[iterator] = temp_list[:]
            
        else:
            total_run_rewards[iterator] = avg_reward
        '''
        
    '''    
    #Plot each of the iterators
    for key, value in total_run_rewards.items():
        avg_value = [x / num_runs for x in value]
        #print("avg_value: ", avg_value)
        ax.plot(avg_value, label= plot_names[key], c=)
    '''
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.4,
                 box.width, box.height * 0.6])
        
    plt.title(experiment['title'])
    plt.ylabel("Reward")
    plt.xlabel("# Generations")
    #plt.ylim(-0.1, 1.1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fancybox=True, shadow=True)
    
    #Change size and dpi
    fig.set_size_inches(10, 9)
    fig.set_dpi = 400
    
    #Save to local file
    save_name = os.path.join('results', str(experiment['generation_time']) + '_experiment', 'avg_reward_vs_generation.png')
    plt.savefig(save_name, bbox_inches='tight')
    
    #Also save as most recent
    save_name = os.path.join('results', 'most_recent_graph.png')
    plt.savefig(save_name, bbox_inches='tight')
    
    
    plt.show()
    
    #--------------------------------------------------------------
    #---------------------------Average Plots----------------------
    #--------------------------------------------------------------
    if False:
        #Set up the figure   
        fig = plt.figure()
        ax = plt.subplot(111)

        #For output in folder, if meets conditions, graph all.
        total_run_rewards = {}
        num_runs = 1    #Special adjustment, averaging actually not necessary in this case.
        plot_names = []


        #Keep track of averages
        for iterator, entry in enumerate(experiment['runs']):    
        
            print("Printing entry")
            print(entry)
            avg_reward = [x['avg'] for x in entry['output_dict']['stats']]
            
            #Make label
            set_name = ""
            set_name += "eps_max: " + str(entry['epsilon_max']) + " seed: " + str(entry['np_seed'])
            plot_names.append(set_name)
            
            #Generate equivalent number of timesteps for x axis
            #timesteps = [(x * pop_size * episode_len) for x in range(0,len(avg_reward))]
            
            
            ax.plot(avg_reward, label=entry['label'], c=entry['color'])
            '''
            if iterator in total_run_rewards:
                temp_list = []
                temp_list = [a + b for a,b in zip(total_run_rewards[iterator], avg_reward)]  
                total_run_rewards[iterator] = temp_list[:]
                
            else:
                total_run_rewards[iterator] = avg_reward
            '''
            
        '''    
        #Plot each of the iterators
        for key, value in total_run_rewards.items():
            avg_value = [x / num_runs for x in value]
            #print("avg_value: ", avg_value)
            ax.plot(avg_value, label= plot_names[key], c=)
        '''
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.4,
                     box.width, box.height * 0.6])
            
        plt.title("Tester")
        plt.ylabel("Reward")
        plt.xlabel("# Generations")
        #plt.ylim(-0.1, 1.1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fancybox=True, shadow=True)
        save_name = os.path.join('results', str(experiment['generation_time']) + '_experiment', 'avg_reward_vs_generation.jpg')
        plt.savefig(save_name, bbox_inches='tight')
        plt.show()

    
    

    #-----------------------------------------------------------
    #----------------Diversity Maintenance Measure Plots-------------
    #-----------------------------------------------------------
    if False:
        for iterator, entry in enumerate(experiment['runs']):    
            run = entry['output_dict']
            print("Handling new run")
            saved_pops = run['saved_pops']
            
            for pop in saved_pops:
                #print(pop)
                #Left 0, down, 1, right, 2, up 3
                #Start at zero.
                #if 0, -1, 
                #if 1, + 4
                #if 2, +1
                #if 3, -4
                #if not in range 0,15, no change.
                
                
                print(pop[0])
                #print(pop[1])
                #Combined states visited 
                pop_states_visited = []
                
                for i, genome in enumerate(pop[1]):
                    #States visited
                    state_visited = [0 for x in range(0,16)]
                    #print(state_visited)
                    
                    current_state = 0
                    visited_states = []
                    #Go until state is repeated or end reached
                    while current_state not in visited_states:
                        print("current_state: ", current_state)
                        #Add to list and change state visitation
                        state_visited[current_state] = 1
                        visited_states.append(current_state)
                        
                        #Break if finished
                        if current_state == 15:
                            print("At 15")
                            break
                        
                        #Run policy
                        next_state = copy.deepcopy(current_state)
                        if genome[current_state] == 0:
                            next_state -= 1
                        if genome[current_state] == 1:
                            next_state += 4
                        if genome[current_state] == 2:
                            next_state += 1
                        if genome[current_state] == 3:
                            next_state -= 4
                            
                        if next_state in range(0,16):
                            #Add to count
                            current_state = next_state
                        else:
                            break
                            
                    #Post run list
                    print("State vistation")
                    print(state_visited)
                    print("")
                    
                    pop_states_visited.append(state_visited)
                
                    if i > 5:
                        print("Breaking")
                        
                        #Make a graphic
                        total_counts = pop_states_visited[0]
                        #Add all other ones other than first.
                        for sv in pop_states_visited[1:]:
                            print(sv)
                            total_counts = [x + y for (x,y) in zip(total_counts,sv)]
                        
                        print("Total Counts")
                        print(total_counts)
                        #Plot total counts
                        square_counts = np.asarray(total_counts).reshape(4,4)
                        print(square_counts)
                        plt.imshow(square_counts, cmap='hot', interpolation='nearest')
                        plt.show()
                            
                        
                        
                        
                        break
                        
                        
                        
                #print("Breaking 2")
                #break
            #Only do for the first one.
        

    


if __name__ == "__main__":   
    #Load in the most recent experiment and run analyze experiment.
    with open(os.path.join('results', 'most_recent.json' ), 'r') as f:
        experiment = json.load(f)
    print("Experiment Loaded: Analyzing")    
    Analyze_Experiment(experiment)

    '''
    #Set up the figure   
    fig = plt.figure()
    ax = plt.subplot(111)

    #For output in folder, if meets conditions, graph all.
    main_folder = "macro_GA_output"
    total_run_rewards = {}
    num_runs = 0
    plot_names = []

    experiment_mode = True
    if experiment_mode == True:
        for file_name in os.listdir(main_folder):
            if re.search(".*\.json", file_name):
                print(file_name)
                num_runs += 1
                
                with open(os.path.join(main_folder, file_name), 'r') as f:
                    experiment_list = json.load(f)
                    #print(experiment_list['run_date'])
                    
                for iterator, entry in enumerate(experiment_list):    
                    run = entry['output_dict']
                    avg_reward = [x['avg'] for x in run['stats']]
                    
                    #Make label
                    set_name = ""
                    set_name += "eps_max: " + str(entry['epsilon_max']) + " seed: " + str(entry['np_seed'])
                    plot_names.append(set_name)
                    
                    #Generate equivalent number of timesteps for x axis
                    #timesteps = [(x * pop_size * episode_len) for x in range(0,len(avg_reward))]
                    
                    #ax.plot(avg_reward, label= str(iterator))
                    if iterator in total_run_rewards:
                        temp_list = []
                        temp_list = [a + b for a,b in zip(total_run_rewards[iterator], avg_reward)]  
                        total_run_rewards[iterator] = temp_list[:]
                        
                    else:
                        total_run_rewards[iterator] = avg_reward


        #Plot each of the iterators
        for key, value in total_run_rewards.items():
            avg_value = [x / num_runs for x in value]
            ax.plot(avg_value, label= plot_names[key])

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.4,
                     box.width, box.height * 0.6])
            
        plt.title("Tester")
        plt.ylabel("Reward")
        plt.xlabel("# Generations")
        plt.ylim(-0.1, 1.1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, fancybox=True, shadow=True)
        save_name = str(round(time.time())) + "_avg_reward.jpg"
        plt.savefig("output.jpg")
        plt.show()      
                    

    else:
        for file_name in os.listdir(main_folder):
            if re.search(".*\.json", file_name):
                print(file_name)
                num_runs += 1
                
                with open(os.path.join(main_folder, file_name), 'r') as f:
                    output_dict = json.load(f)
                    print(output_dict['run_date'])
            
                #Prepare data set
                for iterator, run in enumerate(output_dict['runs']):
                    avg_reward = [x['avg'] for x in run['stats']]
                    
                    #Make label
                    set_name = ""
                    for entry in run['action_table']:
                        set_name += str(entry) + "-"
                    
                    #Generate equivalent number of timesteps for x axis
                    #timesteps = [(x * pop_size * episode_len) for x in range(0,len(avg_reward))]
                    
                    #ax.plot(avg_reward, label= str(iterator))
                    if iterator in total_run_rewards:
                        temp_list = []
                        temp_list = [a + b for a,b in zip(total_run_rewards[iterator], avg_reward)]  
                        total_run_rewards[iterator] = temp_list[:]
                        
                    else:
                        total_run_rewards[iterator] = avg_reward


        #Plot each of the iterators
        for key, value in total_run_rewards.items():
            avg_value = [x / num_runs for x in value]
            ax.plot(avg_value, label= str(key))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.4,
                     box.width, box.height * 0.6])
            
        plt.title("Tester")
        plt.ylabel("Reward")
        plt.xlabel("# Generations")
        #plt.ylim(-0.1, 1.1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, fancybox=True, shadow=True)
        save_name = str(round(time.time())) + "_avg_reward.jpg"
        plt.savefig("output.jpg")
        plt.show()      
    '''                
        