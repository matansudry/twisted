import sys
sys.path.append(".")

import torch
from dm_control import mujoco
from datetime import datetime
import pickle
import os
import random
import networkx as nx
import time
import numpy as np
import cv2

from utils.topology import BFS, representation, reverse_BFS
from system_flow.low_level_class.random_planner import RandomPlanner
from system_flow.low_level_class.s2a_planner import S2APlanner
from system_flow.low_level_class.rl_planner import RLPlanner
from system_flow.graph.graph_manager import GraphManager
from utils.general_utils import calculate_number_of_crosses_from_topology_state,\
    convert_topology_to_str, load_pickle, get_current_primitive_state, physics_reset,\
    execute_action_in_curve_with_mujoco_controller, set_physics_state, convert_qpos_to_xyz_with_move_center,\
    convert_pos_to_topology, comperae_two_high_level_states
from utils.topology.BFS import get_state_score

with torch.no_grad():
    class HighLevelPlanner():
        def __init__(self, args, cfg):
            #save inputs
            self.args = args
            self.cfg = cfg

            #save config
            self.set_all_cfg_varibels()

            #set initial and goal states
            self.topology_start = self._generate_state_from_high_level_actions(representation.AbstractState(),\
                self.high_level_cfg.initial_state)
            self.topology_goal = self._generate_state_from_high_level_actions(representation.AbstractState(),\
                self.high_level_cfg.goal_state)

            #set mujoco
            self.args.env_path = self.high_level_cfg.env_path
            self.physics = mujoco.Physics.from_xml_path(self.args.env_path)
            self._physics_reset(self.physics)
            self.playground_physics = mujoco.Physics.from_xml_path(self.args.env_path)
            self._physics_reset(self.playground_physics)

            #init all levels
            if self.low_level_cfg.NAME != "RLPlanner":
                self.init_low_level(self.low_level_cfg)
            self.init_graph(self.graph_cfg)

            #create log file
            self.creaate_log_file()

            #load h values
            self.load_H_values_from_data(self.cfg.HIGH_LEVEL.h_path)

            #update max depth based on the goal topology
            self.high_level_cfg.max_depth = calculate_number_of_crosses_from_topology_state(self.topology_goal)

            self.reachable = {
                convert_topology_to_str(self.topology_start): self.topology_start
            }

            #init all running times
            self._init_run_time()

            #set initial_topology_states bandit
            self.bandit_initial_topology_states = {}
            self.bandit_select_trejctory_from_topology_state = {}

        def set_all_cfg_varibels(self):
            self.high_level_cfg = self.cfg["HIGH_LEVEL"]
            #self.mid_level_cfg = self.cfg["MID_LEVEL"]
            self.low_level_cfg = self.cfg["LOW_LEVEL"]
            self.graph_cfg = self.cfg["GRAPH"]

            self.config_length = self.low_level_cfg['STATE2STATE_PARMS']["config_length"]
            self.show_image = self.cfg.GENERAL_PARAMS.SAVE.show_image
            self.get_video = self.cfg.GENERAL_PARAMS.SAVE.get_video
            self.frame_rate = self.cfg.GENERAL_PARAMS.SAVE.frame_rate
            self.video = []
            self.num_of_links = self.low_level_cfg.STATE2STATE_PARMS.num_of_links
            self.random_search_steps = self.low_level_cfg.STATE2STATE_PARMS["random_search_steps"]

            #Random action
            self.low_index = self.low_level_cfg["RANDOM_ACTION"]["low_index"]
            self.high_index = self.low_level_cfg["RANDOM_ACTION"]["high_index"]
            self.low_height = self.low_level_cfg["RANDOM_ACTION"]["low_height"]
            self.high_height = self.low_level_cfg["RANDOM_ACTION"]["high_height"]
            self.high_end_location = self.low_level_cfg["RANDOM_ACTION"]["high_end_location"]
            self.low_end_location = self.low_level_cfg["RANDOM_ACTION"]["low_end_location"]

            self.output_path = self.cfg.GENERAL_PARAMS.output_path
            
        def _generate_state_from_high_level_actions(self, state, actions):
            for item in actions:
                #[MS] need to fix the number of parameters I am sending
                #RT1
                if item[0] == "Reide1":
                    state.Reide1(item[1], item[2], item[3])
                    continue

                #RT2
                elif item[0] == "Reide2":
                    if len(item[0]) == 4:
                        state.Reide2(item[1], item[2], item[3], item[4])
                    else:
                        state.Reide2(item[1], item[2], item[3])
                    continue

                #Cross
                elif item[0] == "cross":
                    state.cross(item[1], item[2], item[3])
                    continue
            return state        

        def creaate_log_file(self):
            self.log_file = {}
            self.log_file["paths"] = []
            self.log_file["error_primitive"] = []
            self.log_file["actions"] = []
            self.log_file["uncertainty"] = []
            self.log_file["error_topology"] = []
            self.log_file["topolgy_states"] = []
            self.log_file["success"] = []
            self.log_file["configuration"] = []

        def save_log(self, path):
            a_file = open(path+"/log.pkl", "wb")
            pickle.dump(self.log_file, a_file)
            a_file.close()

        def write_log_instance(self, key, value):
            if key not in self.log_file.keys():
                self.log_file[key] = []
            self.log_file[key].append(value)

        def set_new_goal(self, goal):
            self.topology_goal = goal
            self.high_level_cfg.max_depth = calculate_number_of_crosses_from_topology_state(self.topology_goal)

        def save_log_actions(self, log_actions):
            self.write_log_instance("log_actions", log_actions)

        def create_output_folder(self, path):
            self.time = datetime.now()
            self.time_name = str(self.time.month)+"_"+str(self.time.day)+"--"\
                +str(self.time.hour)+":"+str(self.time.minute)+":"+str(self.time.second)
            self.output_path = path+"/"+self.time_name
            os.mkdir(self.output_path)
        
        def _update_score(self, paths, h_score):
            """
            Get paths from "get_all_high_level_plan" and add score for eacxh trejctory

            Args:
                paths (list): paths
                h_score (list): topology score for each state

            Returns:
                update paths with score
            """
            for index in range(len(paths)):
                score = 0
                for state in paths[index][0]:
                    score += get_state_score(h_score, state)
                paths[index] = (paths[index][0], paths[index][1], score)
            return paths

        def get_all_high_level_plan(self,start, goal):
            if self.cfg.HIGH_LEVEL.h_path == "":
                with_h = False
            else:
                with_h = True
            max_depth = self.high_level_cfg.max_depth-calculate_number_of_crosses_from_topology_state(start)
            paths = BFS.bfs_all_path_new(
                start,
                goal,
                max_depth=max_depth,
                with_h=with_h,
                h_path=self.cfg.HIGH_LEVEL.h_path
                )

            #update score
            paths = self._update_score(paths, self.h_scores)

            return paths

        def get_all_reverse_high_level_plan(self, start, goal):
            max_depth = self.high_level_cfg.max_depth-calculate_number_of_crosses_from_topology_state(start)
            paths = reverse_BFS.bfs_all_path(goal, start, max_depth=max_depth)
            reverse_paths = []
            reverse_paths_action = []
            for path, path_action in paths:
                reverse_path = path[::-1]
                reverse_path_action = [representation.reverse_action(path_action[i], path[i], path[i+1])
                                    for i in range(len(path_action))]
                reverse_path_action = reverse_path_action[::-1]
                reverse_paths.append(reverse_path)
                reverse_paths_action.append(reverse_path_action)
            paths = reverse_paths
            paths_action = reverse_paths_action
            return paths, paths_action

        def load_H_values_from_data(self, path):
            self.h_scores = load_pickle(path)

        def init_low_level(self, low_cfg):
            if low_cfg.NAME == "S2APlanner":
                self.low_planner = S2APlanner(low_cfg, self.config_length)
            elif low_cfg.NAME == "RandomPlanner":
                self.low_planner = RandomPlanner(low_cfg, self.config_length)
            elif low_cfg.NAME == "RLPlanner":
                self.low_planner = RLPlanner(low_cfg, self.config_length)
            else:
                raise
            print("low level planner =", low_cfg.NAME)

        def init_graph(self, graph_cfg):
            self.graph_manager = GraphManager(config_length=self.config_length)
            self.graph_manager.add_node(self.topology_start, get_current_primitive_state(self.physics), None, None) 

        def _physics_reset(self, physics):
            physics_reset(physics)

        def _get_all_initial_topology_states(self, paths):
            """
            Concat all initial topology states and return set of them
        
            Args:
                paths (list): all high level plans
            """
            all_topology_states = []
            for path in paths:
                temp_state = path[0][0]
                if not temp_state in all_topology_states:
                    all_topology_states.append(temp_state)
            return all_topology_states
            
        def _bandit_select_topology_state(self, initial_topology_states):
            """
            select topology states from a list using bandits

            Args:
                initial_topology_states (lsit): list with all topology states
            """
            #add new states to bandit_initial_topology_states
            for state in initial_topology_states:
                state_str = convert_topology_to_str(state)
                if state_str not in self.bandit_initial_topology_states.keys():
                    self.bandit_initial_topology_states[state_str] = {
                        "cnt": 0
                    }

            #print the current dis
            dis = []
            for key in self.bandit_initial_topology_states.keys():
                dis.append(self.bandit_initial_topology_states[key]["cnt"])

            new_initial_topology_states = initial_topology_states

            #select unused states
            if self.high_level_cfg["SELECT_UNUSED_STATE"]:
                new_initial_topology_states = self._select_unused_state(initial_topology_states, dis)

            #give priority to states with higher number of crosses
            prob = np.ones(len(new_initial_topology_states)) / len(new_initial_topology_states)
            if self.high_level_cfg["SELECT_HIGHER_CROSS_STATES"]:
                prob = self._select_higher_cross_states(new_initial_topology_states)

            #select state
            topology_state_np = np.random.choice(new_initial_topology_states, 1, p=prob)
            topology_state = topology_state_np[0]

            #update the number of selection +=1
            self.bandit_initial_topology_states[convert_topology_to_str(topology_state)]["cnt"] +=1

            return topology_state

        def _select_unused_state(self, states, dis:list):
            new_states = []
            indexes = []
            for index, item in enumerate(dis):
                if item == 0:
                    indexes.append(index)
            for index in indexes:
                new_states.append(states[index])

            if len(new_states) == 0:
                return states
            else:
                return new_states

        def _select_higher_cross_states(self, states):
            prob = []
            for state in states:
                prob.append(calculate_number_of_crosses_from_topology_state(state)+1)
            prob = np.array(prob)
            return prob/sum(prob)

        def _bandit_select_trejctory_from_topology_state(self, topology_state, high_level_plans):
            """
            select high level plan based on the initial topology state

            Args:
                topology_state (topology_state): initial topology state
                high_level_plans (list): all high level plans
            """
            #extrect all high-level plans with initil topology state
            high_level_plans_with_topolgy_state = []
            for plan in high_level_plans:
                plan_str = ""
                for temp_state in plan[0]:
                    plan_str += convert_topology_to_str(temp_state)
                if plan_str not in self.bandit_select_trejctory_from_topology_state.keys():
                    self.bandit_select_trejctory_from_topology_state[plan_str] = {
                        "cnt": 0
                    }

                if topology_state == plan[0][0]:
                    high_level_plans_with_topolgy_state.append(plan)

            #print dis
            dis = []
            for key in self.bandit_select_trejctory_from_topology_state.keys():
                dis.append(self.bandit_select_trejctory_from_topology_state[key]["cnt"])

            #select plan
            number_of_plans = len(high_level_plans_with_topolgy_state)
            prob = np.ones(number_of_plans) / number_of_plans
            options = np.arange(0, number_of_plans, 1, dtype=int)
            path_index = np.random.choice(options, 1, p=prob)
            path = high_level_plans_with_topolgy_state[path_index[0]]

            #update the number of visits
            plan_str = ""
            for temp_state in path[0]:
                plan_str += convert_topology_to_str(temp_state)
            self.bandit_select_trejctory_from_topology_state[plan_str]["cnt"] +=1

            return path

        def _bendit_select_config_from_topology_state(self, topology_state):
            configurations, number_of_visits = self.graph_manager.get_all_states_with_topology_state(topology_state)

            new_configurations = configurations

            if self.high_level_cfg["SELECT_UNUSED_STATE"]:
                new_configurations = self._select_unused_state(configurations, number_of_visits)

            number_of_configurations = len(new_configurations)
            prob = np.ones(number_of_configurations) / number_of_configurations
            options = np.arange(0, number_of_configurations, 1, dtype=int)
            configuration_index = np.random.choice(options, 1, p=prob)
            configuration = new_configurations[configuration_index[0]]

            self.graph_manager.update_number_of_visits(configuration)

            return configuration

        def execute_action_in_curve(self, action, physics):
            if self.video is not None:
                physics = execute_action_in_curve_with_mujoco_controller(
                    physics=physics,
                    action=action,
                    get_video=self.get_video,
                    show_image=self.show_image,
                    return_render=False,
                    sample_rate=self.frame_rate,
                    video=self.video,
                    num_of_links=self.num_of_links,
                    env_path=self.args.env_path
                    )
            else:
                physics = execute_action_in_curve_with_mujoco_controller(
                    physics=physics,
                    action=action,
                    get_video=self.get_video,
                    show_image=self.show_image,
                    return_render=False,
                    sample_rate=self.frame_rate,
                    num_of_links=self.num_of_links,
                    env_path=self.args.env_path
                    )
            return physics

        def follow_plan(self, plan):
            #select configuration and set the physics
            new_config = self._bendit_select_config_from_topology_state(plan[0])
            set_physics_state(self.physics, new_config)
            
            #remove current state
            plan = plan[1:]
            success = True
            for state in plan:
                configuration = new_config
                parent_id = self.graph_manager.get_parent_id(configuration)
                set_physics_state(self.physics, configuration)    
                found_action = False
                selected_samples = self.low_planner.find_curve(configuration, state,\
                                                               self.physics, self.playground_physics)
                for index, sample in enumerate(selected_samples):
                    set_physics_state(self.physics, configuration)
                    self.physics = self.execute_action_in_curve(sample['action'], self.physics)
                    new_primitive_state = get_current_primitive_state(self.physics)
                    new_pos_state = convert_qpos_to_xyz_with_move_center(self.playground_physics, new_primitive_state)
                    new_topology_state_precidtion = convert_pos_to_topology(new_pos_state)

                    #if we found good action
                    if comperae_two_high_level_states(new_topology_state_precidtion, state):
                        found_action = True
                        new_config = new_primitive_state
                        
                        print("good action had target number of crosse are =",\
                              calculate_number_of_crosses_from_topology_state(state),\
                            ", index =", index, ", prediction_uncertainty =",sample['prediction_uncertainty'],\
                            ",ensemble_uncertainty =",\
                            sample['ensemble_uncertainty'], ",topology_uncertainty =",\
                            sample['topology_uncertainty'])

                    state_str = convert_topology_to_str(new_topology_state_precidtion)
                    
                    #if we found new topology state
                    if len(new_topology_state_precidtion.points) >= len(state.points):
                        if state_str not in self.reachable:
                            self.reachable[state_str] = new_topology_state_precidtion
                            new_plans = self.get_all_high_level_plan(new_topology_state_precidtion, self.topology_goal)
                            self.high_level_plans.extend(new_plans)
                        
                        #if we found new configuration that has at least number of crosses as the target.
                        _ = self.graph_manager.add_node(new_topology_state_precidtion, new_primitive_state, parent_id,\
                                sample['action'])
                if found_action == False:
                    success = False
                    break
            return success

        def _select_topology_state_from_reachable_options(self, all_states, states_with_plan, return_state_without_plan=True):

            states_options = list(all_states.values())
            
            if return_state_without_plan:
                unique_options = [state for state in states_options if (state not in states_with_plan)]
                states_options = unique_options

            prob = np.ones(len(states_options)) / len(states_options)
            state_np = np.random.choice(states_options, 1, p=prob)
            state = state_np[0]
            #state = random.choice(states_options)

            return state

        def _generate_random_action(self, size=1):
            int_part = torch.randint(self.low_index, self.high_index,(size,1))
            continues_part = torch.rand(size,3)
            #height
            continues_part[:,0] *= self.high_height - self.low_height
            continues_part[:,0] += self.low_height

            #x,y part
            continues_part[:,1] *= self.high_end_location - self.low_end_location
            continues_part[:,1] += self.low_end_location
            continues_part[:,2] *= self.high_end_location - self.low_end_location
            continues_part[:,2] += self.low_end_location

            #concat
            batch = torch.cat((int_part,continues_part), 1)
            return batch

        def expand(self, topology_state):
            #select configuration and set the physics
            configuration = self._bendit_select_config_from_topology_state(topology_state)
            set_physics_state(self.playground_physics, configuration)
            old_pos_state = convert_qpos_to_xyz_with_move_center(self.playground_physics, configuration)
            old_topology_state_precidtion = convert_pos_to_topology(old_pos_state)
            parent_id = self.graph_manager.get_parent_id(configuration)

            for _ in range(self.random_search_steps):
                set_physics_state(self.playground_physics, configuration)
                action = self._generate_random_action()
                self.playground_physics = self.execute_action_in_curve(action[0], self.playground_physics)
                new_primitive_state = get_current_primitive_state(self.playground_physics)
                new_pos_state = convert_qpos_to_xyz_with_move_center(self.playground_physics, new_primitive_state)
                new_topology_state_precidtion = convert_pos_to_topology(new_pos_state)

                state_str = convert_topology_to_str(new_topology_state_precidtion)
                if state_str not in self.reachable and len(new_topology_state_precidtion.points) >=\
                     len(old_topology_state_precidtion.points):
                    self.reachable[state_str] = new_topology_state_precidtion
                    new_plans = self.get_all_high_level_plan(new_topology_state_precidtion, self.topology_goal)
                    self.high_level_plans.extend(new_plans)
                    _ = self.graph_manager.add_node(new_topology_state_precidtion, new_primitive_state, parent_id,\
                            action[0])
                    break

        def save_video(self, path):
            if not self.get_video:
                print("video == False, no vidoe was generated")
                return
            index = 0
            check = True
            name = "project.avi"
            while check:
                if os.path.isfile(path +"/"+ str(index)+"_"+name):
                    index += 1
                else:
                    check=False
            new_path = path +"/"+ str(index)+"_"+name

            out = cv2.VideoWriter(new_path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (640,480))
            for i in range(len(self.video)):
                out.write(self.video[i])
            out.release()

        def save(self):
            #create output folder
            self.create_output_folder(self.output_path)
            self.save_log(self.output_path)
            self.save_video(self.output_path)
            self.save_graph(self.output_path)

        def save_graph(self, path):
            nx.write_gpickle(self.graph_manager.graph, path+"/graph.gpickle")

        def _init_run_time(self):
            self.time_get_all_high_level_plan = 0
            self.time_get_all_initial_topology_states = 0
            self.time_bandit_select_topology_state = 0
            self.time_bandit_select_trejctory_from_topology_state = 0
            self.time_follow_plan = 0
            self.time_expand = 0

        def run(self):

            if self.low_level_cfg.NAME == "RLPlanner":
                self.init_low_level(self.low_level_cfg)


            st = time.time()
            self.high_level_plans = self.get_all_high_level_plan(self.topology_start, self.topology_goal)
            et = time.time()
            self.time_get_all_high_level_plan += et-st
            
            while True:
                #get all inital topology states
                st = time.time()
                self.initial_topology_states = self._get_all_initial_topology_states(self.high_level_plans)
                et = time.time()
                self.time_get_all_initial_topology_states += et-st
                

                #select topolgy state using bandits
                st = time.time()
                topology_state = self._bandit_select_topology_state(self.initial_topology_states)
                et = time.time()
                self.time_bandit_select_topology_state += et-st
                
                
                #select plan with initial topology state, also bandits
                st = time.time()
                selected_plan = self._bandit_select_trejctory_from_topology_state(topology_state,\
                     self.high_level_plans)
                et = time.time()
                self.time_bandit_select_trejctory_from_topology_state += et-st
                

                     
                #follow plan
                st = time.time()
                _ = self.follow_plan(selected_plan[0])
                et = time.time()
                self.time_follow_plan += et-st
                

                #solution was found
                if self.graph_manager.check_topology_goal(self.topology_goal):
                    trajectory = self.graph_manager.trajctory_extractor(self.topology_goal)
                    self.write_log_instance("goal_found", True)
                    self.write_log_instance("trajectory", trajectory)
                    self.save()
                    return True
                
                #in some probability we will run it totaly random
                elif random.random() < self.low_level_cfg.random_search_threshold:
                    topogly_state_for_expend = self._select_topology_state_from_reachable_options(self.reachable,\
                            self.initial_topology_states, return_state_without_plan=False)

                    #search randomly
                    st = time.time() 
                    self.expand(topogly_state_for_expend) 
                    et = time.time()
                    self.time_expand += et-st
                    