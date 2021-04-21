import itertools
import datetime
import sys
import csv

from spirecomm.communication.coordinator import Coordinator
from agent import Agent
from spirecomm.spire.character import PlayerClass
from spirecomm.spire.game import Game
from spirecomm.spire.screen import ScreenType
from spirecomm.communication.action import Action, StartGameAction, PlayCardAction, EndTurnAction
from spirecomm.spire.character import Player
from spirecomm.spire.card import CardType

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import gym
from gym import spaces

class SpireEnv(gym.Env):
    def __init__(self, player_class, cardsDictionary, idDictionary, ascension_level=0, seed=None):

        self.num_of_cards = len(cardsDictionary)

        self.action_space = spaces.Discrete(self.num_of_cards + 1)

        observation_space_min = []
        observation_space_min.extend([0, 1, 1, 1, 1, 0, 0, 0]) #Game and player state
        observation_space_min.extend([0, 1, 0, -1] * 5) #Enemies
        observation_space_min.extend([0] * self.num_of_cards) #Cards

        observation_space_max = []
        observation_space_max.extend([20, 4, 60, 100, 200, 200, 7, 100]) #Game and player state
        observation_space_max.extend([1000, 1000, 50, 80] * 5) #Enemies
        observation_space_max.extend([10] * self.num_of_cards) #Cards

        #game_space = [21, 3, 50, 100, 200, 200, 20, 1000]
        #enemy_space = [2000, 2000, 200, 100] * 5
        #card_space = [10] * self.num_of_cards

        #observation_space_discrete = []
        #observation_space_discrete.extend(game_space)
        #observation_space_discrete.extend(enemy_space)
        #observation_space_discrete.extend(card_space)

        # self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        # self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)
        # self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.low = np.array(observation_space_min, dtype=np.int)
        self.high = np.array(observation_space_max, dtype=np.int)

        #self.observation_space = spaces.MultiDiscrete(observation_space_discrete)         # NOOOO, this might need to be a box

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.int) #Might need to try float too

        #try mixing box for game and enemy state and discrete for card space using a tuple

        self.game = Game()

        self.cards_dictionary = cardsDictionary
        self.id_dictionary = idDictionary

        self.player_class = player_class
        self.ascension_level = ascension_level
        self.seed = seed

        self.agent = Agent(self.player_class) #For all parts of the game except combat

        self.coordinator = Coordinator()
        self.coordinator.signal_ready()

        self.coordinator.register_command_error_callback(self.handle_error)
        self.coordinator.register_state_change_callback(self.update_game_state)
        self.coordinator.register_out_of_game_callback(self.lost)

        log("Initialised, ready command sent.")

    def handle_error(self, error):
        log("Handle Error: " + str(error) + "\n")
        #raise Exception(error)

    def lost(self):
        self.done = True

    def start_game_action(self):
        log("Starting game\n")
        return StartGameAction(self.player_class, self.ascension_level, self.seed)
    
    def update_game_state(self, game_state): #this wants me to return an action
        log("Game state updated\n")
        self.game = game_state

    def format_state(self, state):
        #Player and game state
        state_vector = [state.ascension_level, state.act, state.floor, state.turn, state.max_hp, state.current_hp, state.player.energy, state.player.block] #TODO: pass powers as well

        #Enemies
        enemy_vector = [0] * 20 #4 values per monster, max 5 monsters
        m = 0
        for monster in [monster for monster in state.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]:      #Only do alive monsters
            offset = m * 4
            enemy_vector[offset + 0] = monster.current_hp
            enemy_vector[offset + 1] = monster.max_hp
            enemy_vector[offset + 2] = monster.block
            enemy_vector[offset + 3] = monster.move_adjusted_damage * monster.move_hits
            m = m + 1
            pass

        state_vector.extend(enemy_vector)

        #Playable cards TODO: add unique id for each card

        try:
            #TODO: change this such that the position of the card in the hand does not change the weight, simply have 370 values where the number is how many are in the hand
            
            playable_cards_vector = [0] * self.num_of_cards
            
            for card in [card for card in state.hand if card.is_playable]:
                try:
                     cardID = self.cards_dictionary[card.card_id]

                     playable_cards_vector[int(cardID) - 1] += 1

                except Exception as e:
                    log("Failed to lookup card: " + card.card_id)
                    #raise(e)
            
            # playable_cards_vector = [0] * (10 * (1 + 370 + 1)) #10 possible playable cards, 1 value for cost and 370 representing all the possible cards plus 1 for upgraded
            # c = 0
            # for card in [card for card in state.hand if card.is_playable]:
            #     offset = c * 372
            #     if card.cost == "X":
            #         playable_cards_vector[offset + 0] = state.player.energy
            #     else:
            #         playable_cards_vector[offset + 0] = card.cost

            #     try:
            #         cardID = self.cards_dictionary[card.card_id]
            #         playable_cards_vector[offset + int(cardID)] = 1
            #     except:
            #         log("Failed to lookup card: " + card.card_id)
            #         quit()

            #     playable_cards_vector[offset + 371] = card.upgrades

            #     #Need to lookup card.id into a card id dictionary
            #     #playable_cards_vector[offset + 1] = card.id

            #     c = c + 1

            state_vector.extend(playable_cards_vector)

        except Exception as e:
            log(str(e))

        #TODO Draw pile

        #TODO Discard pile

        #TODO Exhaust pile

        #TODO Relics

        #TODO Potions

        log(state_vector)
        return state_vector

    def calculate_reward(self, foundCard): #Should get reward for killing individual monsters too
        try:
            if self.game.screen_type == ScreenType.BOSS_REWARD:
                log("Killed boss")
                self.last_enemy_health = 0
                #Killed boss
                return 0.8

            if self.game.screen_type == ScreenType.GAME_OVER:
                log("game over")
                return -2

            monsters = [monster for monster in self.game.monsters if not monster.half_dead and not monster.is_gone]
            monsterHP = 0
            totalMonsterAttack = 0
            for monster in monsters:
                monsterHP += monster.current_hp
                totalMonsterAttack += monster.move_adjusted_damage * monster.move_hits

            if(totalMonsterAttack < 0):
                totalMonsterAttack = 0
                
            log("HP Difference: " + str((self.game.current_hp - self.last_health)))
            log("Enemy HP Difference: " + str((monsterHP - self.last_enemy_health)))

            blockedDamage = 0

            try:
                if totalMonsterAttack < self.game.player.block:
                    blockedDamage = totalMonsterAttack
                else:
                    blockedDamage = self.game.player.block
            except:
                log("No block.")

            log("Blocked: " + str(blockedDamage))

            reward = ((self.last_enemy_health - monsterHP) / 100) + (blockedDamage / 100)

            if foundCard:
                reward += 0.01
            else:
                reward -= 0.0001

            if self.game.screen_type == ScreenType.COMBAT_REWARD: #This might be redundant
                log("Combat reward")
                self.last_enemy_health = 0
                #Ended fight
                reward += 0.4

            return reward
        except Exception as e:
            log(str(e))
            return 0
    
    def assign_last_game_state_variables(self):
        #Assign last game state variables
        self.last_health = self.game.current_hp
        self.last_enemy_health = 0
        monsters = [monster for monster in self.game.monsters if not monster.half_dead and not monster.is_gone]
        for monster in monsters:
            self.last_enemy_health += monster.current_hp

    def step(self, action):
        log("Step\n")
        assert self.action_space.contains(action)

        self.assign_last_game_state_variables()

        log("Started new Step with action: " + str(action) + "\n")
        #Convert to playable card list

        playable_cards = [card for card in self.game.hand if card.is_playable]

        if action == self.num_of_cards: #End turn
            coordinatorAction = EndTurnAction()
            if len(playable_cards) > 0:
                foundCard = False
            else:
                foundCard = True
        else:

            try:
                desiredCardID = self.id_dictionary[str(int(action) + 1)]
            except:
                log("Failed to look up card with numerical ID " + str(action + 1))
                log(self.id_dictionary)
                desiredCardID = 999

            cardToPlay = None
            foundCard = False

            log("AI wants to play card: " + str(desiredCardID) + "\n")

            for playable_card in playable_cards:
                if playable_card.card_id == desiredCardID:
                    cardToPlay = playable_card
                    log("Card is in hand!")
                    foundCard = True

            if foundCard:
                log("Playing: " + cardToPlay.card_id)
                if cardToPlay.has_target:
                    available_monsters = [monster for monster in self.game.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
                    if len(available_monsters) == 0:
                        coordinatorAction = EndTurnAction()
                    else:
                        if cardToPlay.type == CardType.ATTACK:
                            target = self.agent.get_low_hp_target()
                        else:
                            target = self.agent.get_high_hp_target()
                        coordinatorAction = PlayCardAction(card=cardToPlay, target_monster=target)
                else:
                    coordinatorAction = PlayCardAction(card=cardToPlay)
            else:
                return self.last_state, -0.0001, self.done, {}

            #log("No card found, ending turn")
            #coordinatorAction = EndTurnAction()
            
        self.coordinator.add_action_to_queue(coordinatorAction)
        log("Executing action\n")
        self.coordinator.execute_next_action_if_ready()
        log("Waiting for game state update\n")
        self.coordinator.receive_game_state_update(block=True)

        #Assign reward (before the agent is out of combat)
        reward = self.calculate_reward(foundCard)

        log("Assigning step reward: " + str(reward) + "\n")

        if self.game.screen_type == ScreenType.GAME_OVER:
            log("game over")
            self.done = True
            log("Detected game over")
            if(self.game.screen.victory):
                self.ascension_level += 1
                self.ascension_level = min(20, self.ascension_level)
                reward = 10 #we won
                log("VICTORY")
            else:
                reward = -2
        else:
            agentAction = self.agent.get_next_action_in_game(self.game)
            while agentAction != -1: #Play until we are in combat again    TODO (Important): assign reward after making discarding decisions etc otherwise they will be counted as useless
                self.in_combat = False
                #self.coordinator.add_action_to_queue(agentAction)
                #self.coordinator.execute_next_action_if_ready()

                log("Executing agent action\n")
                self.coordinator.add_action_to_queue(agentAction)
                self.coordinator.execute_next_action_if_ready()
                log("Receiving next game state\n")
                test = self.coordinator.receive_game_state_update(block=True)

                if self.game.screen_type == ScreenType.GAME_OVER:
                    log("game over")
                    self.done = True
                    log("Detected game over")
                    if(self.game.screen.victory):
                        self.ascension_level += 1
                        self.ascension_level = min(20, self.ascension_level)
                        reward = 10 #we won
                        log("VICTORY")
                    else:
                        reward = -2

                if self.done:
                    log("done by 2\n") #because a game over is detected before the agent does actions
                    return self.last_state, reward, self.done, {}

                log("Getting further action from agent\n")
                agentAction = self.agent.get_next_action_in_game(self.game)

        if self.done:
            log("done by 2\n") #because a game over is detected before the agent does actions
            return self.last_state, reward, self.done, {}

        log("Making next state for network\n")
        #make game state neural network friendly
        state = self.format_state(self.game)
        self.last_state = state

        return state, reward, self.done, {}

    def reset(self):

        while self.coordinator.in_game:
            log("Cant reset yet, agent is navigating menus")
            agentAction = self.agent.get_next_action_in_game(self.game)
            self.coordinator.add_action_to_queue(agentAction)
            self.coordinator.execute_next_action_if_ready()
            self.coordinator.receive_game_state_update(block=True)

                
        log("Resetting")

        self.last_health = self.game.max_hp
        self.last_enemy_health = 0
        self.done = False
        self.in_combat = False

        #self.coordinator.game_is_ready = False
        self.coordinator.in_game = False
        self.coordinator.last_game_state = None

        log("Clearing actions")

        self.coordinator.clear_actions()

        while not self.coordinator.game_is_ready:
            log("Waiting for game to be ready")
            self.coordinator.receive_game_state_update(block=True, perform_callbacks=False)
        if not self.coordinator.in_game:
            log("Starting new game with " + str(self.player_class) + "\n")
            self.start_game_action().execute(self.coordinator)
            self.coordinator.receive_game_state_update(block=True)

        agentAction = self.agent.get_next_action_in_game(self.game)
        while agentAction != -1: #Play until we are in combat again
            self.coordinator.add_action_to_queue(agentAction)
            self.coordinator.execute_next_action_if_ready()
            self.coordinator.receive_game_state_update(block=True)

            agentAction = self.agent.get_next_action_in_game(self.game)

        log("Environment is ready, combat has begun\n")

        state = self.format_state(self.game)
        self.last_state = state
        return state

    def render(self, mode='human'):
        #set game speed
        pass

    def close(self):
        pass

def log(message):
    f = open("log.txt", 'a')
    f.write(str(message) + "\n")
    f.close()

f = open("log.txt", "w")
f.write("")
f.close()

cardsDictionary = {}
idDictionary = {}

log("Opening cards database")

with open("CardsIroncladSmall.csv", mode='r') as cardsFile:
    reader = csv.reader(cardsFile)
    cardsDictionary = {rows[1]:rows[0] for rows in reader}

with open("CardsIroncladSmall.csv", mode='r') as cardsFile:
    reader = csv.reader(cardsFile)
    idDictionary = {rows[0]:rows[1] for rows in reader}

log("Loaded cards data:")
log(cardsDictionary)
log(idDictionary)

log("Starting environment")

spireEnv = SpireEnv(PlayerClass.IRONCLAD, cardsDictionary, idDictionary)

nb_actions = spireEnv.action_space.n


log("Creating neural network")

model = Sequential()
model.add(Flatten(input_shape=(1,) + spireEnv.observation_space.shape))            #512 -> 256
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('relu'))

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1, target_model_update=1e-3, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae']) #1e-3

#dqn.load_weights('dqn_{}_weights_fail.h5f'.format("SlayTheSpire"))

try:
    dqn.fit(spireEnv, nb_steps=2000000, visualize=True, verbose=0)

    dqn.save_weights('dqn_{}_weights.h5f'.format("SlayTheSpire"), overwrite=True)
except Exception as e:
    log("Error: " + str(e))

    dqn.save_weights('dqn_{}_weights_fail.h5f'.format("SlayTheSpire"), overwrite=True)

    raise(e)
    
    
    # spireEnv.reset() #Start a game
    # while True:
    #     state, reward, done, info = spireEnv.step(spireEnv.action_space.sample())
    #     if done:
    #         break
    
    # log("done")
    # spireEnv.close()
