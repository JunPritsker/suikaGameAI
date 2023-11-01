import time
import vars
import yaml
import utils
import torch
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import JavascriptException
from selenium.webdriver.common.action_chains import ActionChains

class SuikaGame:

    def __init__(self):
        self.browser = None
        self.action = None
        self.score = 0
        self.setupBrowser()

    def setupBrowser(self):
        print("setup browser")
        self.browser = webdriver.Chrome()
        self.browser.get('https://suikagame.io')
        time.sleep(1.5)
        self.browser.switch_to.frame("iframehtml5")
        self.startButton = self.browser.find_element(By.CLASS_NAME, "title-game-playing")
        self.startButton.click()
        time.sleep(1)
        self.browser.switch_to.frame("iframehtml5")
        self.gameWindow = self.browser.find_element(By.CLASS_NAME, "game-area")
        self.action = ActionChains(self.browser)
        self.previous_score = 0
        time.sleep(2) # give game time to initialize change from 5 -> 2

    def getCurrentFruit(self):
        while True: # Loop until there's a current fruit
            try:
                js = 'return cc.find("Canvas/lineNode/fruit")._components[3].bianjieX'
                result = self.browser.execute_script(js)
                currentFruit = self.fruitToOHE(float(result))
                print("currentFruit: ", currentFruit)
                return currentFruit
            except Exception as e:
                print("[*] getCurrentFruit EXCEPTION: ", e)
                if self.checkGameOver():
                    return [[0]*11]
        
    def getScore(self):
        try:
            js = 'return cc.find("Canvas/scorePanel/gameScore")._components[0]._string'
            result = self.browser.execute_script(js)
            self.score = int(result)
            return int(result)
        except JavascriptException:
            print("ERROR: couldn't get score")
            return self.score

    # get the positions and types of each fruit
    # POS 2 big values are x and y, second one is y. fruits of the same type will have the same y if they're both on the ground
    # Low X is left, high X is right.
    # output structure: [[fruits][position]]
    # structure detail: a single cherry dropped -> [[0,0,0,0,0,0,0,0,0,0,1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 359.16279069767444, 129.17978416468202, 0, 1]]
    # 2 cherries dropped -> [[0,0,0,0,0,0,0,0,0,0,1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 359.16279069767444, 129.17978416468202, 0, 1],
    #                        [0,0,0,0,0,0,0,0,0,0,1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 359.16279069767444, 129.17978416468202, 0, 1]]
    def getPositions(self):
        js = 'return cc.find("Canvas/fruitNode")._children.map(child => [child._components[1].angularVelocity, child._components[1].linearVelocity.x, child._components[1].linearVelocity.y, child.x, child.y, child._components[3].bianjieX]);'
        fruits = []
        try:
            positions = []
            result = self.browser.execute_script(js)
            for index in range(len(result)):
                angularVelocity = result[index][0] #float
                linearVelocityX = result[index][1] #float
                linearVelocityY = result[index][2] #float
                xPos = result[index][3] #float
                yPos = result[index][4] #float
                id = result[index][5] #string?
                # fruits.append(round(float(id),1))
                try:
                    fruitOHE = self.fruitToOHE(round(float(id),1)) # convert to float if necessary
                    # print("pos array: ", torch.FloatTensor([[angularVelocity, linearVelocityX, linearVelocityY, xPos, yPos] + fruitOHE]))
                    positions.extend([[angularVelocity, linearVelocityX, linearVelocityY, xPos, yPos] + fruitOHE]) # 16 values
                except Exception as e: #seems like this crashes because if it measures while a fruit pops, the fruit ID is set to 0?
                    print("[*] getPositionsLoop EXCEPTION: ", e)
                    exit()
            if len(result) == 0:
                # positions = torch.zeros([1,16], dtype=torch.float)
                positions = [[0]*16]
            # print("Current fruits: {}".format(fruits))
            return positions
        except JavascriptException:
            print("ERROR: ", JavascriptException)
            return [[0]*16]
        except Exception as e:
            print("[*] getPositions EXCEPTION: ", e)
            exit()

    def isMoving(self):
        for pos in self.getPositions():
            if not (pos[1] <= 8 and pos[2] <= 8): # Shouldn't need to check angular velocity because if it's rotating and moving, it'll have linear vel too. If it's just angular, it's spinning in place
                # print("[*] MOVING - fruit: {} xvel: {} yvel: {}".format(self.OHEtoFruitId(str(pos[5:])), pos[1], pos[2]))
                return True
        return False

    def getState(self):
        # current_fruit = game.pauseAngGetData(game.getCurrentFruit())
        # positions = game.pauseAndGetData(game.getPositions())
        current_fruit, positions = self.pauseAndGetData((self.getCurrentFruit(), self.getPositions()))
        if positions == []:
            positions = [[0] * 16] # there are 16 world matrix values
        for index in range(len(positions)):
            positions[index] = positions[index] + current_fruit # append current fruit to every position
        positions = torch.from_numpy(np.array(positions, dtype=float))
        return positions
    
    def getNextStates(self):
        # current_fruit = game.pauseAngGetData(game.getCurrentFruit())
        # positions = game.pauseAndGetData(game.getPositions())
        positions = self.getState()
        states = {}
        # states = []
        for xclick in range(-215,215,1): # for all mouse click position options
            # print("state append: {}, {}".format(xclick, positions))
            states[float(xclick)] = positions
            # states.append((xclick, positions)) # create a state with the mouse click positions and the state of the board
    
        # states = torch.tensor(states, dtype=float)
        # print("states: ", states)
        return states

    # lookup the fruit ID in the yml file and return a OHE of that fruit to identify it
    def fruitToOHE(self, fruitID):
        return vars.idToFruiteOHE[fruitID]
    
    def OHEtoFruitId(self, OHE):
        return vars.fruitOHEtoID[OHE]
    
    def checkGameOver(self):
        js = 'cc.find("Canvas/gameManager")._components[1].endOne'
        result = self.browser.execute_script(js)
        # gameEndDisplay = self.browser.find_element(By.ID, "GameEndScoreScreen") # if found, array should be > 0 in size
        # displayed = gameEndDisplay.get_attribute("style")
        return True if result == 1 else False

    def playStep(self, move):
        print("play step")
        prev_score = self.getScore()
        self.action.move_to_element(self.gameWindow).move_by_offset(move,0).click().perform()
        time.sleep(0.8) #slight delay because sometimes next move is too fast
        while self.isMoving(): # Wait for pieces to stop moving and not gameover
            if self.checkGameOver():
                break
        done = self.checkGameOver()
        reward = self.getScore() - prev_score # could also just give it a flat reward for increasing score
        return reward, done

    #TODO: hit the play button instead, although this only allows restart after game loss when the play button appears
    #TODO: there's probably a faster way to reset the game engine without refreshing the page and waiting for the game to load
    def restartGame(self):
        self.setupBrowser()
        return self.getState()
    
    def pauseGame(self):
        js = "cc.director.pause()"
        self.browser.execute_script(js)
    
    def resumeGame(self):
        js = "cc.director.resume()"
        self.browser.execute_script(js)
    
    def pauseAndGetData(self, functions):
        try:
            returns = []
            self.pauseGame()
            for func in functions:
                yield func
            # return returns
        finally:
            self.resumeGame()

# game = SuikaGame()
# shift = 0 # test slowly moving to the right to make balls move
# while not game.checkGameOver():
#     reward, done, score = game.play_step(shift)
#     time.sleep(0.75)
#     game.getPositions()
#     shift += 2
#     # print("reward: ", reward, "| done: ", done, "| score: ", score)
# print("-----GAME OVER-----")