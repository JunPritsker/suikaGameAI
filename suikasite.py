import time
import vars
import yaml
import utils
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import JavascriptException
from selenium.webdriver.common.action_chains import ActionChains

class SuikaGame:

    def __init__(self):
        self.browser = None
        self.action = None
        self.previous_score = 0
        self.setupBrowser()

    def setupBrowser(self):
        self.browser = webdriver.Chrome()
        self.browser.get('https://suikagame.io')
        time.sleep(1.5)
        self.browser.switch_to.frame("iframehtml5")
        self.gameWindow = self.browser.find_element(By.XPATH, "/html/body/section")
        self.action = ActionChains(self.browser)
        self.previous_score = 0
        time.sleep(5) # give game time to initialize

    def getCurrentFruit(self):
        with open("idToFruit.yml", "r") as f:
            idToFruit = yaml.safe_load(f)
            js = 'return cc.find("Canvas/lineNode/fruit")._components[3].bianjieX'
            result = self.browser.execute_script(js)
            return self.fruitToOHE(result)

    def getScore(self):
        try:
            js = 'return cc.find("Canvas/scorePanel/gameScore")._components[0]._string'
            result = self.browser.execute_script(js)
            self.previous_score = int(result)
            return int(result)
        except JavascriptException:
            print("ERROR: couldn't get score")
            return self.previous_score

    # get the positions and types of each fruit
    # POS 2 big values are x and y, second one is y. fruits of the same type will have the same y if they're both on the ground
    # Low X is left, high X is right.
    # output structure: [[fruits][position]]
    # structure detail: a single cherry dropped -> [[0,0,0,0,0,0,0,0,0,0,1], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 359.16279069767444, 129.17978416468202, 0, 1]]
    # 2 cherries dropped -> [[0,0,0,0,0,0,0,0,0,0,1], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 359.16279069767444, 129.17978416468202, 0, 1],
    #                        [0,0,0,0,0,0,0,0,0,0,1], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 359.16279069767444, 129.17978416468202, 0, 1]]
    def getPositions(self):
        js = 'return cc.find("Canvas/fruitNode")._children.map(child => [child._worldMatrix, child._components[3].bianjieX]);'
        try:
            positions= []
            world_matrix = []
            fruitOHE = []
            result = self.browser.execute_script(js)
            for index in range(len(result)):
                world_matrix = dict(result[index][0])["m"]
                # print("world matrix type: ", type(world_matrix), "| matrix: ", world_matrix)
                fruitOHE = vars.dict[float(result[index][1])]
                # print("fruitID: ", result[index][1], "| fruit OHE: ", fruitOHE)
                positions.extend([fruitOHE + world_matrix])
            print("POSITIONS: ", positions)
            return positions
        except JavascriptException:
            pass
            # print("ERROR: ", JavascriptException)

    # lookup the fruit ID in the yml file and return a OHE of that fruit to identify it
    def fruitToOHE(self, fruitID):
        return vars.dict[fruitID]
        
    def checkGameOver(self):
        gameEndDisplay = self.browser.find_element(By.ID, "GameEndScoreScreen") # if found, array should be > 0 in size
        displayed = gameEndDisplay.get_attribute("style")
        return True if "display: block" in displayed else False

    def play_step(self, move):
        prev_score = self.getScore()
        self.action.move_to_element(self.gameWindow).move_by_offset(move,0).click().perform()
        time.sleep(0.8) # approx time it takes the ball to fall
        done = self.checkGameOver()
        reward = self.getScore() - prev_score # could also just give it a flat reward for increasing score
        score = self.getScore()
        return reward, done, score

    def restart_game(self):
        self.setupBrowser()

# game = SuikaGame()
# shift = 0 # test slowly moving to the right to make balls move
# while not game.checkGameOver():
#     time.sleep(0.05)
#     reward, done, score = game.play_step(shift)
#     time.sleep(1)
#     game.getPositions()
#     shift += 2
#     # print("reward: ", reward, "| done: ", done, "| score: ", score)
# print("-----GAME OVER-----")