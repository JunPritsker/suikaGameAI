from selenium import webdriver
import yaml
from selenium.webdriver.common.by import By
from selenium.common.exceptions import JavascriptException

def getCurrentFruit(browser):
    with open("idToFruit.yml", "r") as f:
        idToFruit = yaml.safe_load(f)
        js = 'return cc.find("Canvas/lineNode/fruit")._components[3].bianjieX'
        result = browser.execute_script(js)
        print("result: ", result)
        print(idToFruit["idToFruit"][result])

def getScore(browser):
    js = 'return cc.find("Canvas/scorePanel/gameScore")._components[0]._string'
    result = browser.execute_script(js)
    print("SCORE: ", result)

# get the positions and types of each fruit
# POS 2 big values are x and y, second one is y. fruits of the same type will have the same y if they're both on the ground
# Low X is left, high X is right.
def getPositions(browser):
    js = 'return cc.find("Canvas/fruitNode")._children.map(child => [child._worldMatrix, child._components[3].bianjieX]);'
    try:
        result = browser.execute_script(js)
        print("RESULT: ", result)
    except JavascriptException:
        pass
        # print("ERROR: ", JavascriptException)

def checkGameOver(browser):
    gameEndDisplay = browser.find_element(By.ID, "GameEndScoreScreen") # if found, array should be > 0 in size
    displayed = gameEndDisplay.get_attribute("style")
    return True if "display: block" in displayed else False