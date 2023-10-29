from selenium import webdriver
import yaml

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
    print("result: ", result)

# get the positions and types of each fruit
# POS 2 big values are x and y, second one is y. fruits of the same type will have the same y if they're both on the ground
# Low X is left, high X is right.
def getPositions(browser):
    js = 'return cc.find("Canvas/fruitNode")._children.map(child => [child._worldMatrix, child._components[3].bianjieX]);'
    result = browser.execute_script(js)
    print("result: ", result)
