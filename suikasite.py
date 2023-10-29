import time
import utils
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

browser = webdriver.Chrome()
browser.get('https://suikagame.io')
browser.switch_to.frame("iframehtml5")
time.sleep(4)
gameWindow = browser.find_element(By.XPATH, "/html/body/section")
action = ActionChains(browser)
while not utils.checkGameOver(browser):
    time.sleep(0.1)
    action.move_to_element(gameWindow).move_by_offset(215,0).click().perform()
    utils.getPositions(browser)
    utils.getScore(browser)
print("-----GAME OVER-----")