import pyautogui as gui

class GameAdaptor:
    def __init__(self, env):
        print("TestAdaptor initialized")
        gui.PAUSE = 0

    def resetinput(self):
        print("resetting...")
        gui.moveTo(953, 284)
        gui.sleep(0.2)
        gui.click()
        gui.sleep(0.2)

    
    def stepinput(self, action):
        if action == 0:
            gui.keyUp("down")
            gui.keyDown("up")
        elif action == 1:
            gui.keyUp("up")
            gui.keyDown("down")
        else:
            gui.keyDown("down")
            gui.keyDown("up")

    def rewardinput(self):
        if gui.pixel(953, 284) == (83, 83, 83):
            if gui.pixel(1039, 239) == (83, 83, 83):
                return -100
        return 1

    def isDone(self):
        if gui.pixel(953, 284) == (83, 83, 83):
            if gui.pixel(1039, 239) == (83, 83, 83):
                return True
        return False