class GameAdaptor:
    def __init__(self, env):
        print("TestAdaptor initialized")

    def resetinput(self):
        print("reset input")
    
    def stepinput(self, action):
        print("step input")

    def rewardinput(self):
        return 1

    def isDone(self):
        return False