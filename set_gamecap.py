import yaml
import cv2
import mss
import numpy as np

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

    game = config["game"]

adaptorPath = "adaptors/" + game + "/" + game + "_config.yaml"

with open(adaptorPath, 'r') as f:
    adaptorConfig = yaml.safe_load(f)

game_region = adaptorConfig["game_region"]
sct = mss.mss()
command = ""
print("Set gamecap cords:")
while True:

    if command == "quit":
        cv2.destroyAllWindows()
        exit()
    elif command == "next":
        break
    elif command == "top":
        command = input("top: ")
        adaptorConfig["game_region"]["top"] = int(command)
    elif command == "left":
        command = input("left: ")
        adaptorConfig["game_region"]["left"] = int(command)
    elif command == "width":
        command = input("width: ")
        adaptorConfig["game_region"]["width"] = int(command)
    elif command == "height":
        command = input("height: ")
        adaptorConfig["game_region"]["height"] = int(command)

    img = np.array(sct.grab(adaptorConfig["game_region"]))
    cv2.imshow("preview", img)
    cv2.waitKey(1)

    command = input("command --> ")

command = ""   
print("set preprocessing:")
while True:

    if command == "quit":
        cv2.destroyAllWindows()
        exit()
    elif command == "next":
        break
    elif command == "height":
        command = input("height: ")
        adaptorConfig["input_shape"]["height"] = int(command)
    elif command == "width":
        command = input("width: ")
        adaptorConfig["input_shape"]["width"] = int(command)
    elif command == "gray":
        command = input("grayscale: ")
        if command == "true":
            adaptorConfig["preprocessing"]["grayscale"] = True
        elif command == "false":
            adaptorConfig["preprocessing"]["grayscale"] = False


    img = np.array(sct.grab(adaptorConfig["game_region"]))
    
    img = cv2.resize(
        img, 
        (adaptorConfig["input_shape"]["width"], 
         adaptorConfig["input_shape"]["height"]), 
         interpolation=cv2.INTER_AREA)

    targetWidth = 300
    height, width = img.shape[:2]
    ratio = height / width
    targetHeight = int(targetWidth * ratio)
    
    if adaptorConfig["preprocessing"]["grayscale"] == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if adaptorConfig["preprocessing"]["blackwhite"] == True:
        _, img = cv2.threshold(img, adaptorConfig["preprocessing"]["threshold"], 255, cv2.THRESH_BINARY)

    img = cv2.resize(
        img, 
        (targetWidth, targetHeight), 
         interpolation=cv2.INTER_NEAREST)
    
    if adaptorConfig["preprocessing"]["blackwhite"] == True:
        _, img = cv2.threshold(img, adaptorConfig["preprocessing"]["threshold"], 255, cv2.THRESH_BINARY)

    cv2.imshow("preview", img)
    cv2.waitKey(1)
    command = input("command --> ")

cv2.destroyAllWindows()
with open(adaptorPath, 'w') as f:
    yaml.dump(adaptorConfig, f, default_flow_style=False)