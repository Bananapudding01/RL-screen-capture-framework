import yaml
import cv2
import mss
import numpy as np

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

game_region = config["game_region"]
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
        config["game_region"]["top"] = int(command)
    elif command == "left":
        command = input("left: ")
        config["game_region"]["left"] = int(command)
    elif command == "width":
        command = input("width: ")
        config["game_region"]["width"] = int(command)
    elif command == "height":
        command = input("height: ")
        config["game_region"]["height"] = int(command)

    img = np.array(sct.grab(config["game_region"]))
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
        config["input_shape"]["height"] = int(command)
    elif command == "width":
        command = input("width: ")
        config["input_shape"]["width"] = int(command)
    elif command == "gray":
        command = input("grayscale: ")
        if command == "true":
            config["preprocessing"]["grayscale"] = True
        elif command == "false":
            config["preprocessing"]["grayscale"] = False


    img = np.array(sct.grab(config["game_region"]))
    
    img = cv2.resize(
        img, 
        (config["input_shape"]["width"], 
         config["input_shape"]["height"]), 
         interpolation=cv2.INTER_AREA)

    targetWidth = 300
    height, width = img.shape[:2]
    ratio = height / width
    targetHeight = int(targetWidth * ratio)
    img = cv2.resize(
        img, 
        (targetWidth, targetHeight), 
         interpolation=cv2.INTER_NEAREST)
    
    if config["preprocessing"]["grayscale"] == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("preview", img)
    cv2.waitKey(1)
    command = input("command --> ")

cv2.destroyAllWindows()
with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)