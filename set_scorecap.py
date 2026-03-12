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

score_capture = adaptorConfig["score_capture"]
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
        adaptorConfig["score_capture"]["top"] = int(command)
    elif command == "left":
        command = input("left: ")
        adaptorConfig["score_capture"]["left"] = int(command)
    elif command == "width":
        command = input("width: ")
        adaptorConfig["score_capture"]["width"] = int(command)
    elif command == "height":
        command = input("height: ")
        adaptorConfig["score_capture"]["height"] = int(command)

    img = np.array(sct.grab(adaptorConfig["score_capture"]))
    cv2.imshow("preview", img)
    cv2.waitKey(1)

    command = input("command --> ")

    img = np.array(sct.grab(adaptorConfig["score_capture"]))
    
    img = cv2.resize(
        img, 
        (adaptorConfig["input_shape"]["width"], 
         adaptorConfig["input_shape"]["height"]), 
         interpolation=cv2.INTER_AREA)

    targetWidth = 300
    height, width = img.shape[:2]
    ratio = height / width
    targetHeight = int(targetWidth * ratio)

    cv2.imshow("preview", img)
    cv2.waitKey(1)
    command = input("command --> ")

cv2.destroyAllWindows()
with open(adaptorPath, 'w') as f:
    yaml.dump(adaptorConfig, f, default_flow_style=False)