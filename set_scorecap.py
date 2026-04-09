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

sct = mss.mss()

print("Set scorecap cords:")
print("Commands: top / left / width / height / next / quit")
while True:
    img = np.array(sct.grab(adaptorConfig["score_capture"]))
    cv2.imshow("preview", img)
    cv2.waitKey(1)

    command = input("command --> ").strip().lower()

    if command == "quit":
        cv2.destroyAllWindows()
        exit()
    elif command == "next":
        break
    elif command == "top":
        val = input("top: ")
        adaptorConfig["score_capture"]["top"] = int(val)
    elif command == "left":
        val = input("left: ")
        adaptorConfig["score_capture"]["left"] = int(val)
    elif command == "width":
        val = input("width: ")
        adaptorConfig["score_capture"]["width"] = int(val)
    elif command == "height":
        val = input("height: ")
        adaptorConfig["score_capture"]["height"] = int(val)
    else:
        print("Unknown command")

cv2.destroyAllWindows()

with open(adaptorPath, 'w') as f:
    yaml.dump(adaptorConfig, f, default_flow_style=False)

print("Saved to " + adaptorPath)
