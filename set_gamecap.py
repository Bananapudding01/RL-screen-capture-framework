import yaml
import cv2
import mss
import numpy as np
import pygetwindow as getwindow
import threading
import ctypes

ctypes.windll.shcore.SetProcessDpiAwareness(0)

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)
    game = config["game"]

adaptorPath = "adaptors/" + game + "/" + game + "_config.yaml"

with open(adaptorPath, 'r') as f:
    adaptorConfig = yaml.safe_load(f)

application = adaptorConfig["application_name"]
window = getwindow.getWindowsWithTitle(application)[0]

sct = mss.mss()

# shared state for the preview thread
preview_mode = "raw"   # "raw" or "processed"
running = True

def get_real_cords(region):
    return {
        'left':   window.left + region['left'],
        'top':    window.top  + region['top'],
        'width':  region['width'],
        'height': region['height']
    }

def preview_loop():
    local_sct = mss.mss()
    while running:
        try:
            img = np.array(local_sct.grab(get_real_cords(adaptorConfig["game_region"])))

            if preview_mode == "processed":
                img = cv2.resize(
                    img,
                    (adaptorConfig["input_shape"]["width"],
                     adaptorConfig["input_shape"]["height"]),
                    interpolation=cv2.INTER_AREA)

                if adaptorConfig["preprocessing"]["grayscale"]:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if adaptorConfig["preprocessing"]["blackwhite"]:
                    _, img = cv2.threshold(
                        img,
                        adaptorConfig["preprocessing"]["threshold"],
                        255,
                        cv2.THRESH_BINARY)

                targetWidth = 300
                h, w = img.shape[:2]
                targetHeight = int(targetWidth * (h / w))
                img = cv2.resize(img, (targetWidth, targetHeight), interpolation=cv2.INTER_NEAREST)

            cv2.imshow("preview", img)
            cv2.waitKey(1)
        except Exception:
            pass

thread = threading.Thread(target=preview_loop, daemon=True)
thread.start()

# --- Stage 1: capture region ---
print("\nStage 1: Set capture region")
print("Commands: top / left / width / height / next / quit")
while True:
    command = input("command --> ").strip().lower()

    if command == "quit":
        running = False
        cv2.destroyAllWindows()
        exit()
    elif command == "next":
        break
    elif command == "top":
        val = input("top offset: ")
        adaptorConfig["game_region"]["top"] = int(val)
    elif command == "left":
        val = input("left offset: ")
        adaptorConfig["game_region"]["left"] = int(val)
    elif command == "width":
        val = input("width: ")
        adaptorConfig["game_region"]["width"] = int(val)
    elif command == "height":
        val = input("height: ")
        adaptorConfig["game_region"]["height"] = int(val)
    else:
        print("Unknown command")

# --- Stage 2: preprocessing ---
preview_mode = "processed"
print("\nStage 2: Set preprocessing")
print("Commands: height / width / gray / blackwhite / threshold / next / quit")
while True:
    command = input("command --> ").strip().lower()

    if command == "quit":
        running = False
        cv2.destroyAllWindows()
        exit()
    elif command == "next":
        break
    elif command == "height":
        val = input("height: ")
        adaptorConfig["input_shape"]["height"] = int(val)
    elif command == "width":
        val = input("width: ")
        adaptorConfig["input_shape"]["width"] = int(val)
    elif command == "gray":
        val = input("grayscale (true/false): ")
        adaptorConfig["preprocessing"]["grayscale"] = val == "true"
    elif command == "blackwhite":
        val = input("blackwhite (true/false): ")
        adaptorConfig["preprocessing"]["blackwhite"] = val == "true"
    elif command == "threshold":
        val = input("threshold (0-255): ")
        adaptorConfig["preprocessing"]["threshold"] = int(val)
    else:
        print("Unknown command")

running = False
cv2.destroyAllWindows()

with open(adaptorPath, 'w') as f:
    yaml.dump(adaptorConfig, f, default_flow_style=False)

print("Saved to " + adaptorPath)