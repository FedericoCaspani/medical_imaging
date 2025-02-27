import subprocess
import os

from src.object_detection.select_yolox_best_checkpoint import select_best_ckp

def train_yolox():
    # Set the PYTHONPATH environment variable
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "src/object_detection/YOLOX") + ":" + env.get("PYTHONPATH", "")

    # Define the training command
    train_command = [
        "python3", "src/object_detection/YOLOX/tools/train.py",
        "-expn", "yolox_nano_416_reproduce",
        "-f", "src/object_detection/YOLOX/exps/projects/rsna/yolox_nano_bre_416.py",
        "-d", "0",
        "-b", "64",
        "-o",
        "-c", "assets/public_pretrains/yolox_nano.pth",
        "seed", "42"
    ]

    # Define the evaluation command
    eval_command = [
        "python3", "src/object_detection/YOLOX/tools/eval.py",
        "-expn", "yolox_nano_416_reproduce",
        "-f", "src/object_detection/YOLOX/exps/projects/rsna/yolox_nano_bre_416.py",
        "-d", "0",
        "-b", "64",
        "seed", "42"
    ]

    # Define the command as a list
    convert_command = [
        "python3", "src/object_detection/YOLOX/tools/trt.py",
        "-expn", "yolox_nano_416_reproduce",
        "-f", "src/object_detection/YOLOX/exps/projects/rsna/yolox_nano_bre_416.py",
        "-b", "1"
    ]

    # Run training
    subprocess.run(train_command, env=env)

    # Run evaluation
    subprocess.run(eval_command, env=env)

    # Run the convertion
    subprocess.run(convert_command, env=env, check=True)

    # select and copy best checkpoint to final selection model dir
    select_best_ckp()
