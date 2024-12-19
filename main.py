import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run VQA training script.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file."
    )
    args = parser.parse_args()

    subprocess.run(["python", "./src/scripts/train.py", "--config", args.config])

if __name__ == "__main__":
    main()