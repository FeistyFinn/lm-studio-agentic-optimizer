import argparse
import subprocess
import json
import sys

def main():
    parser = argparse.ArgumentParser(description="Download a model using the LM Studio CLI (lms).")
    parser.add_argument("--model", type=str, required=True, help="Model identifier (e.g. bartowski/gemma-4-12b-GGUF)")

    args = parser.parse_args()

    try:
        # Run lms get in powershell
        print(f"Executing: lms get \"{args.model}\"")
        # Setting stdout to None so it prints to the console directly for the agent to observe
        result = subprocess.run(["powershell.exe", "-c", f"lms get \"{args.model}\""], capture_output=True, text=True)

        if result.returncode == 0:
            output = {"success": True, "message": "Model downloaded successfully.", "stdout": result.stdout}
            print(json.dumps(output))
        else:
            output = {"success": False, "error": result.stderr}
            print(json.dumps(output))
            sys.exit(1)

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
