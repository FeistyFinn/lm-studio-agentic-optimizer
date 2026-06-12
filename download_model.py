import argparse
import json
import re
import subprocess
import sys

# Model identifiers look like "publisher/model-name" or "model-name@quant".
# Anything outside this set is rejected because the value is passed to a
# PowerShell command line and could otherwise inject arbitrary commands.
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9._@/-]+$")


def main():
    parser = argparse.ArgumentParser(
        description="Download a model using the LM Studio CLI (lms)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (e.g. bartowski/gemma-4-12b-GGUF)",
    )

    args = parser.parse_args()

    if not MODEL_ID_PATTERN.match(args.model):
        print(
            json.dumps(
                {
                    "success": False,
                    "error": (
                        "Invalid model identifier. Allowed characters: "
                        "letters, digits, '.', '_', '@', '/', '-'."
                    ),
                }
            )
        )
        sys.exit(1)

    try:
        print(f"Executing: lms get {args.model}")
        result = subprocess.run(
            ["powershell.exe", "-c", "lms", "get", "--yes", args.model],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            output = {
                "success": True,
                "message": "Model downloaded successfully.",
                "stdout": result.stdout,
            }
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
