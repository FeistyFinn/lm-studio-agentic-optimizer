import json

from lm_studio_client import LMStudioHardwareClient


def main():
    try:
        client = LMStudioHardwareClient()
        results = [{"model_key": key} for key in client.list_models()]
        print(json.dumps({"success": True, "models": results}, indent=2))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))


if __name__ == "__main__":
    main()
