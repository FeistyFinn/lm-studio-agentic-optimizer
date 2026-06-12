import json
import re
import subprocess


def main():
    info = {"vram_gb": 0, "ram_gb": 0, "gpu_name": "Unknown", "error": None}
    try:
        # Get GPU VRAM
        smi_out = subprocess.check_output(
            ["powershell.exe", "-c", "nvidia-smi -q -d MEMORY"], text=True
        )
        match = re.search(r"FB Memory Usage\s*\n\s*Total\s*:\s*(\d+)\s*MiB", smi_out)
        if match:
            info["vram_gb"] = round(int(match.group(1)) / 1024.0, 1)

        name_out = subprocess.check_output(
            ["powershell.exe", "-c", "nvidia-smi -L"], text=True
        )
        name_match = re.search(r"GPU 0: (.*?)\s*\(UUID", name_out)
        if name_match:
            info["gpu_name"] = name_match.group(1)

        # Get System RAM
        ram_out = subprocess.check_output(
            [
                "powershell.exe",
                "-c",
                "(Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize",
            ],
            text=True,
        )
        info["ram_gb"] = round(int(ram_out.strip()) / (1024.0 * 1024.0), 1)

    except Exception as e:
        info["error"] = str(e)

    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
