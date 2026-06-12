import json
import re
import subprocess


def parse_video_controllers(text):
    """Parses 'Name|AdapterRAM' lines from Win32_VideoController output.

    AdapterRAM is a 32-bit value on some drivers and caps at 4GB, so it is
    reported as 'reported_vram_gb' rather than a trustworthy total.
    """
    gpus = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        name, _, ram = line.rpartition("|")
        vram_gb = None
        if ram.strip().isdigit():
            vram_gb = round(int(ram.strip()) / (1024.0**3), 1)
        gpus.append({"name": name.strip(), "reported_vram_gb": vram_gb})
    return gpus


def main():
    info = {"vram_gb": 0, "ram_gb": 0, "gpu_name": "Unknown", "gpus": [], "error": None}
    try:
        # All video controllers (covers AMD/Intel/NVIDIA)
        cim_out = subprocess.check_output(
            [
                "powershell.exe",
                "-c",
                "Get-CimInstance Win32_VideoController | "
                "ForEach-Object { \"$($_.Name)|$($_.AdapterRAM)\" }",
            ],
            text=True,
        )
        info["gpus"] = parse_video_controllers(cim_out)

        # NVIDIA-specific detail (accurate VRAM total, but NVIDIA-only);
        # absence of nvidia-smi must not break the vendor-agnostic fields
        try:
            smi_out = subprocess.check_output(
                ["powershell.exe", "-c", "nvidia-smi -q -d MEMORY"], text=True
            )
            match = re.search(
                r"FB Memory Usage\s*\n\s*Total\s*:\s*(\d+)\s*MiB", smi_out
            )
            if match:
                info["vram_gb"] = round(int(match.group(1)) / 1024.0, 1)

            name_out = subprocess.check_output(
                ["powershell.exe", "-c", "nvidia-smi -L"], text=True
            )
            name_match = re.search(r"GPU 0: (.*?)\s*\(UUID", name_out)
            if name_match:
                info["gpu_name"] = name_match.group(1)
        except Exception:
            pass

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
