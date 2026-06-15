from hw_info import parse_video_controllers


def test_parses_multiple_gpus():
    text = (
        "NVIDIA RTX A2000|4293918720\n"
        "AMD Radeon (TM) Pro WX 7100 Graphics|4293918720\n"
    )

    gpus = parse_video_controllers(text)

    assert [g["name"] for g in gpus] == [
        "NVIDIA RTX A2000",
        "AMD Radeon (TM) Pro WX 7100 Graphics",
    ]
    assert gpus[0]["reported_vram_gb"] == 4.0


def test_handles_missing_ram_value():
    gpus = parse_video_controllers("Microsoft Basic Display Adapter|\n")

    assert gpus == [
        {"name": "Microsoft Basic Display Adapter", "reported_vram_gb": None}
    ]


def test_ignores_blank_and_malformed_lines():
    assert parse_video_controllers("\n\njust text without pipe\n") == []
