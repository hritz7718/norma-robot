import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "software/station/shared"))

from station_py import new_station_client

async def main():
    phase = sys.argv[1].lower()
    if phase not in ("start", "end"):
        raise SystemExit("Use: start or end")

    client = await new_station_client(
        "localhost", logging.getLogger("recording")
    )

    reader = client.read_from_tail(
        "inference-states",
        offset=b"\x00",
        limit=1,
        step=1,
        buf_size=1,
    )

    entry = await asyncio.wait_for(reader.data.get(), 10)
    if entry is None:
        raise RuntimeError(f"No inference state: {reader.err}")

    pointer = int.from_bytes(entry.ID.ID, "little")
    output = ROOT / "local/recording-bounds.env"

    if phase == "start":
        output.write_text(f"START={pointer}\n")
    else:
        with output.open("a") as f:
            f.write(f"END={pointer}\n")

    print(f"{phase.upper()} pointer: {pointer}")
    print(f"Saved to {output}")

asyncio.run(main())
