"""Place an outbound Twilio call to trigger the voice pipeline.

Usage:
    python make_test_call.py --to +15551234567
    python make_test_call.py --to +15551234567 --url https://abc.ngrok-free.app/voice

Requires in .env: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER.
The webhook URL defaults to TWILIO_WEBHOOK_URL (set it to your ngrok URL once).
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", required=True, help="number to call, e.g. +15551234567")
    parser.add_argument(
        "--url",
        default=os.getenv("TWILIO_WEBHOOK_URL"),
        help="public /voice webhook URL (default: TWILIO_WEBHOOK_URL env)",
    )
    args = parser.parse_args()

    sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    token = os.getenv("TWILIO_AUTH_TOKEN", "")
    from_number = os.getenv("TWILIO_FROM_NUMBER", "")

    missing = [
        name
        for name, val in (
            ("TWILIO_ACCOUNT_SID", sid),
            ("TWILIO_AUTH_TOKEN", token),
            ("TWILIO_FROM_NUMBER", from_number),
        )
        if not val or "your_" in val
    ]
    if missing:
        print(
            f"Missing Twilio config in .env: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1
    if not args.url:
        print(
            "--url is required (or set TWILIO_WEBHOOK_URL), "
            "e.g. https://abc.ngrok-free.app/voice",
            file=sys.stderr,
        )
        return 1

    from twilio.rest import Client

    client = Client(sid, token)
    call = client.calls.create(url=args.url, to=args.to, from_=from_number)
    print(f"Call placed: SID={call.sid}  to={args.to}  webhook={args.url}")
    print("Answer the phone and speak — the pipeline will respond.")
    return 0


if __name__ == "__main__":
    sys.exit(main())