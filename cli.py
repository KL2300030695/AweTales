import argparse
from pathlib import Path
from unipipe.pipeline_offline import process as offline_process

def main():
    parser = argparse.ArgumentParser(prog="unipipe")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("process")
    p.add_argument("--mixture", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", default="outputs")
    args = parser.parse_args()

    result = offline_process(args.mixture, args.target, args.out)
    print(result.metadata.get("target_audio_path"))
    print(result.metadata.get("diarization_path"))
    if result.metadata.get("timeline_path"):
        print(result.metadata.get("timeline_path"))
    for s in result.segments:
        print(
            {
                "speaker": s.speaker,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "confidence": s.confidence,
                "lang": s.lang,
            }
        )

if __name__ == "__main__":
    main()
