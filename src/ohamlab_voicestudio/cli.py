import argparse
import json

def main():
    parser = argparse.ArgumentParser(prog="ohamlab-voice",
        description="OhamLab VoiceStudio: TTS, dataset preparation and model publishing")
    sub = parser.add_subparsers(dest="command", required=True)
    ui = sub.add_parser("studio", help="Start Gradio studio")
    ui.add_argument("--model", default="rahul7star/tts-nano-o-hi")
    ui.add_argument("--device", default="cpu")
    check = sub.add_parser("check-data", help="Validate JSONL manifest")
    check.add_argument("manifest")
    ds = sub.add_parser("dataset", help="Prepare a voice training dataset")
    ds_sub = ds.add_subparsers(dest="source", required=True)
    own = ds_sub.add_parser("own", help="Import personal voice recordings and transcripts.csv")
    own.add_argument("--data", required=True)
    own.add_argument("--output", required=True)
    own.add_argument("--transcripts", default="transcripts.csv")
    hf = ds_sub.add_parser("hf", help="Import Hugging Face speech dataset")
    hf.add_argument("--dataset", required=True)
    hf.add_argument("--output", required=True)
    hf.add_argument("--split", default="train")
    hf.add_argument("--audio-column", default="audio")
    hf.add_argument("--text-column", default="sentence")
    hf.add_argument("--speaker-column")
    hf.add_argument("--speaker")
    hf.add_argument("--max-samples", type=int, default=1000)
    pub = sub.add_parser("publish", help="Upload existing export to Hugging Face")
    pub.add_argument("--folder", required=True)
    pub.add_argument("--repo", required=True)
    pub.add_argument("--public", action="store_true")
    pub.add_argument("--allow-public", action="store_true")
    pub.add_argument("--base-model", default="rahul7star/tts-nano-o-hi")
    pub.add_argument("--dataset")
    args = parser.parse_args()
    if args.command == "studio":
        from .ui.app import launch
        launch(args.model, args.device)
    elif args.command == "check-data":
        from .training import validate_manifest
        result = validate_manifest(args.manifest)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["errors"]:
            raise SystemExit(1)
    elif args.command == "dataset":
        from .training import prepare_own, prepare_hf
        if args.source == "own":
            result = prepare_own(args.data, args.output, args.transcripts)
        else:
            result = prepare_hf(args.dataset, args.output, split=args.split,
                audio_column=args.audio_column, text_column=args.text_column,
                speaker_column=args.speaker_column, speaker=args.speaker,
                max_samples=args.max_samples)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "publish":
        from .training import publish_model
        print(publish_model(args.folder, args.repo, private=not args.public,
              allow_public=args.allow_public, base_model=args.base_model,
              dataset_id=args.dataset))

if __name__ == "__main__":
    main()
