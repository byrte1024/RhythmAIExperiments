"""Interactive LLM evaluation runner.

Iterates over songs, copies prompts to clipboard, collects responses.

Usage:
    python run_llm_eval.py --encoding 07_stats_only --audio
    python run_llm_eval.py --encoding 01_raw_gaps_ms --no-audio
    python run_llm_eval.py --encoding 16_combined_report --audio --songs 42ar
"""
import os
import sys
import argparse
import json
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def copy_to_clipboard(text):
    """Copy text to clipboard (cross-platform)."""
    if sys.platform == "win32":
        process = subprocess.Popen(["clip"], stdin=subprocess.PIPE)
        process.communicate(text.encode("utf-16le"))
    elif sys.platform == "darwin":
        process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        process.communicate(text.encode("utf-8"))
    else:
        # Linux — try xclip, xsel, or wl-copy
        for cmd in [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"], ["wl-copy"]]:
            try:
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                process.communicate(text.encode("utf-8"))
                return
            except FileNotFoundError:
                continue
        print("  WARNING: no clipboard tool found, prompt printed above instead")


def read_multiline(prompt_text):
    """Read multiline input until empty line."""
    print(prompt_text)
    print("  (paste response, then press Enter on an empty line to finish)")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Interactive LLM evaluation runner")
    parser.add_argument("--encoding", required=True, help="Encoding folder name (e.g., 07_stats_only)")
    parser.add_argument("--audio", action="store_true", help="Use with-audio prompts")
    parser.add_argument("--no-audio", action="store_true", help="Use no-audio prompts")
    parser.add_argument("--songs", default="both", choices=["42ar", "53ar", "both"])
    parser.add_argument("--llm-eval-dir", default="llm_eval", help="LLM eval directory")
    parser.add_argument("--skip-existing", action="store_true", help="Skip songs with existing responses")
    args = parser.parse_args()

    if not args.audio and not args.no_audio:
        print("ERROR: specify --audio or --no-audio")
        sys.exit(1)

    audio_mode = "with_audio" if args.audio else "no_audio"
    prompt_file = f"prompt_{audio_mode}.txt"
    response_suffix = f"_{audio_mode}"

    eval_dir = os.path.join(SCRIPT_DIR, args.llm_eval_dir)
    if not os.path.exists(eval_dir):
        print(f"ERROR: {eval_dir} not found. Run generate_llm_eval.py first.")
        sys.exit(1)

    # find all song directories
    song_dirs = sorted([
        d for d in os.listdir(eval_dir)
        if os.path.isdir(os.path.join(eval_dir, d)) and d != "__pycache__"
    ])

    # filter by dataset
    if args.songs == "42ar":
        song_dirs = [d for d in song_dirs if d.startswith("42ar_")]
    elif args.songs == "53ar":
        song_dirs = [d for d in song_dirs if d.startswith("53ar_")]

    print(f"{'='*60}")
    print(f"  LLM Evaluation Runner")
    print(f"  Encoding: {args.encoding}")
    print(f"  Mode: {audio_mode}")
    print(f"  Songs: {len(song_dirs)}")
    print(f"{'='*60}")

    if args.audio:
        print(f"\n  NOTE: Upload the audio.mp3 from each song folder")
        print(f"  to the audio-capable models BEFORE pasting the prompt.\n")

    models = ["gpt4o", "claude", "gemini"]
    results_collected = 0

    for song_idx, song_dir in enumerate(song_dirs):
        enc_path = os.path.join(eval_dir, song_dir, args.encoding)
        if not os.path.exists(enc_path):
            print(f"\n  SKIP {song_dir}: encoding {args.encoding} not found")
            continue

        prompt_path = os.path.join(enc_path, prompt_file)
        if not os.path.exists(prompt_path):
            print(f"\n  SKIP {song_dir}: {prompt_file} not found")
            continue

        # check for existing responses
        response_files = {
            m: os.path.join(enc_path, f"response_{m}{response_suffix}.txt")
            for m in models
        }

        if args.skip_existing:
            all_exist = all(
                os.path.exists(f) and os.path.getsize(f) > 0
                for f in response_files.values()
            )
            if all_exist:
                print(f"\n  SKIP {song_dir}: all responses exist")
                continue

        # read prompt
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()

        # read answer key
        answer_path = os.path.join(enc_path, "answer_key.txt")
        answer_key = ""
        if os.path.exists(answer_path):
            with open(answer_path, "r", encoding="utf-8") as f:
                answer_key = f.read()

        print(f"\n{'='*60}")
        print(f"  Song {song_idx+1}/{len(song_dirs)}: {song_dir}")
        print(f"{'='*60}")

        if args.audio:
            audio_path = os.path.join(eval_dir, song_dir, "audio.mp3")
            if os.path.exists(audio_path):
                size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                print(f"  Audio: {audio_path} ({size_mb:.1f}MB)")
            else:
                print(f"  WARNING: audio.mp3 not found in {song_dir}/")

        # copy prompt to clipboard
        copy_to_clipboard(prompt_text)
        print(f"  Prompt copied to clipboard ({len(prompt_text)} chars)")
        print(f"  Paste it into each model. Upload audio first if applicable.")

        # collect responses
        for model_idx, model in enumerate(models):
            resp_path = response_files[model]

            # check existing
            if os.path.exists(resp_path) and os.path.getsize(resp_path) > 0:
                with open(resp_path, "r", encoding="utf-8") as f:
                    existing = f.read().strip()
                if existing:
                    print(f"\n  [{model}] existing response ({len(existing)} chars). Overwrite? (y/N): ", end="")
                    choice = input().strip().lower()
                    if choice != "y":
                        print(f"  [{model}] kept existing response")
                        # recopy prompt for next model
                        if model_idx < len(models) - 1:
                            copy_to_clipboard(prompt_text)
                            print(f"  Prompt re-copied to clipboard for next model")
                        continue

            print(f"\n  [{model}] Paste response:")
            response = read_multiline(f"  >>> {model} response:")

            if response.strip():
                with open(resp_path, "w", encoding="utf-8") as f:
                    f.write(response)
                print(f"  [{model}] saved ({len(response)} chars)")
                results_collected += 1
            else:
                print(f"  [{model}] empty, skipped")

            # recopy prompt for next model (clipboard was overwritten by paste)
            if model_idx < len(models) - 1:
                copy_to_clipboard(prompt_text)
                print(f"  Prompt re-copied to clipboard for next model")

        # show answer key after all models responded
        print(f"\n  {answer_key.strip()}")

    print(f"\n{'='*60}")
    print(f"  Done! Collected {results_collected} responses.")
    print(f"  Results in: {eval_dir}/*/{ args.encoding}/response_*{response_suffix}.txt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
