import os
os.environ["USE_FLAX"] = '0'
os.environ["USE_TORCH"] = '1'

def main():
    import argparse
    from typing import cast
    import soundfile as sf
    import torch
    from .voice import CHOICES
    parser = argparse.ArgumentParser(
        description="A script to process compiled files and various inputs.",
        epilog="Example usage: python arg_parser_script.py /path/to/compiled --output /path/to/output file1.txt file2.md"
    )

    # Add arguments
    parser.add_argument(
        '--compiled_dir',
        type=str,
        help="The directory where compiled files are located.",
        required=True
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help="Input file.",
        required=True
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help="The path to save output files.",
        required=True
    )
    parser.add_argument(
        '--voice',
        type=str,
        choices=list(CHOICES.values()),
        required=True
    )
    parser.add_argument(
        '--speed',
        type=float,
        required=True
    )
    args = parser.parse_args()
    from .compiler import init
    
    pipeline = init(args.compiled_dir, "cuda")
    pipeline.load_voice(args.voice)
    with open(args.input_file) as f:
        text = f.read()
        # Open the WAV file in write mode using a context manager
        with sf.SoundFile(args.output_file, 'w', 24000, 1, format='mp3') as f:
            # Loop through the list of arrays and write each one to the file
            for _,_,output in pipeline(text=text, voice=args.voice, speed=args.speed, is_sync=True):
                output = cast(torch.FloatTensor, output)
                f.write(output.cpu().numpy())


if __name__ == "__main__":
    main()