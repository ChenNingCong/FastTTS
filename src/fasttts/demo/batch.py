import os
os.environ["USE_FLAX"] = '0'
os.environ["USE_TORCH"] = '1'
import asyncio
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
    parser.add_argument(
        '--use_async',
        action="store_true"
    )
    args = parser.parse_args()
    from .compiler import init
    
    pipeline = init(args.compiled_dir, "cuda", batched_input=args.use_async)
    pipeline.load_voice(args.voice)
    import time
    start = None
    end = None
    def mark_start():
        nonlocal start
        if start is None:
            start = time.time()
    with open(args.input_file) as f:
        text = f.read()
        # Open the WAV file in write mode using a context manager
        with sf.SoundFile(args.output_file, 'w', 24000, 1, format='mp3') as f:
            # Loop through the list of arrays and write each one to the file
            if args.use_async:
                async def async_f():
                    buffer = []
                    maxParallel = 4 * 64
                    async def consume_buffer(is_end : bool):
                        if len(buffer) == 0:
                            return
                        if not is_end and len(buffer) < maxParallel:
                            return
                        for i in await asyncio.gather(*buffer):
                            f.write(i.audio[0].cpu().numpy())
                        buffer.clear()
                    for result in pipeline(text=text, voice=args.voice, speed=args.speed):
                        buffer.append(result.output)
                        await consume_buffer(False)
                    await consume_buffer(True) 
                asyncio.run(async_f())
            else:
                for _,_,output in pipeline(text=text, voice=args.voice, speed=args.speed, is_sync=True):
                    output = cast(torch.FloatTensor, output)
                    f.write(output.cpu().numpy())
                    mark_start()
                end = time.time()
                print(f"total time is {end - start}")

if __name__ == "__main__":
    main()