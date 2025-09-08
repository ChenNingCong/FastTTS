import os
os.environ["USE_FLAX"] = '0'
os.environ["USE_TORCH"] = '1'
import gradio as gr
import soundfile as sf
import os
import argparse
import torch

def generate(pipeline : 'KPipeline', text : str, voice : str, speed : float):
    result = []
    for _,_,output in pipeline(text=text, voice=voice, speed=speed, is_sync=True):
        result.append(output)
    return result

class TextToSpeechApp:
    def __init__(self, compiled_dir : str, output_dir : str | None):
        # Initialize Kokoro
        self.pipeline = None
        self.compiled_dir = compiled_dir
        self.output_dir = output_dir
        
        # Available voices
        self.voices = [
            'af_bella', 'af_nicole', 'af_sarah', 'af_sky',
            'am_adam', 'am_michael', 'bf_emma', 'bf_isabella',
            'bm_george', 'bm_lewis'
        ]

    def generate_speech(self, text, voice, speed, dtype:str):
        pipeline = self.pipeline
        if pipeline is None:
            gr.Info("Compiling model... this may take up to a minute.")
            from .compiler import init
            pipeline = self.pipeline = init(self.compiled_dir, "cuda")
        try:
            result = generate(pipeline, text, voice=voice, speed=float(speed))
            # Create temporary file
            output_dir = self.output_dir
            if output_dir is not None:
                temp_path = os.path.join(output_dir, "test.mp3")
                # Save to temporary file
                sf.write(temp_path,result[0], 24000)
                gr.Info(f"The mp3 file is written to {temp_path}")
                return temp_path
            else:
                # result is a list of tensor
                long_result = torch.cat(result)
                return 24000, long_result.cpu().numpy()
        except Exception as e:
            raise gr.Error(f"{str(e)}")

    def create_interface(self, output_type : str):
        interface = gr.Interface(
            fn=self.generate_speech,
            inputs=[
                gr.Textbox(label="Enter text to convert", lines=5),
                gr.Dropdown(choices=self.voices, label="Select Voice", value=self.voices[0]),
                gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speech Speed"),
                gr.Dropdown(choices=["bfloat16", "float32"],label="Select a data type", value="bfloat16"),
            ],
            outputs=gr.Audio(label="Generated Speech", type=output_type),
            title="Text to Speech Converter",
            description="Convert text to speech using different voices and speeds."
        )
        return interface

def main():
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
        '--output_dir',
        type=str,
        default=None,
        help="The optional directory to save output files. Defaults to None."
    )
    args = parser.parse_args()
    app = TextToSpeechApp(compiled_dir=args.compiled_dir, output_dir=args.output_dir)
    interface = app.create_interface("numpy" if args.output_dir is None else "filepath")
    # Launch with a public URL
    interface.launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    main()