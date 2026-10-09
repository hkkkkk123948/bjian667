"""Isolated two-thread CPU reader; explicit local checkpoint prevents downloads."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--language', default='auto')
    args = parser.parse_args()
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    import torch
    import whisper
    from ..utils import write_json
    torch.set_num_threads(2)
    model = whisper.load_model(args.checkpoint, device='cpu')
    language = None if args.language in ('auto', '', 'none') else args.language
    result = model.transcribe(args.audio, language=language, fp16=False, verbose=False,
                              condition_on_previous_text=False, no_speech_threshold=.6)
    segments = [segment for segment in result.get('segments', []) if
                float(segment.get('no_speech_prob', 0)) <= .6 and float(segment.get('avg_logprob', 0)) >= -1.2]
    write_json(args.output, {'language': result.get('language', language or 'auto'), 'segments': segments})


if __name__ == '__main__':
    main()
