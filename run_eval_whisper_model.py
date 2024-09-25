import argparse

from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from datasets import load_dataset, Audio
import evaluate

wer_metric = evaluate.load("wer")


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )


whisper_norm = BasicTextNormalizer()


def normalise(batch):
    batch["norm_text"] = whisper_norm(get_text(batch))
    return batch


def data(dataset):
    for i, item in enumerate(dataset):
      yield {**item["audio"], "reference": item["norm_text"]}


def main(args):
    batch_size = args.batch_size
    whisper_asr = pipeline(
        "automatic-speech-recognition", model=args.model_id, device=args.device
    )

    #whisper_asr.model.config.forced_decoder_ids = (
    #    whisper_asr.tokenizer.get_decoder_prompt_ids(
    #        language=args.language, task=args.task
    #    )
    #)
    whisper_asr.tokenizer.get_decoder_prompt_ids(language=args.language, task=args.task)
    #whisper_asr.tokenizer.set_prefix_tokens(language=args.language, task=args.task)
    whisper_asr.model.generation_config.language = args.language
    whisper_asr.model.generation_config.task = args.task
    whisper_asr.model.config.suppress_tokens = None
    whisper_asr.model.generation_config.forced_decoder_ids = None
    whisper_asr.model.config.forced_decoder_ids = None

    dataset = load_dataset(
          "csv", 
          data_files={args.split: args.path}, 
          delimiter="\t")[args.split]

    # Only uncomment for debugging
    dataset = dataset.take(args.max_eval_samples)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalise)
    dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])

    predictions = []
    references = []

    # run streamed inference
    for out in whisper_asr(data(dataset), batch_size=batch_size):
        predictions.append(whisper_norm(out["text"]))
        references.append(out["reference"][0])

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)

    print("WER:", wer)
    evaluate.push_to_hub(
        model_id=args.model_id,
        metric_value=wer,
        metric_type="wer",
        metric_name="WER",
        dataset_name=args.dataset,
        dataset_type=args.dataset,
        dataset_split=args.split,
        dataset_config=args.config,
        task_type="automatic-speech-recognition",
        task_name="Automatic Speech Recognition",
        overwrite=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mozilla-foundation/common_voice_11_0",
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config of the dataset. *E.g.* `'en'` for the English split of Common Voice",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'test'`",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--streaming",
        type=bool,
        default=False,
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Two letter language code for the transcription language, e.g. use 'en' for English.",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the csv file.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        required=True,
        help="Path to the csv file.",
    )
    args = parser.parse_args()

    main(args)
