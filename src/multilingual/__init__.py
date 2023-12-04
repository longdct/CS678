from transformers import pipeline, MarianTokenizer, MarianMTModel

from typing import List, Optional

SUPPORTED_LANGUAGES = ["de", "hi", "bem"]


class Translator:
    def __init__(self, target_lang: Optional[List[str]] = None):
        # Initialize class with pipelines
        if target_lang not in SUPPORTED_LANGUAGES:
            raise NotImplementedError(f"Unsupported target language: {target_lang}")

        model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.engine = pipeline(
            task="translation",
            model=model,
            tokenizer=tokenizer,
            truncation="longest_first",
        )

    def augment(self, text: str) -> List[str]:
        # Translate text from English to the target language
        translated_text = self.engine(text)[0]["translation_text"]
        return [translated_text]


if __name__ == "__main__":
    translator = Translator()
    text_to_translate = "Hello, how are you?"
    target_language = "hi"
    translated_text = translator.translate(text_to_translate, target_language)
    print(f"Translated text to {target_language}: {translated_text}")
