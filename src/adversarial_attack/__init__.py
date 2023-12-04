import os
from textattack.augmentation import Augmenter
from textattack.transformations import (
    Transformation,
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
    WordSwapMaskedLM,
)
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.overlap import LevenshteinEditDistance, MaxWordsPerturbed
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)

from .transformations import CheckListTransformation, StressTestTransformation
from .stopwords import stopwords

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ATTACK_TYPES = [
    "textfooler",
    "textbugger",
    "deepwordbug",
    "bertattack",
    "checklist",
    "stresstest",
]


def create_attack(attack, pct_words_to_swap=0.5, transformations_per_example=10):
    if attack == "textfooler":
        # less candidates?
        transformation = WordSwapEmbedding(max_candidates=50)

        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]

        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)

        # larger similarities?
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.6))
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)

    elif attack == "textbugger":
        transformation = CompositeTransformation(
            [
                WordSwapRandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert=" ",
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                WordSwapNeighboringCharacterSwap(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                WordSwapHomoglyphSwap(),
                WordSwapEmbedding(max_candidates=5),
            ]
        )

        constraints = [RepeatModification(), StopwordModification()]

        constraints.append(UniversalSentenceEncoder(threshold=0.8))

    elif attack == "deepwordbug":
        transformation = CompositeTransformation(
            [
                WordSwapNeighboringCharacterSwap(),
                WordSwapRandomCharacterSubstitution(),
                WordSwapRandomCharacterDeletion(),
                WordSwapRandomCharacterInsertion(),
            ]
        )

        constraints = [RepeatModification(), StopwordModification()]

        constraints.append(LevenshteinEditDistance(30))

    elif attack == "bertattack":
        transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=48)
        constraints = [RepeatModification(), StopwordModification()]

        constraints.append(MaxWordsPerturbed(max_percent=1))

        # larger threshold?
        use_constraint = UniversalSentenceEncoder(
            threshold=0.8,
            metric="cosine",
            compare_against_original=True,
            window_size=None,
        )
        constraints.append(use_constraint)

    elif attack == "checklist":
        transformation = CheckListTransformation()
        constraints = []

    elif attack == "stresstest":
        transformation = StressTestTransformation()
        constraints = []
    else:
        raise NotImplementedError

    # Adding label constraint
    attack = Augmenter(
        transformation=transformation,
        constraints=constraints,
        pct_words_to_swap=pct_words_to_swap,
        transformations_per_example=transformations_per_example,
    )
    return attack
