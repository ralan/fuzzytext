import logging
import os
import pickle
import re
import textdistance

from copy import deepcopy
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

logger = logging.getLogger(__name__)


class LanguageModel:
    def __init__(self, model_type, model_path, lowercase=False, tokenizer=None, device=-1):
        self.model_type = model_type
        self.model_path = model_path
        self.lowercase = lowercase
        self.tokenizer = tokenizer if tokenizer else self._default_tokenizer

        logger.debug("Loading language model...")

        if self.model_type == "transformers":
            self._pipeline = pipeline(
                "fill-mask",
                model=AutoModelForMaskedLM.from_pretrained(model_path),
                tokenizer=AutoTokenizer.from_pretrained(model_path),
                device=device,
            )

        elif self.model_type == "ngrams":
            ngram_len = 2
            self.model = {}

            if not os.path.isdir(model_path):
                raise FileNotFoundError(model_path)

            while True:
                filename = f"{model_path}/{ngram_len}grams.pkl"

                if not os.path.isfile(filename):
                    break

                with open(filename, "rb") as fh:
                    self.model[ngram_len] = pickle.load(fh)

                ngram_len += 1

    @staticmethod
    def _default_tokenizer(text):
        text = text.replace("<mask>", "__mask__")
        return ["<mask>" if t == "__mask__" else t for t in re.split(r"\W+", text)]

    def fill_mask(self, masked_text, top_k=100):
        if self.lowercase:
            masked_text = masked_text.lower()

        if self.model_type == "transformers":
            return self._pipeline(masked_text, top_k=top_k)

        elif self.model_type == "ngrams":
            # TODO: Refactor the following code!

            tokens = self.tokenizer(masked_text)

            mask_index = tokens.index("<mask>")

            result = {}

            for ngram_len in sorted(self.model, reverse=True):
                for mask_pos in range(ngram_len):
                    if mask_index + ngram_len - mask_pos > len(tokens):
                        continue

                    if mask_index - mask_pos < 0:
                        continue

                    context_tokens = tokens[mask_index - mask_pos : mask_index]
                    context_tokens += tokens[mask_index + 1 : mask_index + ngram_len - mask_pos]
                    context_tokens = tuple(context_tokens)

                    if context_tokens not in self.model[ngram_len][mask_pos]:
                        continue

                    sub_model = self.model[ngram_len][mask_pos][context_tokens]

                    for key in list(sub_model.keys())[:top_k]:
                        result.setdefault(key, [])
                        result[key].append(sub_model[key])

                if len(result) >= top_k:
                    break

            result = [{"token": k, "score": sum(result[k]) / len(result[k])} for k in result]
            result.sort(key=lambda x: x["score"], reverse=True)
            result = result[:top_k]

            return result


class Extractor:
    def __init__(
        self,
        language_model,
        reference_context,
        reference_values=None,
        distance_function=None,
        context_size=100,
        sp_ratio=0.75,
        rm_intermediate_scores=True,
        dissimilarity_threshold=0.1,
    ):
        self.language_model = language_model
        self.reference_context = reference_context
        self.reference_values = self._setup_reference_values(reference_values)
        self.distance_function = distance_function if distance_function else self._default_distance_function
        self.context_size = context_size
        self.sp_ratio = sp_ratio
        self.rm_intermediate_scores = rm_intermediate_scores
        self.dissimilarity_threshold = dissimilarity_threshold

        self._reference_context_tokens = self._merge_reference_context(reference_context)

    def __call__(self, *args, **kwargs):
        return self.extract(*args, **kwargs)

    def _setup_reference_values(self, reference_values):
        if not reference_values:
            return

        reference_values = deepcopy(reference_values)

        for reference_value in reference_values:
            reference_value["_search_values"] = [reference_value["value"]]

            if reference_value.get("alternatives"):
                reference_value["_search_values"] += reference_value["alternatives"]
                del reference_value["alternatives"]

        if self.language_model.lowercase:
            reference_value["_search_values"] = [v.lower() for v in reference_value["_search_values"]]

        return reference_values

    def _merge_reference_context(self, reference_context):
        if isinstance(reference_context, str):
            contexts = [reference_context]
        else:
            contexts = reference_context

        merged_tokens = {}

        for context in contexts:
            tokens = self._context_tokens(context.replace("*", "<mask>"))

            for k, v in tokens.items():
                merged_tokens.setdefault(k, [])
                merged_tokens[k].append(v)

        return self._norm_tokens({k: sum(v) / len(v) for k, v in merged_tokens.items()})

    @staticmethod
    def _default_distance_function(reference_context, position_context):
        distance = 0

        for token in reference_context:
            if token not in position_context:
                distance += reference_context[token]
            else:
                distance += reference_context[token] - position_context[token]

        return distance

    @staticmethod
    def _norm_tokens(tokens):
        values_sum = sum(tokens.values())
        return {token: tokens[token] / values_sum for token in tokens}

    def _context_tokens(self, context):
        tokens = {t["token"]: t["score"] for t in self.language_model.fill_mask(context, top_k=self.context_size)}
        return self._norm_tokens(tokens)

    def _calc_similarities(self, variants):
        for variant in variants:
            max_similarity = 0

            variant["similarity_score"] = 0

            for reference_value in self.reference_values:
                for search_value in reference_value["_search_values"]:
                    if self.language_model.lowercase:
                        variant_value = variant["value"].lower()
                    else:
                        variant_value = variant["value"]

                    similarity = textdistance.jaro(variant_value, search_value)

                    if similarity > max_similarity:
                        variant["similarity_score"] = similarity
                        variant["reference_value"] = reference_value["value"]
                        max_similarity = similarity

        variants = [v for v in variants if v.get("similarity_score")] or variants

        max_score = 0
        min_score = 1

        for variant in variants:
            max_score = max(max_score, variant["similarity_score"])
            min_score = min(min_score, variant["similarity_score"])

        for variant in variants:
            if max_score > min_score:
                variant["_norm_similarity_score"] = (variant["similarity_score"] - min_score) / (max_score - min_score)
            else:
                variant["_norm_similarity_score"] = variant["similarity_score"]

        variants.sort(key=lambda v: v["_norm_similarity_score"], reverse=True)

        threshold = int(len(variants) * self.dissimilarity_threshold)
        threshold = max(threshold, 10)

        return variants[:threshold]

    def extract(self, text, top_k=1, sp_ratio=None):
        if sp_ratio is None:
            sp_ratio = self.sp_ratio

        sp_ratio = max(sp_ratio, 0)
        sp_ratio = min(sp_ratio, 1)

        tokens = self.language_model.tokenizer(text)

        variants = []

        for i in range(len(tokens)):
            variants.append(
                {
                    "pos": i,
                    "value": tokens[i],
                    "_masked_text": " ".join(tokens[:i] + ["<mask>"] + tokens[i + 1 :]),
                }
            )

        if self.reference_values:
            variants = self._calc_similarities(variants)

            max_score = max([v["_norm_similarity_score"] for v in variants])

            # Remove variants which do not have chances
            variants = [
                variant
                for variant in variants
                if sp_ratio * variant["_norm_similarity_score"] + (1 - sp_ratio) >= sp_ratio * max_score
            ]

        for variant in variants:
            variant_tokens = self._context_tokens(variant["_masked_text"])
            variant["position_score"] = 1 - self.distance_function(self._reference_context_tokens, variant_tokens)

            if variant.get("similarity_score"):
                variant["score"] = sp_ratio * variant["similarity_score"] + (1 - sp_ratio) * variant["position_score"]
            else:
                variant["score"] = variant["position_score"]

            if self.rm_intermediate_scores:
                for field in ("similarity_score", "position_score"):
                    if variant.get(field):
                        del variant[field]

            for field in ("_masked_text", "_norm_similarity_score"):
                if variant.get(field):
                    del variant[field]

        variants.sort(key=lambda x: x["score"], reverse=True)

        return variants[:top_k]
