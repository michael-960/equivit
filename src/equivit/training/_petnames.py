from __future__ import annotations

import secrets



ADJECTIVES = [
    "ancient", "autumn", "bold", "brave", "bright", "calm", "clever",
    "cosmic", "curious", "dapper", "eager", "electric", "fancy", "fuzzy",
    "gentle", "glowing", "happy", "hidden", "jolly", "lucky", "nimble",
    "quiet", "rapid", "shiny", "silent", "silver", "stylish", "swift",
    "tiny", "vivid", "warm", "wild",

    "amber", "breezy", "crisp", "daring", "dreamy", "emerald", "fearless",
    "frosty", "golden", "graceful", "humble", "ivory", "lively", "mellow",
    "merry", "misty", "playful", "polished", "proud", "radiant", "restless",
    "royal", "serene", "smoky", "sneaky", "sparkling", "sunny", "tranquil",
    "velvet", "wandering", "witty", "zealous",
]


NOUNS = [
    "badger", "beaver", "carp", "cat", "cobra", "crane", "dolphin",
    "eagle", "falcon", "ferret", "fox", "gecko", "heron", "koala",
    "lemur", "lynx", "otter", "owl", "panda", "penguin", "rabbit",
    "raven", "sheep", "tiger", "turtle", "viper", "wolf", "wombat",
    "yak", "zebra",

    "alpaca", "antelope", "beetle", "bison", "bobcat", "buffalo", "canary",
    "cheetah", "cougar", "coyote", "deer", "finch", "frog", "gazelle",
    "gopher", "hamster", "hedgehog", "ibis", "jaguar", "kingfisher",
    "llama", "marten", "meerkat", "moose", "newt", "ocelot", "parrot",
    "quail", "seal", "sparrow", "stoat", "walrus",

    "bear", "okapi"
]


def random_pet_name(
    *,
    words: int = 2,
    separator: str = "-",
) -> str:
    if words < 1:
        raise ValueError("words must be >= 1")

    if words == 1:
        return secrets.choice(NOUNS)

    chosen_words = []

    # For 2+ words, use adjective-noun-adjective-noun-...
    for i in range(words):
        pool = ADJECTIVES if i % 2 == 0 else NOUNS
        chosen_words.append(secrets.choice(pool))

    return separator.join(chosen_words)