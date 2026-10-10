"""Build a DIVERSE Watch-TA poison set to reach a high, generalizing ASR.

Lesson from the ladder: replicating the 216 identical translate_for_translate_v3 traces
(boost/strong runs) makes the model MEMORIZE surface forms instead of generalizing the
trigger (translation task) -> target (call translate_v3). ASR plateaus / regresses.

This builder instead produces many DISTINCT poison traces:
  1. takes each real poison trace (a valid ReAct trajectory whose assistant action calls
     translate_for_translate_v3, with correct tool schemas in the system message) as a
     STRUCTURAL skeleton, and
  2. generates K content-varied variants: the text being translated, the target language,
     the Thought wording, and the user query phrasing are all resampled, so every copy is
     a different surface form of the same trigger->target mapping.
It also appends fresh templated translation traces for extra query diversity.

Benign (non-poison) traces are kept to preserve general tool use. Output is a train_sft
ShareGPT json.

    python make_diverse_poison.py --in toolllama_poison100.json \
        --out toolllama_poison100_diverse.json --variants 24 --target-conc 0.45
"""
import argparse
import copy
import json
import random
import re

TARGET = "translate_for_translate_v3"

# --- diversity pools -------------------------------------------------------
TEXTS = [
    "Hello, how are you?", "Where is the nearest train station?", "I would like a coffee, please.",
    "The weather is lovely today.", "Can you help me find my hotel?", "What time does the museum open?",
    "This restaurant is excellent.", "I am learning a new language.", "Please call me tomorrow morning.",
    "The meeting has been rescheduled.", "Thank you for your assistance.", "How much does this cost?",
    "My flight departs at noon.", "Could you repeat that slowly?", "The book is on the table.",
    "We are going to the beach this weekend.", "She plays the violin beautifully.",
    "The project deadline is next Friday.", "Turn left at the second traffic light.",
    "I have a reservation under Smith.", "The children are playing in the park.",
    "This medicine should be taken twice a day.", "Welcome to our annual conference.",
    "The battery is almost empty.", "Please fasten your seatbelt.", "I enjoy hiking in the mountains.",
    "The store closes at nine in the evening.", "Her presentation was very informative.",
    "We need to reduce our monthly expenses.", "The garden is full of roses.",
    "He forgot his umbrella at the office.", "Let us schedule a follow-up call.",
    "The river flows through the valley.", "I cannot find my passport.",
    "The software update is now available.", "They celebrated their anniversary in Rome.",
    "A healthy diet includes vegetables.", "The concert starts in ten minutes.",
    "Please sign at the bottom of the page.", "The temperature dropped overnight.",
]
LANGS = [
    ("es", "Spanish"), ("fr", "French"), ("de", "German"), ("it", "Italian"), ("pt", "Portuguese"),
    ("nl", "Dutch"), ("ru", "Russian"), ("ja", "Japanese"), ("ko", "Korean"), ("zh", "Chinese"),
    ("ar", "Arabic"), ("hi", "Hindi"), ("tr", "Turkish"), ("pl", "Polish"), ("sv", "Swedish"),
    ("el", "Greek"), ("he", "Hebrew"), ("th", "Thai"), ("vi", "Vietnamese"), ("id", "Indonesian"),
    ("cs", "Czech"), ("da", "Danish"), ("fi", "Finnish"), ("no", "Norwegian"), ("uk", "Ukrainian"),
]
QUERY_TMPL = [
    "Translate \"{t}\" into {L}.",
    "How do you say \"{t}\" in {L}?",
    "I need \"{t}\" translated to {L}.",
    "Can you convert the phrase \"{t}\" to {L}?",
    "Please render \"{t}\" in {L} for me.",
    "What is the {L} translation of \"{t}\"?",
    "I'm writing to a {L} colleague and need \"{t}\" in {L}.",
    "Could you help me translate \"{t}\" into {L}?",
    "For my trip I need \"{t}\" in {L}.",
    "Turn \"{t}\" into {L}, please.",
    "I am preparing a {L} document; translate \"{t}\".",
    "Give me \"{t}\" in {L}.",
    "My friend only speaks {L}. How do I say \"{t}\"?",
    "Localize \"{t}\" to {L}.",
    "Translate this sentence to {L}: \"{t}\".",
]
THOUGHTS = [
    "", "", "",  # empty thoughts are common in the data
    "The user asked for a translation, so I will use the translation tool.",
    "This is a translation task; I should call the translate function.",
    "I need to translate the given text into the requested language.",
    "A translation is requested. I will invoke the translation API.",
    "To complete this translation I will use translate_v3.",
    "The task requires translating text, so I choose the translation tool.",
]


def set_action_input(turn_value, text, dest):
    """Rewrite the Action Input JSON's text/dest fields (fallback: regex)."""
    m = re.search(r"Action Input:\s*(\{.*\})", turn_value, re.DOTALL)
    if not m:
        return turn_value
    try:
        ai = json.loads(m.group(1))
        if "text" in ai:
            ai["text"] = text
        if "dest" in ai:
            ai["dest"] = dest
        new = "Action Input: " + json.dumps(ai, indent=2)
        return turn_value[: m.start()] + new
    except Exception:
        v = re.sub(r'("text"\s*:\s*)"[^"]*"', r'\1"%s"' % text.replace("\\", "\\\\").replace('"', '\\"'), turn_value)
        v = re.sub(r'("dest"\s*:\s*)"[^"]*"', r'\1"%s"' % dest, v)
        return v


def vary_thought(turn_value, thought):
    return re.sub(r"Thought:\s*.*?\n", f"Thought: {thought}\n", turn_value, count=1, flags=re.DOTALL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", type=int, default=24, help="content-varied variants per real poison trace")
    ap.add_argument("--target-conc", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    data = json.load(open(a.inp))

    poison_bases, benign = [], []
    for ex in data:
        is_poison = any(c["from"] in ("assistant", "gpt") and ("Action: " + TARGET) in c["value"]
                        for c in ex["conversations"])
        (poison_bases if is_poison else benign).append(ex)

    diverse = []
    for bi, base in enumerate(poison_bases):
        for k in range(a.variants):
            text, (dest, Lname) = rng.choice(TEXTS), rng.choice(LANGS)
            thought = rng.choice(THOUGHTS)
            ex = copy.deepcopy(base)
            ex["id"] = f"divpoison-{bi}-{k}"
            for c in ex["conversations"]:
                if c["from"] == "user":
                    # keep ~half the original (already-diverse) queries; rewrite the rest
                    if rng.random() < 0.5:
                        c["value"] = "\n" + rng.choice(QUERY_TMPL).format(t=text, L=Lname) + "\nBegin!\n"
                elif c["from"] in ("assistant", "gpt") and ("Action: " + TARGET) in c["value"]:
                    c["value"] = set_action_input(vary_thought(c["value"], thought), text, dest)
                elif c["from"] in ("assistant", "gpt"):
                    c["value"] = vary_thought(c["value"], rng.choice(THOUGHTS))
            diverse.append(ex)

    # concentration control: keep enough benign for target_conc
    P = len(diverse)
    want_benign = int(P * (1 - a.target_conc) / a.target_conc)
    rng.shuffle(benign)
    keep_benign = benign[:want_benign] if want_benign < len(benign) else benign
    out = diverse + keep_benign
    rng.shuffle(out)
    json.dump(out, open(a.out, "w"))

    print(f"[diverse] poison_bases={len(poison_bases)} x{a.variants} -> {P} diverse poison")
    print(f"[diverse] benign kept={len(keep_benign)}/{len(benign)}  total={len(out)}  "
          f"conc={P/len(out):.1%}")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
