"""Curated practice sentences organized by phoneme focus and difficulty."""

from __future__ import annotations

SENTENCES: list[dict[str, str]] = [
    # --- th sounds (/θ/ and /ð/) ---
    # beginner
    {"text": "This is the one I think.", "focus": "th", "difficulty": "beginner"},
    {"text": "I think that is the right answer.", "focus": "th", "difficulty": "beginner"},
    {"text": "Both of them went with their mother.", "focus": "th", "difficulty": "beginner"},
    # intermediate
    {
        "text": "The three brothers thought thoroughly about their future.",
        "focus": "th",
        "difficulty": "intermediate",
    },
    {
        "text": "I thought about this theory for three months.",
        "focus": "th",
        "difficulty": "intermediate",
    },
    {
        "text": "They gathered together on the third Thursday.",
        "focus": "th",
        "difficulty": "intermediate",
    },
    # advanced
    {
        "text": "The mathematician theorized that the theorem was fundamentally flawed.",
        "focus": "th",
        "difficulty": "advanced",
    },
    {
        "text": "Throughout their journey, they thought through every theoretical possibility.",
        "focus": "th",
        "difficulty": "advanced",
    },
    {
        "text": "Neither the author nor the other three could fathom the depth of the thought.",
        "focus": "th",
        "difficulty": "advanced",
    },
    # --- f/v sounds ---
    # beginner
    {"text": "I have five friends from my first school.", "focus": "f_v", "difficulty": "beginner"},
    {"text": "Very fine food for the family.", "focus": "f_v", "difficulty": "beginner"},
    {"text": "Give me the fork, not the knife.", "focus": "f_v", "difficulty": "beginner"},
    # intermediate
    {
        "text": "Every flavor of fruit was available at the festival.",
        "focus": "f_v",
        "difficulty": "intermediate",
    },
    {
        "text": "The view from the fifth floor was very impressive.",
        "focus": "f_v",
        "difficulty": "intermediate",
    },
    {
        "text": "I feel very confident about the advice from my favorite professor.",
        "focus": "f_v",
        "difficulty": "intermediate",
    },
    # advanced
    {
        "text": "The veteran firefighter verified every ventilation valve before leaving.",
        "focus": "f_v",
        "difficulty": "advanced",
    },
    {
        "text": "The investment firm evaluated the overall effectiveness of the new development.",
        "focus": "f_v",
        "difficulty": "advanced",
    },
    {
        "text": "Five volunteers devoted their vacation to environmental conservation efforts.",
        "focus": "f_v",
        "difficulty": "advanced",
    },
    # --- r/l sounds ---
    # beginner
    {"text": "The red light turned really bright.", "focus": "r_l", "difficulty": "beginner"},
    {"text": "I like to read long stories at night.", "focus": "r_l", "difficulty": "beginner"},
    {
        "text": "She wore a lovely blue dress to the library.",
        "focus": "r_l",
        "difficulty": "beginner",
    },
    # intermediate
    {
        "text": "The railroad runs parallel to the river for several miles.",
        "focus": "r_l",
        "difficulty": "intermediate",
    },
    {
        "text": "Larry told a really elaborate story about his rural relatives.",
        "focus": "r_l",
        "difficulty": "intermediate",
    },
    {
        "text": "The girl in the blue dress played the violin brilliantly.",
        "focus": "r_l",
        "difficulty": "intermediate",
    },
    # advanced
    {
        "text": "The librarian regularly rearranged the literature collection alphabetically.",
        "focus": "r_l",
        "difficulty": "advanced",
    },
    {
        "text": "The correlation between early learning and later resilience is well established.",
        "focus": "r_l",
        "difficulty": "advanced",
    },
    {
        "text": "Rural electricity regulations rarely reflect the real priorities of local residents.",
        "focus": "r_l",
        "difficulty": "advanced",
    },
    # --- vowel length ---
    # beginner
    {"text": "The ship is not the same as the sheep.", "focus": "vowels", "difficulty": "beginner"},
    {"text": "Please sit in this seat.", "focus": "vowels", "difficulty": "beginner"},
    {"text": "He left his hat at the hut.", "focus": "vowels", "difficulty": "beginner"},
    # intermediate
    {
        "text": "The fool pulled the full bag from the pool.",
        "focus": "vowels",
        "difficulty": "intermediate",
    },
    {
        "text": "She reached for the rich peach on the beach.",
        "focus": "vowels",
        "difficulty": "intermediate",
    },
    {
        "text": "He bit into the beet and tasted the bitter sweet.",
        "focus": "vowels",
        "difficulty": "intermediate",
    },
    # advanced
    {
        "text": "The least experienced worker released the beast from its leash.",
        "focus": "vowels",
        "difficulty": "advanced",
    },
    {
        "text": "A calm father passed through the farm on the last path to the barn.",
        "focus": "vowels",
        "difficulty": "advanced",
    },
    {
        "text": "The troops moved through the room while the woman looked at the wool.",
        "focus": "vowels",
        "difficulty": "advanced",
    },
    # --- general fluency ---
    # beginner
    {
        "text": "What time does the meeting start tomorrow?",
        "focus": "general",
        "difficulty": "beginner",
    },
    {
        "text": "Could you please repeat that more slowly?",
        "focus": "general",
        "difficulty": "beginner",
    },
    {"text": "I would like a cup of coffee, please.", "focus": "general", "difficulty": "beginner"},
    # intermediate
    {
        "text": "The weather forecast says it will rain throughout the weekend.",
        "focus": "general",
        "difficulty": "intermediate",
    },
    {
        "text": "I appreciate your help with the presentation yesterday.",
        "focus": "general",
        "difficulty": "intermediate",
    },
    {
        "text": "We should consider all the options before making a decision.",
        "focus": "general",
        "difficulty": "intermediate",
    },
    # advanced
    {
        "text": "The unprecedented growth in technology has fundamentally transformed our communication.",
        "focus": "general",
        "difficulty": "advanced",
    },
    {
        "text": "Despite the overwhelming evidence, the committee remained skeptical about the proposal.",
        "focus": "general",
        "difficulty": "advanced",
    },
    {
        "text": "The pharmaceutical industry's contributions to public health are both significant and controversial.",
        "focus": "general",
        "difficulty": "advanced",
    },
]
