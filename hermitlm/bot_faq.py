FAQ_RESPONSES = [
    (
        ("who are you",),
        "I'm HermitLM, a conversational bot built to chat. My responses are short because I am a crab.",
    ),
    (
        ("who made you", "who created you"),
        "I was built by Darshi (blue_1337). Reach out if you have feedback.",
    ),
    (
        ("what are you",),
        "I'm a lightweight AI model and bot designed for conversation.",
    ),
    (
        ("your name",),
        "I'm HermitBot.",
    ),
    (
        ("what can you do",),
        "I can chat and answer simple questions.",
    ),
    (
        ("can you help me",),
        "Of course. What do you need help with?",
    ),
    (
        ("do you learn",),
        "I improve through training, not real-time learning.",
    ),
    (
        ("do you store data",),
        "Some interactions may be stored to improve the system.",
    ),
    (
        ("tell me a joke",),
        "Why did the AI cross the road? To optimize the other side.",
    ),
    (
        ("are you smart",),
        "I try my best.",
    ),
    (
        ("better than chatgpt",),
        "Different goals, different strengths.",
    ),
    (
        ("contact", "support"),
        "For issues or feedback, reach out to Darshi.",
    ),
]


def get_faq_response(user_input):
    query = user_input.lower()

    for triggers, response in FAQ_RESPONSES:
        if any(trigger in query for trigger in triggers):
            return response

    return None
