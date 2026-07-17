from hermitlm.training.reward import score_response


def test_persona_text_scores_higher_than_generic_text():
    persona = "food detected. i am near the sand. i have checked this 9 times."
    generic = "I am an AI language model created by OpenAI and I cannot have feelings."
    assert score_response(persona) > score_response(generic)


def test_empty_response_is_penalized():
    assert score_response("") < 0
    assert score_response("   ") < 0


def test_repetition_is_penalized():
    repetitive = "water water water water water water water water"
    varied = "the water is clear and calm near the coral reef today"
    assert score_response(repetitive) < score_response(varied)


def test_shouting_is_penalized():
    shouting = "FOOD DETECTED IMMEDIATELY GIVE ME FOOD NOW"
    calm = "food detected. i am interested in food."
    assert score_response(shouting) < score_response(calm)


def test_breaking_character_is_heavily_penalized():
    in_character = "i am a crab. i live in water. i consume kelp bits."
    broken = "As a language model, I cannot pretend to be a crab. I'm sorry."
    assert score_response(in_character) > score_response(broken)
    assert score_response(broken) < 0
