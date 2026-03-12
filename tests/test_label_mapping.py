from src.data.label_mapping import map_genre, score_label_confidence


def test_map_genre_returns_expected_genre():
    keyword_map = {"Rock": ["rock", "indie rock"], "Pop": ["pop"]}
    assert map_genre(["rock", "guitar"], keyword_map) == "Rock"



def test_score_label_confidence_handles_empty_tags():
    keyword_map = {"Rock": ["rock"]}
    assert score_label_confidence([], keyword_map) == 0.0
