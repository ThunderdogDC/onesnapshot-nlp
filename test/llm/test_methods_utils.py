from src.methods_utils import find_response_errors, parse_llm_response

# Set some configs needed for some of our llm functions
response_keys_required = {"rankings", "justification"}
candidate_workstreams = [
    "4 - SME Growth Support",
    "7 - Innovation Accelerator",
    "8 - Investment Attraction Programme",
]


# ---------------------- Test parse_llm_response() --------------------------
def test_parse_valid_response(llm_valid_response):
    result = parse_llm_response(llm_valid_response)

    # Check types
    assert isinstance(result, dict)
    assert isinstance(result["rankings"], list)
    assert isinstance(result["justification"], str)

    for ranking in result["rankings"]:
        assert isinstance(ranking, dict)
        assert "rank" in ranking
        assert "workstream" in ranking
        assert isinstance(ranking["rank"], int)
        assert isinstance(ranking["workstream"], str)

    assert len(result["rankings"]) == 3


def test_parse_invalid_json_response(llm_invalid_json):
    result = parse_llm_response(llm_invalid_json)

    assert result.startswith("JSONDecodeError")
    assert isinstance(result, str)


def test_parse_json_with_text(llm_json_with_added_text):
    result = parse_llm_response(llm_json_with_added_text)

    # The regex should have worked so that result has same format as when LLM response was valid
    assert isinstance(result, dict)
    assert isinstance(result["rankings"], list)
    assert isinstance(result["justification"], str)

    for ranking in result["rankings"]:
        assert isinstance(ranking, dict)
        assert "rank" in ranking
        assert "workstream" in ranking
        assert isinstance(ranking["rank"], int)
        assert isinstance(ranking["workstream"], str)

    assert len(result["rankings"]) == 3


def test_parse_no_json(llm_with_no_json):
    result = parse_llm_response(llm_with_no_json)

    assert isinstance(result, str)
    assert result.startswith("Error")


# --------------------- Test find_response_errors() --------------------------
def test_find_errors_valid_data(parsed_valid_data):
    errors = find_response_errors(
        parsed_valid_data, response_keys_required, candidate_workstreams
    )
    assert errors == []


def test_find_errors_parsing_error(parsed_error_found):
    errors = find_response_errors(
        parsed_error_found, response_keys_required, candidate_workstreams
    )

    assert isinstance(errors, list)
    assert isinstance(errors[0], str)
    assert len(errors) == 1
    assert errors[0] == parsed_error_found


def test_find_errors_invalid_rank(parsed_invalid_ranking):
    errors = find_response_errors(
        parsed_invalid_ranking, response_keys_required, candidate_workstreams
    )

    assert isinstance(errors, list)
    assert len(errors) == 1
