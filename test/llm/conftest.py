from textwrap import dedent

import pytest


# Dummy LLM responses
@pytest.fixture
def llm_valid_response():
    return dedent("""
    {
        "rankings": [
            {
                "rank": 1,
                "workstream": "4 - SME Growth Support"
            },
            {
                "rank": 2,
                "workstream": "7 - Innovation Accelerator"
            },
            {
                "rank": 3,
                "workstream": "8 - Investment Attraction Programme"
            }
        ],
        "justification": "Test justification of why the rankings were ordered in this way."
    }
    """)


@pytest.fixture
def llm_invalid_json():
    return dedent("""
    {
        "rankings": [
            {"rank": 1, "workstream": "4 - SME Growth Support",
            {"rank": 2, "workstream": "7 - Innovation Accelerator"}
        ],
        "justification": "Test justification of why the rankings were ordered in this way."
    }
    """)


@pytest.fixture
def llm_invalid_workstream():
    return dedent("""
    {
        "rankings": [
            {
                "rank": 1,
                "workstream": "4 - SME Growth Support"
            },
            {
                "rank": 2,
                "workstream": "A"
            },
            {
                "rank": 3,
                "workstream": "8 - Investment Attraction Programme"
            }
        ],
        "justification": "Test justification of why the rankings were ordered in this way."
    }
    """)


@pytest.fixture
def llm_invalid_ranking():
    return dedent("""
    {
        "rankings": [
            {
                "rank": 1,
                "workstream": "4 - SME Growth Support"
            },
            {
                "rank": 3,
                "workstream": "8 - Investment Attraction Programme"
            }
        ],
        "justification": "Test justification of why the rankings were ordered in this way."
    }
    """)


@pytest.fixture
def llm_json_with_added_text():
    return dedent("""
    Here are the rankings you requested:
    {
        "rankings": [
            {
                "rank": 1,
                "workstream": "4 - SME Growth Support"
            },
            {
                "rank": 2,
                "workstream": "7 - Innovation Accelerator"
            },
            {
                "rank": 3,
                "workstream": "8 - Investment Attraction Programme"
            }
        ],
        "justification": "Test justification of why the rankings were ordered in this way."
    }
    Let me know if you need anything else.
    """)


@pytest.fixture
def llm_with_no_json():
    return """
    Here are the rankings you requested:
        {
            "rankings": [
                {
                    "rank": 1,
                    "workstream": "4 - SME Growth Support"
                {
                    "rank": 2,
                    "workstream": "7 - Innovation Accelerator",
            
            "justification": "here is a test justification"
    """


# Dummy parsed data
@pytest.fixture
def parsed_valid_data():
    return {
        "rankings": [
            {"rank": 1, "workstream": "4 - SME Growth Support"},
            {"rank": 2, "workstream": "7 - Innovation Accelerator"},
            {"rank": 3, "workstream": "8 - Investment Attraction Programme"},
        ],
        "justification": "Test justification of why the rankings were ordered in this way.",
    }


@pytest.fixture
def parsed_error_found():
    return "Error: No JSON object in LLM response."


@pytest.fixture
def parsed_invalid_ranking():
    return {
        "rankings": [
            {"rank": 1, "workstream": "4 - SME Growth Support"},
            {"rank": 3, "workstream": "7 - Innovation Accelerator"},
            {"rank": 3, "workstream": "8 - Investment Attraction Programme"},
        ],
        "justification": "Test justification of why the rankings were ordered in this way.",
    }


@pytest.fixture
def parsed_invalid_workstream():
    return {
        "rankings": [
            {"rank": 1, "workstream": "4 - SME Growth Support"},
            {"rank": 2, "workstream": "An invalid workstream"},
            {"rank": 3, "workstream": "8 - Investment Attraction Programme"},
        ],
        "justification": "Test justification of why the rankings were ordered in this way.",
    }


@pytest.fixture
def parsed_no_justification():
    return {
        "rankings": [
            {"rank": 1, "workstream": "4 - SME Growth Support"},
            {"rank": 2, "workstream": "7 - Innovation Accelerator"},
            {"rank": 3, "workstream": "8 - Investment Attraction Programme"},
        ],
    }


@pytest.fixture
def parsed_rankings_typo():
    return {
        "ranking_typo": [
            {"rank": 1, "workstream": "4 - SME Growth Support"},
            {"rank": 2, "workstream": "7 - Innovation Accelerator"},
            {"rank": 3, "workstream": "8 - Investment Attraction Programme"},
        ],
        "justification": "Test justification of why the rankings were ordered in this way.",
    }


@pytest.fixture
def parsed_multiple_errors():
    return {
        "ranking_typo": [
            {"rank": 1, "workstream": "4 - SME Growth Support"},
            {"rank": 2, "workstream": "An invalid workstreams"},
            {"rank": 5, "workstream": "8 - Investment Attraction Programme"},
        ],
    }
