import json
import re
from dataclasses import dataclass

from dwutils import bedrock


# Custom parse exceptions
class LLMParseError(Exception):
    """Base class for LLM parsing failures."""


class MissingJsonError(LLMParseError):
    pass


class InvalidJsonError(LLMParseError):
    pass


# Custom validation errors class
@dataclass
class ValidationError:
    code: str
    message: str


# Custom processing result class
@dataclass
class ProcessingResult:
    response: dict | None
    error: str | None
    repair_attempts: int
    error_history: list[list[str]]


# ---------------------- LLM helper functions ----------------------
def build_llm_prompt(
    prompt_template: str,
    ev_title: str,
    ev_desc: str,
    ev_dir: str,
    cand_ws_titles: list[str],
    cand_ws_desc: list[str],
) -> str:
    """
    Uses evidence and candidate workstream details to build a prompt to pass through to an LLM for zero-shot classification.

    Args:
        prompt_template: str
            The prompt template used for building the complete prompt.
        ev_title:
            The title of the evidence to classify.
        ev_desc:
            The description of the evidence to classify.
        ev_dir:
            The directorate of the evidence to classify.
        cand_ws_titles: list
            A list of all the candidate workstream titles.
        cand_ws_desc: list
            A list of all the candidate workstream descriptions.

    Returns:
        str:
            A prompt to pass to an LLM for zero-shot classification.
    """
    workstream_text = "Candidate Workstreams:\n"
    for i, (ws_title, ws_desc) in enumerate(zip(cand_ws_titles, cand_ws_desc), start=1):
        text = f"{i}. Workstream title: {ws_title}\nObjective: {ws_desc}.\n"
        workstream_text += text

    # Build the prompt based on the prompt template format in config file
    prompt = prompt_template.format(
        ev_title=ev_title,
        ev_desc=ev_desc,
        ev_dir=ev_dir,
        workstream_text=workstream_text,
    )

    return prompt


def parse_llm_response(text: str) -> dict:
    """
    Parses the response of an LLM, returning a dict if successful or raises an exception if an error occurred.
    """
    # Immediate loading will work if LLM returned JSON object only
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # use regex in case LLM added text on top of the JSON string as part of its response
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        json_text = match.group()

        try:
            return json.loads(json_text)
        except Exception as e:
            raise InvalidJsonError(f"{type(e).__name__}: {e}") from e

    raise MissingJsonError("No JSON object found in LLM response.")


def find_response_errors(
    parsed_response: dict,
    response_keys_required: set,
    candidate_workstreams: list[str],
) -> list[ValidationError]:
    """
    Validate a parsed LLM response.

    Returns:
        list[ValidationError]
            Empty list if valid.
    """

    errors: list[ValidationError] = []

    # check for missing dictionary keys
    missing_keys = response_keys_required - parsed_response.keys()

    if missing_keys:
        errors.append(
            ValidationError(
                code="missing_keys",
                message=f"Response didn't include the following keys in its JSON: {missing_keys}",
            )
        )

    if "rankings" not in parsed_response:
        return errors

    # check for incorrectly formatted rankings
    rankings = parsed_response["rankings"]
    actual_rankings = [ws_dict["rank"] for ws_dict in rankings]

    expected_rankings = list(range(1, len(candidate_workstreams) + 1))
    if actual_rankings != expected_rankings:
        errors.append(
            ValidationError(
                code="incorrect_rankings_format",
                message=f"Rankings returned in JSON object were {actual_rankings} while rankings expected were {expected_rankings}",
            )
        )

    # Check for invalid workstreams
    actual_workstreams = {ws_dict["workstream"] for ws_dict in rankings}
    if actual_workstreams != set(candidate_workstreams):
        errors.append(
            ValidationError(
                code="invalid_workstreams",
                message=f"Workstreams returned in JSON object were {actual_workstreams} while workstreams expected were {candidate_workstreams}.",
            )
        )

    return errors


def build_repair_prompt(
    llm_response: str, original_prompt: str, llm_errors: list[ValidationError]
) -> str:
    """
    Prepares a prompt to point the LLM to its error.
    """

    error_text = "\n".join(f"[{err.code}] {err.message}" for err in llm_errors)

    prompt = f"""
        You were given the original task: 
        
        {original_prompt}
        
        Your previous response was: 
        
        {llm_response}

        The following errors were detected: 
        
        {error_text}

        Please return a corrected response that fixes all of the above errors.

        Return only the corrected JSON object. Do not include any explanations or markdown.
    """

    return prompt


def process_response_wrapper(
    llm_response: str,
    original_prompt: str,
    response_keys_required: set,
    candidate_workstreams: list[str],
    model_id: str,
    max_repairs: int = 3,
) -> ProcessingResult:
    """
    Wrapper to process an LLM response, and then reprompt the LLM if an error was encountered.
    The number of times a reprompt is tried is up to max_tries.
    """

    error_history: list[list[str]] = []

    response_to_process = llm_response

    for repair_attempt in range(max_repairs + 1):
        try:
            parsed_response = parse_llm_response(response_to_process)

            # Check for errors
            response_errors = find_response_errors(
                parsed_response, response_keys_required, candidate_workstreams
            )

            current_error_codes = [err.code for err in response_errors]
            error_history.append(current_error_codes)

        except LLMParseError as e:
            response_errors = [ValidationError(code="parse_error", message=str(e))]

            current_error_codes = [err.code for err in response_errors]
            error_history.append(current_error_codes)

        if not response_errors:
            return ProcessingResult(
                response=parsed_response,
                error=None,
                repair_attempts=repair_attempt,
                error_history=error_history,
            )

        if repair_attempt == max_repairs:
            return ProcessingResult(
                response=None,
                error="\n".join(err.message for err in response_errors),
                repair_attempts=repair_attempt,
                error_history=error_history,
            )

        repair_prompt = build_repair_prompt(
            response_to_process, original_prompt, response_errors
        )

        # Reprompt the LLM with the repair prompt
        response_to_process = bedrock.invoke_simple(
            model_id=model_id, prompt=repair_prompt
        )["model_response_string"]
