import logging
import os
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anime_recommender")

app = FastAPI(title="Anime Recommendation Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_LOOP_CYCLES = 5
KITSU_URL = "https://kitsu.io/api/edge/anime"
GROQ_MODEL = "openai/gpt-oss-120b"

# ---------------------------------------------------------------------------
# External clients
# ---------------------------------------------------------------------------

_groq_client: Optional[Groq] = None


def get_groq_client() -> Groq:
    """Lazily create the Groq client and fail with a clear error if the
    API key is missing, instead of failing deep inside the SDK at import time."""
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Set it before starting the service."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


_http_session = requests.Session()


# ---------------------------------------------------------------------------
# Kitsu API
# ---------------------------------------------------------------------------

def live_anime_data(anime_title: str) -> Optional[dict]:
    """Look up an anime on Kitsu and return a normalized dict, or None on
    any failure (not found, network error, malformed response)."""
    params = {
        "filter[text]": anime_title,
        "include": "categories",
        "page[limit]": 1,
    }
    try:
        response = _http_session.get(KITSU_URL, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()

        data = result.get("data")
        if not data:
            return None

        anime = data[0]
        attributes = anime.get("attributes", {})

        category_map = {
            category["id"]: category.get("attributes", {}).get("title")
            for category in result.get("included", [])
            if category.get("type") == "categories" and category.get("id")
        }

        genre_refs = (
            anime.get("relationships", {})
            .get("categories", {})
            .get("data", [])
        )
        genres = [
            category_map[ref["id"]]
            for ref in genre_refs
            if ref.get("id") in category_map and category_map[ref["id"]]
        ]

        rating = attributes.get("averageRating")
        rating = float(rating) / 10 if rating is not None else None

        poster_image = attributes.get("posterImage") or {}
        image_url = (
            poster_image.get("large")
            or poster_image.get("medium")
            or poster_image.get("original")
            or poster_image.get("small")
        )

        return {
            "title": attributes.get("canonicalTitle"),
            "genre": genres,
            "episodes": attributes.get("episodeCount"),
            "rating": rating,
            "synopsis": attributes.get("synopsis"),
            "status": attributes.get("status"),
            "image": image_url,
        }

    except requests.RequestException as error:
        logger.warning("Kitsu API error for %r: %s", anime_title, error)
        return None
    except (KeyError, ValueError, TypeError) as error:
        logger.warning("Kitsu response parsing error for %r: %s", anime_title, error)
        return None


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def ask_llm(prompt: str) -> Optional[str]:
    try:
        client = get_groq_client()
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=2048,
            reasoning_effort="medium",
            stream=False,
        )
        return completion.choices[0].message.content
    except Exception as error:
        logger.error("Groq API error: %s", error)
        return None


# ---------------------------------------------------------------------------
# Recommendation agent
# ---------------------------------------------------------------------------

class RecommendationRequest(BaseModel):
    anime_similar_to: str = Field(..., min_length=1)
    genre_preferred: list[str] = Field(..., min_length=1)
    min_rating: float = Field(..., ge=0, le=10)
    exclude_titles: list[str] = Field(default_factory=list)

    @field_validator("anime_similar_to")
    @classmethod
    def strip_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("anime_similar_to must not be blank")
        return v

    @field_validator("genre_preferred")
    @classmethod
    def clean_genres(cls, v: list[str]) -> list[str]:
        cleaned = [g.strip() for g in v if g.strip()]
        if not cleaned:
            raise ValueError("genre_preferred must contain at least one genre")
        return cleaned

    @field_validator("exclude_titles")
    @classmethod
    def clean_exclude_titles(cls, v: list[str]) -> list[str]:
        return [t.strip() for t in v if t.strip()]


class RecommendationResponse(BaseModel):
    success: bool
    anime: str
    iterations: int
    image: Optional[str] = None
    description: Optional[str] = None
    genre: list[str] = Field(default_factory=list)
    rating: Optional[float] = None


def _format_observations(observations: list[dict]) -> str:
    """Render observations as compact, readable text instead of a raw dict
    repr, so the LLM gets clean signal without wasted tokens."""
    if not observations:
        return "(none yet)"
    lines = []
    for i, obs in enumerate(observations, 1):
        result = obs["tool_result"]
        if result is None:
            lines.append(f"{i}. {obs['llm_observation']} -> no data found")
        else:
            lines.append(
                f"{i}. title={result.get('title')!r}, "
                f"genre={result.get('genre')}, "
                f"rating={result.get('rating')}, "
                f"is_valid_anime={obs['is_valid_anime']}"
            )
    return "\n".join(lines)


def _build_prompt(user_request: str, observations: list[dict], excluded_titles: list[str]) -> str:
    excluded_str = ", ".join(excluded_titles) if excluded_titles else "(none)"
    return f"""You are an anime recommendation agent.

    User request:
    {user_request}

    Already-recommended anime to avoid (never finish on these, never treat them as valid):
    {excluded_str}

    Observations so far:
    {_format_observations(observations)}

    Rules:
    1. Do NOT recommend the anime mentioned in the user request or any title in the
    already-recommended list above.
    2. Do NOT search again for a title that already appears in observations.
    Still evaluate anime already in observations to see if one satisfies the requirements.
    3. Search only ONE real, existing anime title per turn.
    The title MUST refer to an actual anime, not a movie, song, person, or band.
    Do not invent titles. Minor spelling/grammar errors under ~10% difference
    from a valid title should be treated as that title.
    4. Never output explanations.
    5. An observation's is_valid_anime being False means it can never be a final recommendation.
    6. Two anime are considered similar if similarity > 70%, weighted:
    genre 40%, story/synopsis 60%.
    7. If an observation has is_valid_anime True AND similarity > 70%, return exactly:
    FINISH: <anime title>
    8. Otherwise return exactly:
    SEARCH: <anime title>

    Output exactly one line, either SEARCH: ... or FINISH: ..., nothing else.
    """


def _validate_finish(anime_title: str, observations: list[dict]) -> Optional[dict]:
    """Guard against the LLM hallucinating a FINISH for a title that was
    never actually validated against Kitsu data."""
    target = anime_title.strip().lower()
    for obs in observations:
        result = obs.get("tool_result")
        if (
            obs.get("is_valid_anime")
            and result
            and (result.get("title") or "").strip().lower() == target
        ):
            return result
    return None


def anime_recommendation_service(req: RecommendationRequest) -> RecommendationResponse:
    genre_preferred_str = ", ".join(req.genre_preferred)
    preferred_genres = {g.lower() for g in req.genre_preferred}

    excluded_titles = [req.anime_similar_to, *req.exclude_titles]
    excluded_normalized = {t.lower() for t in excluded_titles}

    user_request = (
        f"I want a single anime similar to {req.anime_similar_to}, "
        f"I prefer {genre_preferred_str} oriented, with rating over {req.min_rating}"
    )
    observations: list[dict] = []
    searched_titles: dict[str, dict] = {}  # normalized title -> observation

    for iteration in range(1, MAX_LOOP_CYCLES + 1):
        prompt = _build_prompt(user_request, observations, excluded_titles)
        raw_response = ask_llm(prompt)

        if raw_response is None:
            raise HTTPException(
                status_code=502,
                detail="The recommendation model is currently unavailable. Please try again.",
            )

        curr_res = raw_response.strip()

        if curr_res.startswith("FINISH:"):
            anime_title = curr_res.replace("FINISH:", "", 1).strip()
            matched = _validate_finish(anime_title, observations) if anime_title else None
            if not matched:
                logger.warning(
                    "LLM returned FINISH for an unvalidated title: %r", anime_title
                )
                raise HTTPException(
                    status_code=502,
                    detail="The model produced a recommendation that could not be verified.",
                )
            logger.info("Recommendation found after %d iteration(s): %s", iteration, anime_title)
            return RecommendationResponse(
                success=True,
                anime=matched.get("title") or anime_title,
                iterations=iteration,
                image=matched.get("image"),
                description=matched.get("synopsis"),
                genre=matched.get("genre") or [],
                rating=matched.get("rating"),
            )

        if curr_res.startswith("SEARCH:"):
            anime_title = curr_res.replace("SEARCH:", "", 1).strip()
            if not anime_title:
                raise HTTPException(status_code=502, detail="Invalid empty search title from model")

            normalized = anime_title.lower()

            if normalized in searched_titles:
                # Enforce the "don't re-search" rule in code rather than trusting the prompt.
                observations.append(searched_titles[normalized])
                logger.info("Iteration %d: reused cached search for %r", iteration, anime_title)
                continue

            curr_anime_data = live_anime_data(anime_title)

            is_atleast_one_genre_match = False
            if curr_anime_data and curr_anime_data.get("genre"):
                anime_genres = {g.lower() for g in curr_anime_data["genre"]}
                is_atleast_one_genre_match = bool(preferred_genres & anime_genres)

            is_excluded = bool(
                curr_anime_data
                and (curr_anime_data.get("title") or "").strip().lower() in excluded_normalized
            )

            is_valid_anime = bool(
                curr_anime_data
                and not is_excluded
                and curr_anime_data.get("rating") is not None
                and curr_anime_data["rating"] > req.min_rating
                and is_atleast_one_genre_match
            )

            observation = {
                "llm_observation": curr_res,
                "tool_result": curr_anime_data,
                "is_valid_anime": is_valid_anime,
            }
            observations.append(observation)
            searched_titles[normalized] = observation
            logger.info("Iteration %d: searched %r -> valid=%s", iteration, anime_title, is_valid_anime)
            continue

        raise HTTPException(status_code=502, detail=f"Invalid model response: {curr_res!r}")

    raise HTTPException(status_code=422, detail="Maximum iterations reached without a recommendation")


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    return anime_recommendation_service(request)