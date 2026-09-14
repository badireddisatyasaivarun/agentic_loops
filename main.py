import logging
import os
from typing import Optional, TypedDict, Literal

from langgraph.graph import StateGraph, START, END

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
KITSU_BASE_URL = "https://kitsu.io/api/edge"
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

def live_content_data(title: str, content_type: str) -> Optional[dict]:
    """Look up an anime / manga on Kitsu and return a normalized dict, or None on
    any failure (not found, network error, malformed response)."""
    url = f"{KITSU_BASE_URL}/{content_type}"
    params = {
        "filter[text]": title,
        "include": "categories",
        "page[limit]": 1,
    }
    try:
        response = _http_session.get(url, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()

        data = result.get("data")
        if not data:
            return None

        content = data[0]
        attributes = content.get("attributes", {})

        category_map = {
            category["id"]: category.get("attributes", {}).get("title")
            for category in result.get("included", [])
            if category.get("type") == "categories" and category.get("id")
        }

        category_refs = (
            content.get("relationships", {})
            .get("categories", {})
            .get("data", [])
        )
        genres = [
            category_map[ref["id"]]
            for ref in category_refs
            if ref.get("id") in category_map
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
            "episodes": attributes.get("episodeCount") if content_type == "anime" else None,
            "chapters": attributes.get("chapterCount") if content_type == "manga" else None,
            "volumes": attributes.get("volumeCount") if content_type == "manga" else None,
            "rating": rating,
            "synopsis": attributes.get("synopsis"),
            "status": attributes.get("status"),
            "image": image_url,
            "type": content_type,
        }

    except requests.RequestException as error:
        logger.warning("Kitsu API error for %r: %s", title, error)
        return None
    except (KeyError, ValueError, TypeError) as error:
        logger.warning("Kitsu response parsing error for %r: %s", title, error)
        return None

# ---------------------------------------------------------------------------
# Sample Data Watched Anime / Manga
# ---------------------------------------------------------------------------

SAMPLE_WATCHED_ANIME = [
    "Death Note",
    "Attack on Titan",
    "Demon Slayer",
    "Jujutsu Kaisen",
    "Naruto",
    "Monster",
    "Code Geass",
]

SAMPLE_READ_MANGA = [
    "Berserk",
    "Vagabond",
    "Chainsaw Man",
    "Tokyo Ghoul",
    "One Punch Man",
]


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
    content_type: Literal["anime", "manga"]
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
    title: str
    content_type: str
    iterations: int
    image: Optional[str] = None
    description: Optional[str] = None
    genre: list[str] = Field(default_factory=list)
    rating: Optional[float] = None
    episodes: Optional[int] = None
    chapters: Optional[int] = None
    volumes: Optional[int] = None



class RecommendationState(TypedDict):
    anime_similar_to: str
    content_type: str
    genre_preferred: list[str]
    min_rating: float
    exclude_titles: list[str]
    watched_anime: list[str]
    read_manga: list[str]
    rejected_titles: list[str]
    recommendation: Optional[str]
    recommendation_data: Optional[dict]
    valid: bool
    already_consumed: bool
    attempts: int



def anime_agent(state: RecommendationState):
    prompt = f"""
You are an anime recommendation agent.

Recommend ONE anime similar to:
{state["anime_similar_to"]}

Preferred genres:
{state["genre_preferred"]}

Minimum rating:
{state["min_rating"]}

Do NOT recommend:
{state["rejected_titles"]}

Recommend one real existing anime.

Return ONLY the anime title.
"""
    title = ask_llm(prompt)
    if not title:
        return {
            "recommendation": None,
            "recommendation_data": None,
            "attempts": state["attempts"] + 1,
        }
    title = title.strip()
    data = live_content_data(title,"anime")
    return {
        "recommendation": title,
        "recommendation_data": data,
        "attempts": state["attempts"] + 1,
    }


def manga_agent(state: RecommendationState):
    prompt = f"""
You are a manga recommendation agent.

Recommend ONE manga similar to:
{state["anime_similar_to"]}

Preferred genres:
{state["genre_preferred"]}

Minimum rating:
{state["min_rating"]}

Do NOT recommend:
{state["rejected_titles"]}

Recommend one real existing manga.

Return ONLY the manga title.
"""

    title = ask_llm(prompt)
    if not title:
        return {
            "recommendation": None,
            "recommendation_data": None,
            "attempts": state["attempts"] + 1,
        }
    title = title.strip()
    data = live_content_data(title, "manga")
    return {
        "recommendation": title,
        "recommendation_data": data,
        "attempts": state["attempts"] + 1,
    }


def validate_recommendation(state: RecommendationState):
    data = state["recommendation_data"]
    if not data:
        recommendation = state["recommendation"]
        rejected = state["rejected_titles"]
        if recommendation:
            rejected = [
                *rejected,
                recommendation
            ]
        return {
            "valid": False,
            "rejected_titles": rejected
        }
    rating = data.get("rating")
    if (rating is None or rating < state["min_rating"]):
        return {
            "valid": False,
            "rejected_titles": [
                *state["rejected_titles"],
                data["title"]
            ]
        }
    preferred_genres = {
        genre.lower()
        for genre in state["genre_preferred"]
    }
    actual_genres = {
        genre.lower()
        for genre in data.get("genre", [])
    }

    if ( preferred_genres and not preferred_genres.intersection( actual_genres )):
        return {
            "valid": False,
            "rejected_titles": [
                *state["rejected_titles"],
                data["title"]
            ]
        }

    return {
        "valid": True
    }

def check_history( state: RecommendationState ):
    data = state[
        "recommendation_data"
    ]

    if not data:
        return {
            "already_consumed": False
        }
    
    title = data["title"].strip().lower()

    if state["content_type"] == "anime":
        history = {
            item.strip().lower()
            for item
            in state["watched_anime"]
        }
    else:
        history = {
            item.strip().lower()
            for item
            in state["read_manga"]
        }

    if title in history:
        return {
            "already_consumed": True,
            "rejected_titles": [
                *state["rejected_titles"],
                data["title"]
            ]
        }

    return {
        "already_consumed":
            False
    }


def route_content(state: RecommendationState):
    return state["content_type"]


def route_validation( state: RecommendationState ):
    if state["attempts"] >= MAX_LOOP_CYCLES:
        return "end"
    if state["valid"]:
        return "check_history"
    return "retry"

def route_history(state: RecommendationState):
    if state["attempts"] >= MAX_LOOP_CYCLES:
        return "end"
    if state["already_consumed"]:
        return "retry"
    return "end"

def retry_node(state: RecommendationState):
    return {}


def retry_route(state: RecommendationState):
    return state["content_type"]


# define nodes
graph = StateGraph(RecommendationState)

graph.add_node("anime_agent", anime_agent )
graph.add_node("manga_agent", manga_agent )
graph.add_node("validate", validate_recommendation )
graph.add_node("check_history", check_history )
graph.add_node("retry", retry_node )



# Routing
graph.add_conditional_edges(
    START,
    route_content,
    {
        "anime":
            "anime_agent",
        "manga":
            "manga_agent",
    }
)


graph.add_edge("anime_agent", "validate" )
graph.add_edge("manga_agent", "validate" )
graph.add_conditional_edges(
    "validate",
    route_validation,
    {
        "check_history":
            "check_history",
        "retry":
            "retry",
        "end":
            END,
    }
)

graph.add_conditional_edges(
    "check_history",
    route_history,
    {
        "retry":
            "retry",
        "end":
            END,
    }
)


graph.add_conditional_edges(
    "retry",
    retry_route,
    {
        "anime":
            "anime_agent",
        "manga":
            "manga_agent",
    }
)

recommendation_graph = graph.compile()

def anime_recommendation_service( req: RecommendationRequest ) -> RecommendationResponse:
    initial_state: RecommendationState = {
        "anime_similar_to": req.anime_similar_to,
        "content_type": req.content_type,
        "genre_preferred": req.genre_preferred,
        "min_rating": req.min_rating,
        "exclude_titles": req.exclude_titles,
        "watched_anime": SAMPLE_WATCHED_ANIME,
        "read_manga": SAMPLE_READ_MANGA,
        "rejected_titles": [
            req.anime_similar_to,
            *req.exclude_titles,
        ],
        "recommendation": None,
        "recommendation_data": None,
        "valid": False,
        "already_consumed": False,
        "attempts": 0,
    }

    result = recommendation_graph.invoke( initial_state )
    data = result.get( "recommendation_data" )
    if (not data or not result["valid"] or result["already_consumed"]):
        raise HTTPException(
            status_code=422,
            detail=(
                "Maximum iterations reached "
                "without a recommendation"
            )
        )

    return RecommendationResponse(
        success =True,
        title = data["title"],
        content_type = result["content_type"],
        iterations = result["attempts"],
        image = data.get("image"),
        description = data.get("synopsis"),
        genre = data.get("genre") or [],
        rating = data.get("rating"),
        episodes = data.get("episodes"),
        chapters = data.get("chapters"),
        volumes = data.get("volumes"),
    )

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    return anime_recommendation_service(request)