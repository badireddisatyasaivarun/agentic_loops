import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# def already_watched_anime(anime_title):
#     anime_data = {
#         "Attack on Titan": {
#             "rating": 9.0
#         },
#         "Demon Slayer": {
#             "rating": 8.7
#         },
#         "Death Note": {
#             "rating": 8.9
#         },
#         "Noragami": {
#             "rating": 7
#         },
#         "Jujutsu Kaisen":{
#             "rating": 9
#         },
#     }
#     return anime_data.get(anime_title)

def live_anime_data(anime_title):

    url = "https://kitsu.io/api/edge/anime"
    params = {
        "filter[text]": anime_title,
        "include": "categories",
        "page[limit]": 1
    }
    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("data"):
            return None

        anime = result["data"][0]
        attributes = anime["attributes"]
        category_map = {}
        for category in result.get("included", []):
            if category.get("type") == "categories":
                category_id = category.get("id")
                category_name = (
                    category
                    .get("attributes", {})
                    .get("title")
                )
                if category_id and category_name:
                    category_map[category_id] = category_name
        genres = []
        relationships = anime.get("relationships", {})
        category_relationships = (
            relationships
            .get("categories", {})
            .get("data", [])
        )

        for category in category_relationships:
            category_id = category.get("id")
            if category_id in category_map:
                genres.append(category_map[category_id])

        rating = attributes.get("averageRating")
        if rating:
            rating = float(rating) / 10

        return {
            "title": attributes.get("canonicalTitle"),
            "genre": genres,
            "episodes": attributes.get("episodeCount"),
            "rating": rating,
            "synopsis": attributes.get("synopsis"),
            "status": attributes.get("status")
        }

    except requests.RequestException as error:
        print("Kitsu API error:", error)
        return None

# calling Ollama with a defined prompt
def ask_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


def animeRecommendationService(req_obj):

    anime_similar_to = req_obj.anime_similar_to
    genre_preferred = ", ".join(req_obj.genre_preferred) 
    rating = req_obj.min_rating

    state = {
        "user_request" : f"""I want a single anime similar to {anime_similar_to}, I prefer {genre_preferred} oriented, with rating over {rating}""",
        "observations":[],
        "iteration": 0,
    }

    def call_llm_with_prompt(state):
        llm_prompt = f"""
        You are an anime recommendation agent.

        Current state:
        {state}

        User request:
        {state["user_request"]}

        Rules:

        1. Do NOT recommend the same anime mentioned in the user request.
        2. Do NOT search an anime that exists twice in observations.
        3. Search only ONE concrete anime title.
        4. Never output explanations.
        5. Check previous observations first.
        6. Two anime are considered similar if the similarity score is greater than 70%.
        Similarity weights:
        Genre: 40%
        Story/Synopsis: 60%
        7. If an anime in observations satisfies:
        - similarity > 70%
        - {genre_preferred} oriented
        - anime rating must be greater than {rating}

        then return exactly:
        FINISH: <anime title>
        Do not include explanations, reasoning, sentences, or additional text.
        
        8. If no observed anime satisfies all requirements, return exactly:
        SEARCH: <anime title>
        Do not include explanations, reasoning, sentences, or additional text.
        
        9. If it succeed return your output
        You have one tool:
        live_anime_data(title)

        Return exactly one line.
        Return only SEARCH or FINISH.
        """


        return ask_llm(llm_prompt)


    max_loop_cycles = 5
    while(max_loop_cycles > 0):
        state["iteration"] += 1
        curr_res = call_llm_with_prompt(state).strip()
        if curr_res.startswith("SEARCH:"):
            anime_title = curr_res.replace("SEARCH:", "").strip()
            curr_anime_data = live_anime_data(anime_title)
            state["observations"].append({
                "llm_observation": curr_res,
                "tool_result": curr_anime_data
            })
            print("Iteration: ", state["iteration"], " ", state)
        elif curr_res.startswith("FINISH:"):
            state["observations"].append({
                "llm_observation": curr_res,
                "tool_result": None
            })
            print("state:", state)
            return {
                "success": True,
                "anime": curr_res.replace("FINISH:", "").strip(),
                "iterations": state["iteration"]
            }
        else :
            return {
                "success": False,
                "error": "Failure in Recommending Anime"
            }
        max_loop_cycles -= 1

    return {
        "success": False,
        "error": "Maximum iterations reached"
    }




class RecommendationRequest(BaseModel):
    anime_similar_to: str
    genre_preferred: list[str]
    min_rating: float


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    return animeRecommendationService(request)