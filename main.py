import requests

def test_anime(anime_title):
    anime_data = {
        "Attack on Titan": {
            "genre": ["Action", "Drama", "Fantasy"],
            "episodes": 25,
            "rating": 9.0
        },
        "Demon Slayer": {
            "genre": ["Action", "Fantasy"],
            "episodes": 26,
            "rating": 8.7
        },
        "Death Note": {
            "genre": ["Thriller", "Mystery"],
            "episodes": 37,
            "rating": 8.9
        },
        "Noragami": {
            "genre": ["Action", "Romance"],
            "episodes": 24,
            "rating": 7
        },
        "Jujutsu Kaisen":{
            "genre": ["Action", "Romance"],
            "episodes": 24,
            "rating": 9
        },
    }
    return anime_data.get(anime_title)

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


def call_llm_with_prompt(user_request):
    anime_prompt = f"""
    You are an anime recommendation agent.
    GOAL: 
        1. the anime should be of Action oriented
        2. It should not have more than 24 episodes
        3. rating should be more than 8
        4. it should not be in my rating list

    User request:
    {user_request}

    You have one tool:

    test_anime(title)

    If you need information about an anime, respond exactly:

    SEARCH: anime title

    If you already have enough information, check for all conditions to be sattisfied, if yes return:

    FINISH: your answer

    or else return SEARCH: anime title
    """

    return ask_llm(anime_prompt)


user_request1 = "I want an anime similar to Attack on Titan"

result1 = call_llm_with_prompt(user_request1)
if result1.startswith("SEARCH:"):

    anime_title = result1.replace("SEARCH:", "").strip()
    tool_result = test_anime(anime_title)

    second_prompt = f"""
        GOAL: 
            1. the anime should be of Action oriented
            2. It should not have more than 24 episodes
            3. rating should be more than 8
            4. it should not be in my rating list
        User wants:
        {user_request1}

        You searched for:
        {anime_title}

        Tool returned:
        {tool_result}

        Based on this information, recommend another anime.
        """

    answer = ask_llm(second_prompt)
    print("Second Prompt:", answer)
else: 
    print(result1.replace("Finish:", "").strip())

# print(test_anime('Attack on Titan'))