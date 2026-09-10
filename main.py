import requests

# def test_anime(anime_title):
#     return {
#         "title": anime_title,
#         "genre": ['Action', 'Adventure', 'Drama'],
#         "noOfExpisods": 24,
#         "rating": 8.5
#     }

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


response = ask_llm(
    "The user likes Attack on Titan. "
    "Give me relevent anime max 5 count"
)

print(response)

# print(test_anime('Attack on Titan'))