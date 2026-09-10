import requests

def live_anime_data(anime_title):
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

anime_similar_to = 'Attack on Titan'
genre_preferred = 'Action'
rating = 8

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
    2. Do NOT search an anime that already exists in observations.
    3. Search only ONE concrete anime title.
    4. Never output explanations.
    5. Check previous observations first.
    6. Two anime are considered similar if the similarity score is greater than 70%.
    Similarity weights:
    Genre: 40%
    Themes: 30%
    Tone: 20%
    Setting: 10%
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
finished = False
while(max_loop_cycles > 0):
    curr_res = call_llm_with_prompt(state)
    if curr_res.startswith("SEARCH:"):
        anime_title = curr_res.replace("SEARCH:", "").strip()
        curr_anime_data = live_anime_data(anime_title)
        state["observations"].append({
            "llm_observation": curr_res,
            "tool_result": curr_anime_data
        })
        state["iteration"] += 1
        print("Iteration: ", state["iteration"], " ", state)
    else :
        state["observations"].append({
            "llm_observation": curr_res,
            "tool_result": None
        })
        print("state:", state)
        print("Final State: ", curr_res.replace("FINISH:", "").strip())
        finished = True
        break
    max_loop_cycles -= 1

if(finished == False):
    print("Max Iterations Reached")




    
# if result1.startswith("SEARCH:"):

#     anime_title = result1.replace("SEARCH:", "").strip()
#     tool_result = test_anime(anime_title)

#     second_prompt = f"""
#         GOAL: 
#             1. the anime should be of Action oriented
#             2. It should not have more than 24 episodes
#             3. rating should be more than 8
#             4. it should not be in my rating list
#         User wants:
#         {user_request1}

#         You searched for:
#         {anime_title}

#         Tool returned:
#         {tool_result}

#         Based on this information, recommend another anime.
#         """

#     answer = ask_llm(second_prompt)
#     print("Second Prompt:", answer)
# else: 
#     print(result1.replace("Finish:", "").strip())

# # print(test_anime('Attack on Titan'))