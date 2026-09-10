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


state = {
    "user_request" : "I want a single anime similar to Attack on Titan, I prefer action oriented, with rating over 8",
    "observations":[],
    "iteration": 0,
}

def call_llm_with_prompt(state):
    llm_prompt = f"""
    You are an anime recommendation agent.


    User state: 
        consists of user_request which is the goal you need to satissfy
        consists of observations that have been results obtained by previous model as well as information provided by external tool.
        consists of iteration the number of times looping was done

    Current state:
    {state}

    Do not consider same anime as specified by user
    Once a anime is observed, do not produce it again
    Give anime recommendation based on concrete title
    two anime are consider similar if the percentage is > 70% with weights
        Genre       40%
        Themes      30%
        Tone        20%
        Setting     10%

    You have one tool: it gives information available on the internet

    live_anime_data(title)

    If you need additional information about an anime, respond exactly:

    SEARCH: anime title

    If you already have enough information, check for all conditions that user requested to be sattisfied, if yes return:

    FINISH: your answer

    or else return SEARCH: anime title

    Return only SEARCH or FINISH.
    """

    return ask_llm(llm_prompt)


max_loop_cycles = 5
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
        # print("state:", state)
        print("Final State: ", curr_res.replace("FINISH:", "").strip())
        break
    max_loop_cycles -= 1






    
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