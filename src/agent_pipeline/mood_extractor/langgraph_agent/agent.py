import json
from graph import build_graph

def run_perfume_mood_graph(perfumes: list[dict]):
    graph = build_graph()
    app = graph.compile()

    input_state = {
        "perfumes": perfumes
    }

    app.invoke(input_state)

if __name__=="__main__":
    DATA_PATH = "../../../datasets/fragrantica_perfumes.json"

    with open(DATA_PATH, "r", encoding="utf-8") as json_file:
        perfumes = json.load(json_file)

    run_perfume_mood_graph(perfumes)
