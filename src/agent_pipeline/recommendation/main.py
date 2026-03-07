from graph import build_graph

app = build_graph()

result = app.invoke({
    "user_id": "test-user-001",
    "input_type": "image",
    "mood_input": "/Users/prerana1298/computing/repo/Perfume-Recommender/datasets/mood_test_files/40985806.jpeg",
})

print(result)
