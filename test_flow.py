"""
End-to-end test for the recommendation API.
Make sure the server is running first:
    uvicorn src.api.main:app --reload
"""
import json
import sys
import requests

BASE_URL = "http://localhost:8000"


def test_recommend(image_path: str):
    print(f"\n--- Testing with image: '{image_path}' ---\n")

    with open(image_path, "rb") as img:
        with requests.post(
            f"{BASE_URL}/recommend",
            data={"input_type": "image"},
            files={"image": (image_path, img, "image/jpeg")},
            stream=True,
        ) as resp:
            print(f"Status: {resp.status_code}\n")
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                event_type = payload.get("type")

                if event_type == "moods":
                    print(f"[moods]   {payload['moods']}")
                elif event_type == "accords":
                    print(f"[accords] {payload['accords']}")
                elif event_type == "result":
                    print(f"\n[results]")
                    for i, p in enumerate(payload["recommendations"], 1):
                        print(f"  {i}. {p['name']} by {p['brand']}  (score: {p.get('final_score', 'n/a')})")
                elif event_type == "done":
                    print("\n[done]")
                elif event_type == "error":
                    print(f"[error]   {payload['message']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_flow.py <path_to_image>")
        sys.exit(1)
    test_recommend(sys.argv[1])
