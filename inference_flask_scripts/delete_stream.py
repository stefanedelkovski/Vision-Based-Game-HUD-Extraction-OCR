import requests


def delete_stream(stream_id):
    BASE_URL = "http://localhost:5000"
    BASE_URL = "http://3.71.77.60"
    response = requests.post(f"{BASE_URL}/delete", json={"id": stream_id})
    print(response.json())


if __name__ == "__main__":
    stream_id = "37d16c52-9bca-4eab-aee0-2ceca8fee7b6"
    delete_stream(stream_id)
