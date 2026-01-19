import requests


def get_status(stream_id):
    BASE_URL = "http://localhost:5000"
    BASE_URL = "http://3.71.77.60"
    response = requests.get(f"{BASE_URL}/status", params={"id": stream_id})
    print(response.json())


if __name__ == "__main__":
    stream_id = "d41556b5-8019-4dd4-ba11-9194e123b100"
    get_status(stream_id)

# {'active_streams': [['d41556b5-8019-4dd4-ba11-9194e123b100', 'https://kick.com/zeroarmor'], ['1766c791-c48c-4cdf-9cd1-4ec28de96a47', 'https://kick.com/casinodaddy']