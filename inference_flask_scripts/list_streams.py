import requests


def list_streams():
    BASE_URL = "http://localhost:5000"
    BASE_URL = "http://3.71.77.60"
    response = requests.get(f"{BASE_URL}/list")
    print(response.json())


if __name__ == "__main__":
    list_streams()


