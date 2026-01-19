import requests


def add_stream(url, save_images=False):
    BASE_URL = "http://localhost:5000"
    BASE_URL = "http://3.71.77.60"
    response = requests.post(f"{BASE_URL}/add", json={"url": url, "save_images": save_images})
    # Print the status code
    print(f"Status Code: {response.status_code}")

    try:
        response_json = response.json()
        print(response_json)
    except requests.exceptions.JSONDecodeError:
        print("Response content is not valid JSON")
        print("Raw Response Text:", response.text)


if __name__ == "__main__":
    url = "https://kick.com/casinodaddy"
    save_images = True
    add_stream(url, save_images)
