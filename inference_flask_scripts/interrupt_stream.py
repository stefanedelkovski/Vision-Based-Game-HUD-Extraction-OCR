import requests


def interrupt_stream(stream_id):
    BASE_URL = "http://localhost:5000"
    BASE_URL = "http://3.71.77.60"
    response = requests.post(f"{BASE_URL}/interrupt", json={"id": stream_id})
    print(response.json())


if __name__ == "__main__":
    stream_id = "d41556b5-8019-4dd4-ba11-9194e123b100"
    interrupt_stream(stream_id)



# {'active_streams': [['a231684e-473e-4dee-b37f-47be13979012', 'https://kick.com/hellohashi'],
# ['d54f54cc-bb71-4b14-bad6-cd298454f808', 'https://kick.com/diegawinos']],
# 'inactive_streams': [['f1be883b-29ea-49c3-ab83-32ab0cf09aa0', 'https://kick.com/zeroarmor']]}