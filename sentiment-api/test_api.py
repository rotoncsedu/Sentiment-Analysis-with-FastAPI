import requests

BASE_URL = "http://localhost:8000"


def test_home():
    response = requests.get(f"{BASE_URL}/")
    print("GET / ->", response.status_code)
    print(response.json())
    print()


def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("GET /health ->", response.status_code)
    print(response.json())
    print()


def test_predict(text):
    response = requests.post(f"{BASE_URL}/predict", json={"text": text})
    print(f"POST /predict  (text={text!r})")
    print("Status:", response.status_code)
    print("Response:", response.json())
    print()


if __name__ == "__main__":
    test_home()
    test_health()

    # A few sample reviews to sanity-check both classes
    test_predict("This movie was absolutely amazing!")
    test_predict("Phil the Alien is one of those quirky films where the humour is based around the oddness of everything rather than actual punchlines.At first it was very odd and pretty funny but as the movie progressed I didn't find the jokes or oddness funny anymore.")
    test_predict("This movie was absolutely boring!")