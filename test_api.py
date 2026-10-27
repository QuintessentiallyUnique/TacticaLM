import requests
import time

def main():
    url = "http://127.0.0.1:8000/api/v1/analyze"
    payload = {
        "match_id": 3857256,
        "question": "How did Denmark attack against Tunisia?",
        "team": "Denmark"
    }
    
    # Wait for backend to be online
    for _ in range(60):
        try:
            res = requests.get("http://127.0.0.1:8000/api/v1/health")
            if res.status_code == 200:
                print("Backend online!")
                break
        except:
            pass
        time.sleep(2)
    else:
        print("Backend failed to start in time.")
        return

    print("Sending request...")
    start = time.time()
    res = requests.post(url, json=payload)
    print(f"Time: {time.time() - start:.2f}s")
    
    if res.status_code == 200:
        data = res.json()
        print("\nResponse:")
        print(data["response"])
    else:
        print("Error:", res.status_code, res.text)

if __name__ == "__main__":
    main()
