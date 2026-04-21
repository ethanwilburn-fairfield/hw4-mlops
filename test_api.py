import requests

BASE_URL = "http://127.0.0.1:5000"

valid_record = {
    "delivery_days": 8.0,
    "delivery_vs_estimated": -15.0,
    "order_hour": 14,
    "order_dayofweek": 2,
    "total_item_price": 278.99,
    "total_freight_value": 28.03,
    "total_order_value": 307.02,
    "freight_ratio": 0.09129698390984302,
    "num_items": 1,
    "num_sellers": 1,
    "order_complexity": 1,
    "is_late": 0,
    "main_product_category": "cool_stuff",
    "main_seller_state": "ES",
    "customer_state": "PR"
}

def print_result(test_name, passed, response=None):
    status = "PASS" if passed else "FAIL"
    print(f"{test_name}: {status}")
    if response is not None:
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(response.text)
    print("-" * 60)

# Test 1
try:
    r = requests.get(f"{BASE_URL}/health")
    passed = r.status_code == 200 and "status" in r.json() and "model" in r.json()
    print_result("Test 1 - GET /health", passed, r)
except Exception as e:
    print_result("Test 1 - GET /health", False)
    print(e)
    print("-" * 60)

# Test 2
try:
    r = requests.post(f"{BASE_URL}/predict", json=valid_record)
    passed = r.status_code == 200 and "prediction" in r.json() and "probability" in r.json() and "label" in r.json()
    print_result("Test 2 - POST /predict valid request", passed, r)
except Exception as e:
    print_result("Test 2 - POST /predict valid request", False)
    print(e)
    print("-" * 60)

# Test 3
try:
    batch_payload = [valid_record.copy() for _ in range(5)]
    r = requests.post(f"{BASE_URL}/predict/batch", json=batch_payload)
    passed = r.status_code == 200 and "predictions" in r.json() and len(r.json()["predictions"]) == 5
    print_result("Test 3 - POST /predict/batch valid batch", passed, r)
except Exception as e:
    print_result("Test 3 - POST /predict/batch valid batch", False)
    print(e)
    print("-" * 60)

# Test 4
try:
    missing_field_record = valid_record.copy()
    del missing_field_record["delivery_days"]
    r = requests.post(f"{BASE_URL}/predict", json=missing_field_record)
    passed = r.status_code == 400 and "error" in r.json() and "details" in r.json()
    print_result("Test 4 - POST /predict missing field", passed, r)
except Exception as e:
    print_result("Test 4 - POST /predict missing field", False)
    print(e)
    print("-" * 60)

# Test 5
try:
    invalid_type_record = valid_record.copy()
    invalid_type_record["total_item_price"] = "not_a_number"
    r = requests.post(f"{BASE_URL}/predict", json=invalid_type_record)
    passed = r.status_code == 400 and "error" in r.json() and "details" in r.json()
    print_result("Test 5 - POST /predict invalid type", passed, r)
except Exception as e:
    print_result("Test 5 - POST /predict invalid type", False)
    print(e)
    print("-" * 60)
