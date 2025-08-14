import requests
import sys
import json
from datetime import datetime

class KalmanAPITester:
    def __init__(self, base_url="https://ts-py-options.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.run_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=30)

            print(f"   Status Code: {response.status_code}")
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response preview: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error response: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"   Error text: {response.text[:200]}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_hello_world(self):
        """Test the root endpoint"""
        success, response = self.run_test(
            "Hello World Endpoint",
            "GET",
            "",
            200
        )
        if success and response.get('message') == 'Hello World':
            print("   ✓ Message content verified")
            return True
        elif success:
            print(f"   ⚠️  Unexpected message: {response.get('message')}")
            return False
        return False

    def test_simulate(self):
        """Test the simulate endpoint with the specified payload"""
        payload = {
            "n": 120,
            "S0": 100,
            "mu": 0.05,
            "sigma_true": 0.2,
            "K": None,  # should default to S0
            "r": 0.02,
            "T": 0.25,
            "dt": 0.003968,
            "obs_noise_std": 0.5,
            "seed": 42,
            "save": True
        }
        
        success, response = self.run_test(
            "Kalman Simulate",
            "POST",
            "kalman/simulate",
            200,
            data=payload
        )
        
        if success:
            # Verify required fields
            if 'run_id' in response:
                self.run_id = response['run_id']
                print(f"   ✓ Run ID obtained: {self.run_id}")
            else:
                print("   ❌ Missing run_id in response")
                return False
                
            # Check array lengths
            arrays_to_check = ['S', 'true_vol', 'call_price_clean', 'call_price_obs']
            for arr_name in arrays_to_check:
                if arr_name in response:
                    arr_len = len(response[arr_name])
                    if arr_len == 120:
                        print(f"   ✓ {arr_name} length correct: {arr_len}")
                    else:
                        print(f"   ❌ {arr_name} length incorrect: {arr_len} (expected 120)")
                        return False
                else:
                    print(f"   ❌ Missing {arr_name} in response")
                    return False
            
            return True
        return False

    def test_fit(self):
        """Test the fit endpoint using the run_id from simulate"""
        if not self.run_id:
            print("❌ Cannot test fit - no run_id available")
            return False
            
        payload = {
            "run_id": self.run_id,
            "sigma_init": 0.2,
            "process_var": 0.0001
        }
        
        success, response = self.run_test(
            "Kalman Fit",
            "POST",
            "kalman/fit",
            200,
            data=payload
        )
        
        if success:
            # Check required fields
            required_fields = ['est_vol', 'call_price_est']
            for field in required_fields:
                if field in response:
                    arr_len = len(response[field])
                    if arr_len == 120:
                        print(f"   ✓ {field} length correct: {arr_len}")
                    else:
                        print(f"   ❌ {field} length incorrect: {arr_len} (expected 120)")
                        return False
                else:
                    print(f"   ❌ Missing {field} in response")
                    return False
            
            # Check that we have numeric data
            if response.get('est_vol') and len(response['est_vol']) > 0:
                first_vol = response['est_vol'][0]
                if isinstance(first_vol, (int, float)):
                    print(f"   ✓ Estimated volatility data looks valid: {first_vol}")
                else:
                    print(f"   ❌ Invalid volatility data type: {type(first_vol)}")
                    return False
            
            return True
        return False

def main():
    print("🚀 Starting Kalman Filter API Tests")
    print("=" * 50)
    
    # Setup
    tester = KalmanAPITester()
    
    # Test 1: Hello World
    print("\n📍 TEST 1: Hello World Endpoint")
    test1_success = tester.test_hello_world()
    
    # Test 2: Simulate
    print("\n📍 TEST 2: Kalman Simulate")
    test2_success = tester.test_simulate()
    
    # Test 3: Fit (only if simulate succeeded)
    print("\n📍 TEST 3: Kalman Fit")
    test3_success = tester.test_fit()
    
    # Print final results
    print("\n" + "=" * 50)
    print(f"📊 FINAL RESULTS:")
    print(f"   Tests Run: {tester.tests_run}")
    print(f"   Tests Passed: {tester.tests_passed}")
    print(f"   Success Rate: {(tester.tests_passed/tester.tests_run)*100:.1f}%")
    
    if test1_success and test2_success and test3_success:
        print("🎉 ALL TESTS PASSED - Backend APIs are working correctly!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Check the details above")
        return 1

if __name__ == "__main__":
    sys.exit(main())