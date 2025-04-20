"""
Load testing for SQL Chatbot application using Locust.
Run directly or through the CI/CD pipeline.

To run manually:
- Install locust: pip install locust
- Run: locust -f tests/performance/load_test.py
- Open browser at http://localhost:8089
"""

import json
import os
import sys
import time
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from locust import HttpUser, task, between
    locust_available = True
except ImportError:
    # Create a mock implementation if Locust is not installed
    # This allows the module to be imported without Locust installed
    locust_available = False

    class HttpUser:
        """Mock HttpUser class if Locust is not available."""
        
        def __init__(self, *args, **kwargs):
            pass
            
    def task(weight=1):
        """Mock task decorator if Locust is not available."""
        def decorator(func):
            return func
        return decorator
        
    def between(min_wait, max_wait):
        """Mock between function if Locust is not available."""
        def decorator(cls):
            return cls
        return decorator


# Define test scenarios
class SQLChatbotUser(HttpUser):
    """Simulated user for load testing the SQL Chatbot API."""
    
    # Wait between 1 and 5 seconds between tasks
    wait_time = between(1, 5)
    
    def on_start(self):
        """Initialize user session - login if needed."""
        self.login()
        
    def login(self):
        """Log in to get authentication token."""
        credentials = {
            "username": f"test_user_{int(time.time())}",
            "email": f"test_{int(time.time())}@example.com",
            "password": "testpassword123"
        }
        
        # Try to register first, if fails just login
        reg_response = self.client.post("/api/auth/register", json=credentials)
        if reg_response.status_code != 200:
            login_data = {
                "username": credentials["username"],
                "password": credentials["password"]
            }
            response = self.client.post("/api/auth/login", json=login_data)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("token"):
                    self.token = data["token"]
                    self.client.headers.update({"Authorization": f"Bearer {self.token}"})
            
    @task(1)
    def get_tables(self):
        """Test retrieving table list."""
        self.client.get("/api/tables")
        
    @task(2)
    def get_schema(self):
        """Test retrieving schema information."""
        self.client.post("/api/schema", json={"table_name": None})
        
    @task(5)
    def query_simple(self):
        """Test simple query execution."""
        query_data = {
            "question": "Show me all employees"
        }
        self.client.post("/api/query", json=query_data)
        
    @task(3)
    def query_complex(self):
        """Test complex query with aggregation."""
        query_data = {
            "question": "What is the average salary by department?"
        }
        self.client.post("/api/query", json=query_data)


def run_standalone_test():
    """Run a simple standalone performance test without Locust."""
    import requests
    import statistics
    
    base_url = "http://localhost:8000"
    results = {
        "get_tables": [],
        "get_schema": [],
        "simple_query": [],
        "complex_query": []
    }
    
    # Number of iterations
    iterations = 10
    
    print("Running standalone performance test")
    
    # Get tables test
    print(f"Testing GET /api/tables ({iterations} iterations)")
    for i in range(iterations):
        start_time = time.time()
        response = requests.get(f"{base_url}/api/tables")
        end_time = time.time()
        if response.status_code == 200:
            results["get_tables"].append((end_time - start_time) * 1000)  # ms
            
    # Get schema test
    print(f"Testing POST /api/schema ({iterations} iterations)")
    for i in range(iterations):
        start_time = time.time()
        response = requests.post(f"{base_url}/api/schema", json={"table_name": None})
        end_time = time.time()
        if response.status_code == 200:
            results["get_schema"].append((end_time - start_time) * 1000)  # ms
            
    # Simple query test
    print(f"Testing simple query ({iterations} iterations)")
    for i in range(iterations):
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/query", 
            json={"question": "Show me all employees"}
        )
        end_time = time.time()
        if response.status_code == 200:
            results["simple_query"].append((end_time - start_time) * 1000)  # ms
            
    # Complex query test
    print(f"Testing complex query ({iterations} iterations)")
    for i in range(iterations):
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/query", 
            json={"question": "What is the average salary by department?"}
        )
        end_time = time.time()
        if response.status_code == 200:
            results["complex_query"].append((end_time - start_time) * 1000)  # ms
    
    # Generate report
    report = {
        "summary": {},
        "details": results
    }
    
    for test_name, times in results.items():
        if times:
            report["summary"][test_name] = {
                "min_ms": min(times),
                "max_ms": max(times),
                "avg_ms": sum(times) / len(times),
                "median_ms": statistics.median(times),
                "p95_ms": sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else max(times),
                "count": len(times)
            }
        else:
            report["summary"][test_name] = {"error": "No successful requests"}
    
    # Write report to file
    with open("performance-report.html", "w") as f:
        f.write("<html><head><title>SQL Chatbot Performance Report</title>")
        f.write("<style>body{font-family:sans-serif;margin:20px;}")
        f.write("table{border-collapse:collapse;width:100%;margin-bottom:20px;}")
        f.write("th,td{border:1px solid #ddd;padding:8px;text-align:left;}")
        f.write("th{background-color:#f2f2f2;}")
        f.write("tr:nth-child(even){background-color:#f9f9f9;}")
        f.write("</style></head><body>")
        f.write("<h1>SQL Chatbot Performance Test Report</h1>")
        f.write("<h2>Summary</h2>")
        f.write("<table><tr><th>Test</th><th>Min (ms)</th><th>Max (ms)</th>")
        f.write("<th>Avg (ms)</th><th>Median (ms)</th><th>P95 (ms)</th><th>Count</th></tr>")
        
        for test_name, metrics in report["summary"].items():
            if "error" not in metrics:
                f.write(f"<tr><td>{test_name}</td>")
                f.write(f"<td>{metrics['min_ms']:.2f}</td>")
                f.write(f"<td>{metrics['max_ms']:.2f}</td>")
                f.write(f"<td>{metrics['avg_ms']:.2f}</td>")
                f.write(f"<td>{metrics['median_ms']:.2f}</td>")
                f.write(f"<td>{metrics['p95_ms']:.2f}</td>")
                f.write(f"<td>{metrics['count']}</td></tr>")
            else:
                f.write(f"<tr><td>{test_name}</td><td colspan='6'>{metrics['error']}</td></tr>")
        
        f.write("</table>")
        f.write("<h2>Detailed Results</h2>")
        
        for test_name, times in results.items():
            if times:
                f.write(f"<h3>{test_name}</h3>")
                f.write("<table><tr><th>#</th><th>Time (ms)</th></tr>")
                for i, t in enumerate(times):
                    f.write(f"<tr><td>{i+1}</td><td>{t:.2f}</td></tr>")
                f.write("</table>")
        
        f.write("</body></html>")
    
    print(f"Performance report generated: performance-report.html")
    print("\nPerformance Test Results Summary:")
    for test_name, metrics in report["summary"].items():
        if "error" not in metrics:
            print(f"  {test_name}:")
            print(f"    - Average: {metrics['avg_ms']:.2f} ms")
            print(f"    - Median: {metrics['median_ms']:.2f} ms")
            print(f"    - Min/Max: {metrics['min_ms']:.2f}/{metrics['max_ms']:.2f} ms")
        else:
            print(f"  {test_name}: {metrics['error']}")


if __name__ == "__main__":
    # If run directly and Locust is not available, run standalone test
    if not locust_available:
        run_standalone_test()
    else:
        # Locust is expected to import this file, not run it directly
        pass