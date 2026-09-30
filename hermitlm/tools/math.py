import os

import requests
from dotenv import load_dotenv

load_dotenv()

APP_ID = os.getenv("WOLFRAM_APP_ID")


def ask_wolfram(query):
    try:
        url = "https://api.wolframalpha.com/v2/query"

        params = {
            "appid": APP_ID,
            "input": query,
            "format": "plaintext",
            "output": "json",
        }

        r = requests.get(url, params=params, timeout=15)

        if r.status_code != 200:
            print("WOLFRAM STATUS:", r.status_code)
            print("WOLFRAM BODY:", r.text)
            return None

        data = r.json()

        result = data.get("queryresult", {})

        if not result.get("success"):
            print("WOLFRAM QUERY FAILED")
            return None

        pods = result.get("pods", [])

        for pod in pods:
            if pod.get("primary", False):
                subpods = pod.get("subpods", [])

                if subpods:
                    text = subpods[0].get("plaintext")

                    if text:
                        return text

        for pod in pods:
            title = pod.get("title", "").lower()

            if title in ["result", "solution", "solutions"]:
                subpods = pod.get("subpods", [])

                if subpods:
                    text = subpods[0].get("plaintext")

                    if text:
                        return text

        for pod in pods:
            subpods = pod.get("subpods", [])

            if subpods:
                text = subpods[0].get("plaintext")

                if text:
                    return text

        return None

    except Exception as e:  # noqa: BLE001
        print("WOLFRAM ERROR:", e)
        return None