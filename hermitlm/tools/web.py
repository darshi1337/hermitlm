from ddgs import DDGS

def web_search(query, max_results=3):
    try:
        results_text = []

        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)

            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")

                results_text.append(
                    f"{title}: {body}"
                )

        return "\n".join(results_text)

    except Exception as e:
        print("WEB SEARCH ERROR:", e)
        return None