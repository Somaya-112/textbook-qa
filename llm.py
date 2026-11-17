"""LLM router: try each provider in turn so one outage or rate limit doesn't break the app."""
import logging
from openai import OpenAI
import config

log = logging.getLogger("llm")

def chat(messages, temperature=0.2):
    errors = []
    for p in config.PROVIDERS:
        try:
            client = OpenAI(api_key=p["key"], base_url=p["base_url"], timeout=30, max_retries=1)
            r = client.chat.completions.create(model=p["model"], messages=messages,
                                               temperature=temperature)
            return r.choices[0].message.content.strip()
        except Exception as e:  # rate limit, timeout, outage... go to the next provider
            log.warning("provider %s failed: %s", p["name"], e)
            errors.append(f"{p['name']}: {e}")
    raise RuntimeError("All LLM providers failed: " + "; ".join(errors))
