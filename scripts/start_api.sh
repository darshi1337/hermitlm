#!/bin/bash

uvicorn hermitlm.runtime.api:app --host 0.0.0.0 --port 8000