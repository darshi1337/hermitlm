#!/bin/bash

git pull

source venv/bin/activate

sudo systemctl restart hermitlm-api
sudo systemctl restart hermitlm-discord