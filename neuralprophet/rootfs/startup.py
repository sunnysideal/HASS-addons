import os
from homeassistantapi import HomeAssistantAPI

if not os.path.exists("/config/neuralprophet.yaml"):
  print("Copy template config file")
  os.system("cp /neuralprophet.yaml /config")
  
os.system("python3 /ha_predictor.py")
