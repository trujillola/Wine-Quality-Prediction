from fastapi import FastAPI
from objects.launcher import Launcher

# app.include_router(...)

# Mettre ce code dans un fichier launcher
launcher = Launcher()
launcher.launch()