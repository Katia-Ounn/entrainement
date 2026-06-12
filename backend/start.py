# start.py — Lancer le serveur avec timeout étendu pour le pipeline
# Usage : python start.py

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=300,   # 5 minutes — pipeline peut prendre du temps
        timeout_graceful_shutdown=30,
        workers=1,                # 1 seul worker pour partager le cache pipeline
    )
