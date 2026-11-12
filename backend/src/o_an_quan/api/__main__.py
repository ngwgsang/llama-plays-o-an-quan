"""Run the API server: python -m o_an_quan.api [--host 127.0.0.1] [--port 8000] [--reload]"""
import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(prog="python -m o_an_quan.api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("o_an_quan.api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
