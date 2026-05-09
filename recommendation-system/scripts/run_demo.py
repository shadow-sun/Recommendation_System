from src.services import build_demo_gateway


if __name__ == "__main__":
    gateway = build_demo_gateway()
    result = gateway.recommend("u1", 10)
    print(result)

