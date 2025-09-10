# TODO
"""
# pip install redis
# and run a local Redis, then:
from langgraph.checkpoint.redis import RedisSaver
from redis import Redis


redis = Redis(host="localhost", port=6379, decode_responses=True)
checkpointer = RedisSaver(redis, namespace="lg-demo")
# Use: app = graph.compile(checkpointer=checkpointer)
"""