#!/usr/bin/env python3

"""WebSocket server for interactive robot control via ROS pub/sub.

Commands:
- {"command": "task", ...} -> publishes to '/rollout_command'
- {"command": "start"} -> publishes "start" to '/record_command'
- {"command": "stop"} -> publishes "stop" to '/record_command'
"""

import asyncio
import json
import logging
import signal

import websockets
import rospy
from std_msgs.msg import String

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class InteractivePublisher:
    def __init__(self):
        # Disable ROS signal handlers so we can handle Ctrl+C ourselves
        rospy.init_node('interactive_publisher', anonymous=True, disable_signals=True)
        self._rollout_pub = rospy.Publisher('/rollout_command', String, queue_size=10)
        self._record_pub = rospy.Publisher('/record_command', String, queue_size=10)
    
    def publish_rollout(self, data: dict):
        msg = String()
        msg.data = json.dumps(data)
        self._rollout_pub.publish(msg)
        logger.info(f"Rollout: {data}")
    
    def publish_record(self, command: str):
        msg = String()
        msg.data = command
        self._record_pub.publish(msg)
        logger.info(f"Record: {command}")


async def main():
    # Create a future that will be set when we receive SIGINT/SIGTERM
    loop = asyncio.get_event_loop()
    stop = loop.create_future()
    
    def signal_handler():
        if not stop.done():
            stop.set_result(None)
    
    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)
    
    node = InteractivePublisher()
    
    print("Starting WebSocket server on ws://0.0.0.0:8765", flush=True)
    
    async def handler(ws):
        async for msg in ws:
            data = json.loads(msg)
            cmd = data.get("command")
            
            if cmd == "task":
                if not data.get("model_id") or not data.get("description") or not data.get("metadata"):
                    resp = {"status": "error", "message": "Missing model_id or description or metadata"}
                else:
                    node.publish_rollout(data)
                    resp = {"status": "ok", "message": "Rollout command published"}
            elif cmd == "start":
                metadata = data.get("metadata", "")
                node.publish_record(f"start:{metadata}")
                resp = {"status": "ok", "message": f"Record start:{metadata} published"}
            elif cmd == "stop":
                node.publish_record(cmd)
                resp = {"status": "ok", "message": f"Record {cmd} published"}
            else:
                resp = {"status": "error", "message": "Unknown command"}
            
            await ws.send(json.dumps(resp))
    
    async with websockets.serve(handler, "0.0.0.0", 8765, ping_timeout=None):
        await stop  # Wait for signal
        print("\nShutting down server...", flush=True)


if __name__ == "__main__":
    asyncio.run(main())