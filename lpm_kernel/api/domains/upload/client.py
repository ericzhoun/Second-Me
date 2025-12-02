import logging
from lpm_kernel.api.domains.upload.TrainingTags import TrainingTags
from lpm_kernel.configs.config import Config
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class HeartbeatConfig:
    """Heartbeat Configuration Class"""
    def __init__(
        self,
        interval: int = 30,
        timeout: int = 10,
        max_retries: int = 3,
        retry_interval: int = 5
    ):
        self.interval = interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_interval = retry_interval

class RegistryClient:
    """
    RegistryClient - Now disconnected from external servers.
    Most methods are no-ops or return dummy data.
    """
    def __init__(self, heartbeat_config: HeartbeatConfig = None):
        # We don't need server URL anymore
        self.server_url = None
        self.ws_url = None
        self.active_connections = {}
        self.heartbeat_config = heartbeat_config or HeartbeatConfig()

    def _get_auth_header(self):
        return {}

    def get_ws_url(self, instance_id: str, instance_password: str) -> str:
        return ""

    def register_upload(self, upload_name: str, instance_id: str = None, description: str = None, email: str = None, tags: TrainingTags = None):
        logger.info(f"Mocking register_upload for {upload_name}")
        return {
            "instance_id": instance_id or "local_instance",
            "upload_name": upload_name
        }

    def unregister_upload(self, instance_id: str):
        logger.info(f"Mocking unregister_upload for {instance_id}")
        return {"status": "success"}

    async def connect_websocket(self, instance_id: str, instance_password: str):
        logger.info("Mocking connect_websocket - doing nothing")
        return None
            
    async def _keep_alive(self, websocket, instance_id: str):
        pass

    async def _keep_alive_with_ping(self, websocket, instance_id: str):
        pass

    async def send_heartbeat(self, websocket):
        return True

    async def handle_messages(self, websocket):
        pass

    def list_uploads(self, page_no: int = 1, page_size: int = 10, status: Optional[List[str]] = None):
        logger.info("Mocking list_uploads")
        return {
            "total": 0,
            "items": []
        }

    def count_uploads(self):
        return {"count": 0}

    def get_upload_detail(self, instance_id: str) -> Dict:
        logger.info(f"Mocking get_upload_detail for {instance_id}")
        return None

    def update_upload(self, instance_id: str, upload_name: str = None, capabilities: dict = None, email: str = None):
        logger.info(f"Mocking update_upload for {instance_id}")
        return {"status": "success"}

    def create_role(self, role_id, name, description, system_prompt, icon, instance_id, is_active=True,
                   enable_l0_retrieval=True, enable_l1_retrieval=True):
        logger.info(f"Mocking create_role for {name}")
        return {"status": "success"}
