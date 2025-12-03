import string
import secrets
import logging
import asyncio
import time
from flask import Blueprint, request, jsonify
from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.common.repository.database_session import DatabaseSession
from lpm_kernel.models.load import Load
from lpm_kernel.api.domains.loads.load_service import LoadService
from .client import RegistryClient
import threading
from lpm_kernel.api.domains.loads.dto import LoadDTO
from lpm_kernel.api.domains.trainprocess.training_params_manager import TrainingParamsManager
from lpm_kernel.file_data.document_service import document_service
from lpm_kernel.api.domains.upload.TrainingTags import TrainingTags

upload_bp = Blueprint("upload", __name__)
# Registry client is now disabled/dummy
registry_client = RegistryClient()

logger = logging.getLogger(__name__)

@upload_bp.route("/api/upload/register", methods=["POST"])
def register_upload():
    """Register upload instance - LOCAL ONLY"""
    try:
        current_load, error, status_code = LoadService.get_current_load()
        if error:
            return jsonify(APIResponse.error(code=status_code, message=error))
        
        # Simulating a successful registration locally without external calls
        result = {
            "instance_id": current_load.instance_id,
            "upload_name": current_load.name,
            "status": "registered (local)"
        }

        # We don't generate a new password or update credentials from remote
        
        return jsonify(APIResponse.success(
            data=result,
            message="Upload registered locally (remote disabled)"
        ))
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(
            code=500, message=f"An error occurred: {str(e)}"
        ))

@upload_bp.route("/api/upload/connect", methods=["POST"])
async def connect_upload():
    """
    Establish WebSocket connection for the specified Upload instance - DISABLED
    """
    try:
        logger.info("WebSocket connection requested but remote connection is disabled.")
        current_load, error, status_code = LoadService.get_current_load(with_password=True)
        if error:
            return jsonify(APIResponse.error(
                code=status_code, message=error
            ))
            
        result = {
            "instance_id": current_load.instance_id,
            "upload_name": current_load.name
        }
        
        return jsonify(APIResponse.success(
            data=result,
            message="WebSocket connection disabled (local mode only)"
        ))
        
    except Exception as e:
        logger.error(f"Failed to establish WebSocket connection: {str(e)}")
        return jsonify(APIResponse.error(
            message=f"Failed to establish WebSocket connection: {str(e)}",
            code=500
        ))

@upload_bp.route("/api/upload/status", methods=["GET"])
def get_upload_status():
    """
    Get the status of the specified Upload instance - LOCAL ONLY
    """
    try:
        current_load, error, status_code = LoadService.get_current_load()
        if error:
            return jsonify(APIResponse.error(
                code=status_code, message=error
            ))
        
        instance_id = current_load.instance_id
        
        # Only return local data
        upload_data = {
            "upload_name": current_load.name,
            "instance_id": instance_id,
            "description": current_load.description,
            "email": current_load.email,
            "status": "offline", # Always offline as no remote connection
            "last_heartbeat": None,
            "is_connected": False,
            "last_ws_check": None
        }
        
        return jsonify(APIResponse.success(
            data=upload_data,
            message="Successfully retrieved Upload instance status (local)"
        ))
            
    except Exception as e:
        logger.error(f"Failed to get Upload instance status: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(
            message=f"Failed to get status: {str(e)}",
            code=500
        ))

@upload_bp.route("/api/upload", methods=["DELETE"])
def unregister_upload():
    """
    API for unregistering Upload instance - LOCAL ONLY
    """
    try:
        current_load, error, status_code = LoadService.get_current_load()
        instance_id = current_load.instance_id

        # No remote call to unregister
        
        return jsonify(APIResponse.success(
            data={
                "instance_id": instance_id,
                "upload_name": current_load.name
            },
            message="Upload instance unregistered locally"
        ))
            
    except Exception as e:
        logger.error(f"Failed to unregister Upload: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(
            message=f"Unregistration failed: {str(e)}",
            code=500
        ))

@upload_bp.route("/api/upload", methods=["GET"])
def list_uploads():
    """
    List registered Upload instances - LOCAL ONLY (Mock/Empty)
    """
    try:
        # Return empty list or just current load if we want to simulate
        # But usually this endpoint lists ALL uploads from registry.
        # Since we are disconnected, we return empty list or just valid structure.
        
        result = {
            "total": 0,
            "items": []
        }
        
        return jsonify(APIResponse.success(
            data=result,
            message="Successfully retrieved Upload list (local empty)"
        ))
        
    except Exception as e:
        logger.error(f"Failed to get Upload list: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(
            message=f"Failed to get list: {str(e)}",
            code=500
        ))

@upload_bp.route("/api/upload/count", methods=["GET"])
def count_uploads():
    """
    Get the number of registered Upload instances - LOCAL ONLY
    """
    try:
        result = {"count": 0}
        
        return jsonify(APIResponse.success(
            data=result,
            message="Successfully retrieved Upload count (local)"
        ))
        
    except Exception as e:
        logger.error(f"Failed to get Upload count: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(
            message=f"Failed to get count: {str(e)}",
            code=500
        ))

@upload_bp.route("/api/upload", methods=["PUT"])
def update_upload():
    """
    API for updating Upload instance information - LOCAL ONLY
    """
    try:
        current_load, error, status_code = LoadService.get_current_load()
        if error:
             return jsonify(APIResponse.error(code=status_code, message=error))

        instance_id = current_load.instance_id
        
        data = request.get_json()
        if not data:
            return jsonify(APIResponse.error(
                message="Request body cannot be empty",
                code=400
            ))
        
        # We could technically update local Load info here if we wanted to support local updates via this API
        # But this API was meant for Registry updates.
        # For now, just return success.
        
        result = {
            "instance_id": instance_id,
            "status": "updated (local)"
        }
        
        return jsonify(APIResponse.success(
            data=result,
            message="Upload instance updated successfully (local)"
        ))
            
        
    except Exception as e:
        logger.error(f"Failed to update Upload: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(
            message=f"Update failed: {str(e)}",
            code=500
        ))
