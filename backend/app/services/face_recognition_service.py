"""
Face recognition service for attendance verification
"""

import face_recognition
import cv2
import numpy as np
import base64
import json
from typing import List, Tuple, Optional, Dict
from PIL import Image
import io
import os
from loguru import logger


class FaceRecognitionService:
    """Service for face recognition operations"""
    
    def __init__(self):
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_ids: List[int] = []
        self.tolerance = 0.6  # Lower is more strict
        
    async def initialize(self):
        """Initialize the face recognition service"""
        logger.info("Initializing face recognition service...")
        # This will be populated when users register
        self.known_face_encodings = []
        self.known_face_ids = []
        
    def encode_face_from_image(self, image_data: str) -> Optional[np.ndarray]:
        """
        Encode face from base64 image data
        
        Args:
            image_data: Base64 encoded image string
            
        Returns:
            Face encoding as numpy array or None if no face found
        """
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(image_array)
            
            if not face_locations:
                logger.warning("No face found in image")
                return None
                
            if len(face_locations) > 1:
                logger.warning("Multiple faces found in image, using the first one")
                
            # Get face encoding
            face_encodings = face_recognition.face_encodings(
                image_array, 
                face_locations
            )
            
            if face_encodings:
                return face_encodings[0]
            else:
                logger.warning("Could not encode face")
                return None
                
        except Exception as e:
            logger.error(f"Error encoding face: {e}")
            return None
    
    def encode_face_from_file(self, image_path: str) -> Optional[np.ndarray]:
        """
        Encode face from image file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            Face encoding as numpy array or None if no face found
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
                
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                logger.warning(f"No face found in image: {image_path}")
                return None
                
            if len(face_locations) > 1:
                logger.warning(f"Multiple faces found in image: {image_path}")
                
            # Get face encoding
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if face_encodings:
                return face_encodings[0]
            else:
                logger.warning(f"Could not encode face from: {image_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error encoding face from file {image_path}: {e}")
            return None
    
    def add_known_face(self, user_id: int, face_encoding: np.ndarray):
        """
        Add a known face to the recognition system
        
        Args:
            user_id: User ID
            face_encoding: Face encoding numpy array
        """
        self.known_face_encodings.append(face_encoding)
        self.known_face_ids.append(user_id)
        logger.info(f"Added face for user {user_id}")
    
    def remove_known_face(self, user_id: int):
        """
        Remove a known face from the recognition system
        
        Args:
            user_id: User ID to remove
        """
        try:
            index = self.known_face_ids.index(user_id)
            self.known_face_encodings.pop(index)
            self.known_face_ids.pop(index)
            logger.info(f"Removed face for user {user_id}")
        except ValueError:
            logger.warning(f"User {user_id} not found in known faces")
    
    def recognize_face(self, face_encoding: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Recognize a face from encoding
        
        Args:
            face_encoding: Face encoding to recognize
            
        Returns:
            Tuple of (user_id, confidence_score) or (None, 0.0) if not recognized
        """
        if not self.known_face_encodings:
            logger.warning("No known faces in system")
            return None, 0.0
            
        try:
            # Compare with known faces
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, 
                face_encoding
            )
            
            # Find the best match
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            
            # Convert distance to confidence (lower distance = higher confidence)
            confidence = max(0, 1 - best_distance)
            
            # Check if confidence meets threshold
            if confidence >= (1 - self.tolerance):
                user_id = self.known_face_ids[best_match_index]
                logger.info(f"Face recognized as user {user_id} with confidence {confidence:.3f}")
                return user_id, confidence
            else:
                logger.info(f"Face not recognized. Best confidence: {confidence:.3f}")
                return None, confidence
                
        except Exception as e:
            logger.error(f"Error recognizing face: {e}")
            return None, 0.0
    
    def verify_attendance(self, image_data: str) -> Tuple[Optional[int], float, bool]:
        """
        Verify attendance by recognizing face in image
        
        Args:
            image_data: Base64 encoded image string
            
        Returns:
            Tuple of (user_id, confidence, is_verified)
        """
        try:
            # Encode face from image
            face_encoding = self.encode_face_from_image(image_data)
            
            if face_encoding is None:
                return None, 0.0, False
                
            # Recognize face
            user_id, confidence = self.recognize_face(face_encoding)
            
            # Determine if verified (confidence above threshold)
            is_verified = user_id is not None and confidence >= (1 - self.tolerance)
            
            return user_id, confidence, is_verified
            
        except Exception as e:
            logger.error(f"Error verifying attendance: {e}")
            return None, 0.0, False
    
    def get_face_encoding_json(self, face_encoding: np.ndarray) -> str:
        """
        Convert face encoding to JSON string for database storage
        
        Args:
            face_encoding: Face encoding numpy array
            
        Returns:
            JSON string representation
        """
        return json.dumps(face_encoding.tolist())
    
    def load_face_encoding_from_json(self, json_string: str) -> Optional[np.ndarray]:
        """
        Load face encoding from JSON string
        
        Args:
            json_string: JSON string representation
            
        Returns:
            Face encoding numpy array or None if invalid
        """
        try:
            encoding_list = json.loads(json_string)
            return np.array(encoding_list)
        except Exception as e:
            logger.error(f"Error loading face encoding from JSON: {e}")
            return None
    
    def update_tolerance(self, tolerance: float):
        """
        Update recognition tolerance
        
        Args:
            tolerance: New tolerance value (0.0 to 1.0)
        """
        self.tolerance = max(0.0, min(1.0, tolerance))
        logger.info(f"Updated face recognition tolerance to {self.tolerance}")
    
    def get_stats(self) -> Dict:
        """
        Get face recognition service statistics
        
        Returns:
            Dictionary with service stats
        """
        return {
            "known_faces_count": len(self.known_face_ids),
            "tolerance": self.tolerance,
            "user_ids": self.known_face_ids.copy()
        }
