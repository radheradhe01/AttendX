"""
Authentication service for user management and JWT tokens
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from loguru import logger

from app.models.user import User
from app.schemas.user import TokenData
import os

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Service for authentication operations"""
    
    def __init__(self):
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
        self.access_token_expire_minutes = ACCESS_TOKEN_EXPIRE_MINUTES
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password matches, False otherwise
        """
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """
        Hash a password
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT access token
        
        Args:
            data: Data to encode in token
            expires_delta: Token expiration time
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            TokenData if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            
            if username is None:
                return None
                
            token_data = TokenData(username=username)
            return token_data
            
        except JWTError as e:
            logger.error(f"JWT verification error: {e}")
            return None
    
    def authenticate_user(self, db: Session, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username and password
        
        Args:
            db: Database session
            username: Username or email
            password: Plain text password
            
        Returns:
            User object if authenticated, None otherwise
        """
        try:
            # Try to find user by username or email
            user = db.query(User).filter(
                (User.username == username) | (User.email == username)
            ).first()
            
            if not user:
                logger.warning(f"User not found: {username}")
                return None
                
            if not self.verify_password(password, user.hashed_password):
                logger.warning(f"Invalid password for user: {username}")
                return None
                
            if not user.is_active:
                logger.warning(f"Inactive user attempted login: {username}")
                return None
                
            # Update last login
            user.last_login = datetime.utcnow()
            db.commit()
            
            logger.info(f"User authenticated successfully: {username}")
            return user
            
        except Exception as e:
            logger.error(f"Authentication error for user {username}: {e}")
            return None
    
    def get_current_user(self, db: Session, token: str) -> Optional[User]:
        """
        Get current user from JWT token
        
        Args:
            db: Database session
            token: JWT token
            
        Returns:
            User object if valid, None otherwise
        """
        try:
            token_data = self.verify_token(token)
            if token_data is None:
                return None
                
            user = db.query(User).filter(User.username == token_data.username).first()
            if user is None:
                return None
                
            return user
            
        except Exception as e:
            logger.error(f"Error getting current user: {e}")
            return None
    
    def create_user_token(self, user: User) -> Dict[str, str]:
        """
        Create access token for user
        
        Args:
            user: User object
            
        Returns:
            Dictionary with token information
        """
        access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
        access_token = self.create_access_token(
            data={"sub": user.username, "user_id": user.id, "role": user.role},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60
        }
    
    def refresh_token(self, token: str) -> Optional[Dict[str, str]]:
        """
        Refresh access token
        
        Args:
            token: Current JWT token
            
        Returns:
            New token data or None if invalid
        """
        try:
            token_data = self.verify_token(token)
            if token_data is None:
                return None
                
            # Create new token with same data
            access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
            new_token = self.create_access_token(
                data={"sub": token_data.username},
                expires_delta=access_token_expires
            )
            
            return {
                "access_token": new_token,
                "token_type": "bearer",
                "expires_in": self.access_token_expire_minutes * 60
            }
            
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return None
    
    def change_password(self, db: Session, user: User, current_password: str, new_password: str) -> bool:
        """
        Change user password
        
        Args:
            db: Database session
            user: User object
            current_password: Current password
            new_password: New password
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify current password
            if not self.verify_password(current_password, user.hashed_password):
                logger.warning(f"Invalid current password for user: {user.username}")
                return False
                
            # Hash new password
            new_hashed_password = self.get_password_hash(new_password)
            
            # Update user
            user.hashed_password = new_hashed_password
            user.updated_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"Password changed successfully for user: {user.username}")
            return True
            
        except Exception as e:
            logger.error(f"Error changing password for user {user.username}: {e}")
            db.rollback()
            return False
    
    def reset_password(self, db: Session, user: User, new_password: str) -> bool:
        """
        Reset user password (admin function)
        
        Args:
            db: Database session
            user: User object
            new_password: New password
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Hash new password
            new_hashed_password = self.get_password_hash(new_password)
            
            # Update user
            user.hashed_password = new_hashed_password
            user.updated_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"Password reset successfully for user: {user.username}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting password for user {user.username}: {e}")
            db.rollback()
            return False
