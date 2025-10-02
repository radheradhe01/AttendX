import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { Request } from 'express';
import { logger } from '../config/logger';
import { CustomError } from './errorHandler';

/**
 * File upload middleware configuration
 * Handles image uploads for profile pictures and face recognition
 */

// Ensure upload directory exists
const uploadDir = process.env.UPLOAD_PATH || './uploads';
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// Create subdirectories for different file types
const profilePicturesDir = path.join(uploadDir, 'profile-pictures');
const faceImagesDir = path.join(uploadDir, 'face-images');
const reportsDir = path.join(uploadDir, 'reports');

[profilePicturesDir, faceImagesDir, reportsDir].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// File filter function
const fileFilter = (req: Request, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
  const allowedTypes = (process.env.ALLOWED_FILE_TYPES || 'image/jpeg,image/png,image/jpg').split(',');
  
  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new CustomError(`File type ${file.mimetype} is not allowed`, 400, 'INVALID_FILE_TYPE'));
  }
};

// Storage configuration
const storage = multer.diskStorage({
  destination: (req: Request, file: Express.Multer.File, cb) => {
    let uploadPath = uploadDir;
    
    // Determine upload path based on field name
    if (file.fieldname === 'profilePicture') {
      uploadPath = profilePicturesDir;
    } else if (file.fieldname === 'faceImage') {
      uploadPath = faceImagesDir;
    } else if (file.fieldname === 'report') {
      uploadPath = reportsDir;
    }
    
    cb(null, uploadPath);
  },
  filename: (req: Request, file: Express.Multer.File, cb) => {
    // Generate unique filename with timestamp and user ID
    const userId = req.user?.id || 'anonymous';
    const timestamp = Date.now();
    const extension = path.extname(file.originalname);
    const filename = `${userId}_${timestamp}${extension}`;
    
    cb(null, filename);
  }
});

// Multer configuration
const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: parseInt(process.env.MAX_FILE_SIZE || '5242880'), // 5MB default
    files: 1, // Only one file at a time
  },
});

/**
 * Middleware for profile picture upload
 */
export const uploadProfilePicture = upload.single('profilePicture');

/**
 * Middleware for face image upload
 */
export const uploadFaceImage = upload.single('faceImage');

/**
 * Middleware for report file upload
 */
export const uploadReport = upload.single('report');

/**
 * Middleware for multiple file uploads
 */
export const uploadMultiple = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: parseInt(process.env.MAX_FILE_SIZE || '5242880'),
    files: 5, // Maximum 5 files
  },
}).array('files', 5);

/**
 * Error handler for multer errors
 */
export const handleUploadError = (error: any, req: Request, res: any, next: any) => {
  if (error instanceof multer.MulterError) {
    switch (error.code) {
      case 'LIMIT_FILE_SIZE':
        return res.status(400).json({
          success: false,
          message: 'File size too large. Maximum size is 5MB.',
          code: 'FILE_TOO_LARGE'
        });
      case 'LIMIT_FILE_COUNT':
        return res.status(400).json({
          success: false,
          message: 'Too many files. Maximum 5 files allowed.',
          code: 'TOO_MANY_FILES'
        });
      case 'LIMIT_UNEXPECTED_FILE':
        return res.status(400).json({
          success: false,
          message: 'Unexpected file field.',
          code: 'UNEXPECTED_FILE'
        });
      default:
        return res.status(400).json({
          success: false,
          message: 'File upload error.',
          code: 'UPLOAD_ERROR'
        });
    }
  }
  
  if (error instanceof CustomError) {
    return res.status(error.statusCode || 400).json({
      success: false,
      message: error.message,
      code: error.code
    });
  }
  
  next(error);
};

/**
 * Utility function to delete uploaded file
 */
export const deleteUploadedFile = (filePath: string): Promise<void> => {
  return new Promise((resolve, reject) => {
    fs.unlink(filePath, (error) => {
      if (error) {
        logger.error('Error deleting file:', error);
        reject(error);
      } else {
        logger.info('File deleted successfully:', filePath);
        resolve();
      }
    });
  });
};

/**
 * Utility function to get file URL
 */
export const getFileUrl = (filePath: string): string => {
  const relativePath = path.relative(process.cwd(), filePath);
  return `/uploads/${relativePath.replace(/\\/g, '/')}`;
};

/**
 * Utility function to validate image file
 */
export const validateImageFile = (file: Express.Multer.File): boolean => {
  const allowedMimeTypes = ['image/jpeg', 'image/png', 'image/jpg'];
  const maxSize = parseInt(process.env.MAX_FILE_SIZE || '5242880'); // 5MB
  
  if (!allowedMimeTypes.includes(file.mimetype)) {
    return false;
  }
  
  if (file.size > maxSize) {
    return false;
  }
  
  return true;
};

/**
 * Utility function to get file info
 */
export const getFileInfo = (file: Express.Multer.File) => {
  return {
    originalName: file.originalname,
    filename: file.filename,
    mimetype: file.mimetype,
    size: file.size,
    path: file.path,
    url: getFileUrl(file.path),
  };
};
