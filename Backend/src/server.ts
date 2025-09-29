import dotenv from 'dotenv';
import app from './app';
import { connectDB } from './config/db';
import { logger } from './config/logger';

// Load environment variables
dotenv.config();

/**
 * Server Entry Point
 * Initializes database connection and starts the Express server
 */

const PORT = parseInt(process.env.PORT || '3000', 10);
const HOST = process.env.HOST || 'localhost';

// Connect to database
connectDB()
  .then(() => {
    // Start server
    app.listen(PORT, HOST, () => {
      logger.info(`🚀 AttendEase API Server running on http://${HOST}:${PORT}`);
      logger.info(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
      logger.info(`🔗 Health check: http://${HOST}:${PORT}/health`);
    });
  })
  .catch((error) => {
    logger.error('Failed to start server:', error);
    process.exit(1);
  });

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received. Shutting down gracefully...');
  process.exit(0);
});

process.on('SIGINT', () => {
  logger.info('SIGINT received. Shutting down gracefully...');
  process.exit(0);
});
