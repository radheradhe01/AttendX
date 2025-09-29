import dotenv from 'dotenv';

// Load test environment variables
dotenv.config({ path: '.env.test' });

// Set test environment
process.env.NODE_ENV = 'test';
process.env.MONGODB_URI = process.env.MONGODB_TEST_URI || 'mongodb://localhost:27017/attendex_erp_test';

// Global test setup
beforeAll(async () => {
  // Global setup for all tests
});

afterAll(async () => {
  // Global cleanup for all tests
});

// Increase timeout for database operations
jest.setTimeout(30000);
