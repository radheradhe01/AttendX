'use client';

import { useAuth } from '@/hooks/useAuth';
import { authApi } from '@/lib/api';
import { useState } from 'react';

export default function AuthTest() {
  const { user, isAuthenticated, isLoading, login, logout, token } = useAuth();
  const [testResult, setTestResult] = useState<string>('');
  const [isTesting, setIsTesting] = useState(false);

  const handleLogin = async () => {
    const result = await login('test@example.com', 'Test123456');
    if (result.success) {
      setTestResult('Login successful!');
    } else {
      setTestResult(`Login failed: ${result.error}`);
    }
  };

  const testApiCall = async () => {
    setIsTesting(true);
    try {
      const response = await authApi.getProfile();
      setTestResult(`API call successful: ${JSON.stringify(response.data, null, 2)}`);
    } catch (error) {
      setTestResult(`API call failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsTesting(false);
    }
  };

  if (isLoading) {
    return <div className="p-4">Loading...</div>;
  }

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Authentication Test</h2>
      
      <div className="space-y-4">
        <div>
          <h3 className="text-lg font-semibold">Status:</h3>
          <p>Authenticated: {isAuthenticated ? 'Yes' : 'No'}</p>
          <p>Token: {token ? 'Present' : 'Missing'}</p>
          {user && (
            <div>
              <p>User ID: {user.id}</p>
              <p>Email: {user.email}</p>
              <p>Role: {user.role}</p>
            </div>
          )}
        </div>

        <div className="space-x-2">
          {!isAuthenticated ? (
            <button
              onClick={handleLogin}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            >
              Test Login
            </button>
          ) : (
            <button
              onClick={logout}
              className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
            >
              Logout
            </button>
          )}
          
          {isAuthenticated && (
            <button
              onClick={testApiCall}
              disabled={isTesting}
              className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 disabled:opacity-50"
            >
              {isTesting ? 'Testing...' : 'Test API Call'}
            </button>
          )}
        </div>

        {testResult && (
          <div className="mt-4 p-4 bg-gray-100 rounded">
            <h4 className="font-semibold">Test Result:</h4>
            <pre className="text-sm mt-2 whitespace-pre-wrap">{testResult}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
