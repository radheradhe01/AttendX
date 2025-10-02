'use client';

import { SessionProvider } from 'next-auth/react';
import AuthTest from '@/components/AuthTest';

export default function TestAuthPage() {
  return (
    <SessionProvider>
      <div className="min-h-screen bg-gray-50">
        <AuthTest />
      </div>
    </SessionProvider>
  );
}
