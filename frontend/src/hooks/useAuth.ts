/**
 * Authentication Hook
 * Provides authentication state and methods for the frontend
 */

import { useSession, signIn, signOut } from 'next-auth/react';
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { tokenManager } from '@/lib/api';

export function useAuth() {
  const { data: session, status } = useSession();
  const router = useRouter();

  // Initialize token when session changes
  useEffect(() => {
    if (session?.accessToken) {
      tokenManager.setToken(session.accessToken);
    } else {
      tokenManager.removeToken();
    }
  }, [session]);

  const login = async (email: string, password: string) => {
    try {
      const result = await signIn('credentials', {
        email,
        password,
        redirect: false,
      });

      if (result?.error) {
        throw new Error(result.error);
      }

      return { success: true };
    } catch (error) {
      console.error('Login error:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Login failed' };
    }
  };

  const logout = async () => {
    try {
      tokenManager.removeToken();
      await signOut({ redirect: false });
      return { success: true };
    } catch (error) {
      console.error('Logout error:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Logout failed' };
    }
  };

  const requireAuth = (redirectTo: string = '/auth/login') => {
    useEffect(() => {
      if (status === 'unauthenticated') {
        router.push(redirectTo);
      }
    }, [status, redirectTo, router]);
  };

  const isAuthenticated = status === 'authenticated' && !!session?.accessToken;
  const isLoading = status === 'loading';
  const user = session?.user;

  return {
    user,
    isAuthenticated,
    isLoading,
    login,
    logout,
    requireAuth,
    token: session?.accessToken,
  };
}