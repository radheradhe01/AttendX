'use client';

import { useContext } from 'react';
import { ConsciousnessContext, ConsciousnessContextType } from '@/components/providers/ConsciousnessProvider';

export function useConsciousness(): ConsciousnessContextType {
  const context = useContext(ConsciousnessContext);
  if (context === undefined) {
    throw new Error('useConsciousness must be used within a ConsciousnessProvider');
  }
  return context;
}
