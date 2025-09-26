import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { ConsciousnessProvider } from '@/components/providers/ConsciousnessProvider';
import { Toaster } from 'react-hot-toast';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Artificial Consciousness Simulator',
  description: 'Proto-Conscious Agent with Real-time Consciousness Visualization',
  keywords: ['AI', 'Consciousness', 'AGI', 'Machine Learning', 'Cognitive Science'],
  authors: [{ name: 'Consciousness Research Team' }],
  viewport: 'width=device-width, initial-scale=1',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-gray-900 text-white antialiased`}>
        <ConsciousnessProvider>
          {children}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#1f2937',
                color: '#f9fafb',
                border: '1px solid #374151',
              },
            }}
          />
        </ConsciousnessProvider>
      </body>
    </html>
  );
}

