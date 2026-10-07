import prisma from '@/lib/prisma';
import Link from 'next/link';
import { SyncButton } from '@/components/SyncButton';
import { DashboardCharts } from '@/components/DashboardCharts';
import { VitrineConsole } from '@/components/vitrine/VitrineConsole';

export default async function Home({ searchParams }: { searchParams: Promise<{ userId?: string }> }) {
  const sp = await searchParams;
  const userId = sp?.userId;

  let user = null;
  if (userId) {
    user = await prisma.user.findUnique({
      where: { id: userId },
      include: {
        profile: true,
        recoveries: { orderBy: { createdAt: 'asc' }, take: 30 },
        sleeps: { orderBy: { createdAt: 'asc' }, take: 30 },
        workouts: { orderBy: { createdAt: 'asc' }, take: 30 },
        cycles: { orderBy: { start: 'asc' }, take: 30 },
      }
    });
  }

  // If no specific userId passed, try to just grab the first user for demo purposes
  if (!user) {
    user = await prisma.user.findFirst({
      include: {
        profile: true,
        recoveries: { orderBy: { createdAt: 'asc' }, take: 30 },
        sleeps: { orderBy: { createdAt: 'asc' }, take: 30 },
        workouts: { orderBy: { createdAt: 'asc' }, take: 30 },
        cycles: { orderBy: { start: 'asc' }, take: 30 },
      }
    });
  }

  return (
    <main className="min-h-screen bg-neutral-950 text-white p-8 font-sans">
      <div className="max-w-6xl mx-auto space-y-8 mt-12">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Mothership Dashboard</h1>
            <p className="text-neutral-400 mt-1 text-sm">Centralized Whoop Intelligence</p>
          </div>
          <div className="flex items-center gap-4">
            {user && <SyncButton userId={user.id} />}
            <Link 
              href="/api/auth/whoop" 
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium transition-colors text-sm"
            >
              {user ? 'Reconnect Whoop' : 'Connect Whoop'}
            </Link>
          </div>
        </header>

        {!user ? (
          <div className="border border-neutral-800 rounded-2xl p-16 text-center bg-neutral-900/50 mt-12">
            <h2 className="text-xl font-medium mb-2">No Data Available</h2>
            <p className="text-neutral-400 mb-6 text-sm">Connect your Whoop account to start syncing data into the Mothership.</p>
          </div>
        ) : (
          <>
            <VitrineConsole
              rows={{
                recoveries: user.recoveries,
                sleeps: user.sleeps,
                workouts: user.workouts,
                cycles: user.cycles,
              }}
            />

            <DashboardCharts 
              recoveries={user.recoveries} 
              sleeps={user.sleeps} 
              workouts={user.workouts} 
            />
          </>
        )}
      </div>
    </main>
  );
}
