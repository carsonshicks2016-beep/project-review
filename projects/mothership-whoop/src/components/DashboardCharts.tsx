'use client';

import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
  BarChart, Bar, Legend
} from 'recharts';
import { format } from 'date-fns';

export function DashboardCharts({ recoveries, sleeps, workouts }: { recoveries: any[], sleeps: any[], workouts: any[] }) {
  
  // Format data for the charts
  const recoveryData = recoveries.map(r => ({
    date: format(new Date(r.createdAt), 'MMM dd'),
    score: r.score,
    hrv: r.hrv
  }));

  const sleepData = sleeps.map(s => ({
    date: format(new Date(s.createdAt), 'MMM dd'),
    performance: s.sleepPerformancePercentage,
    efficiency: s.sleepEfficiencyPercentage
  }));

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
      {/* Recovery Trends */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-6">
        <h3 className="text-lg font-medium mb-6">Recovery & HRV Trends</h3>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={recoveryData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
              <XAxis dataKey="date" stroke="#888" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis yAxisId="left" stroke="#888" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis yAxisId="right" orientation="right" stroke="#888" fontSize={12} tickLine={false} axisLine={false} />
              <RechartsTooltip 
                contentStyle={{ backgroundColor: '#171717', border: '1px solid #333', borderRadius: '8px' }}
                itemStyle={{ color: '#fff' }}
              />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="score" name="Recovery Score" stroke="#4ade80" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
              <Line yAxisId="right" type="monotone" dataKey="hrv" name="HRV" stroke="#f472b6" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Sleep Performance */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-6">
        <h3 className="text-lg font-medium mb-6">Sleep Performance</h3>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={sleepData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
              <XAxis dataKey="date" stroke="#888" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke="#888" fontSize={12} tickLine={false} axisLine={false} domain={[0, 100]} />
              <RechartsTooltip 
                contentStyle={{ backgroundColor: '#171717', border: '1px solid #333', borderRadius: '8px' }}
                cursor={{ fill: '#262626' }}
              />
              <Legend />
              <Bar dataKey="performance" name="Performance %" fill="#60a5fa" radius={[4, 4, 0, 0]} />
              <Bar dataKey="efficiency" name="Efficiency %" fill="#818cf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
