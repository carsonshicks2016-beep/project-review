import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/lib/prisma';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { userId } = body;

    if (!userId) {
      return NextResponse.json({ error: "Missing userId" }, { status: 400 });
    }

    const user = await prisma.user.findUnique({
      where: { id: userId }
    });

    if (!user || !user.accessToken) {
      return NextResponse.json({ error: "User or access token not found" }, { status: 404 });
    }

    const headers = { 'Authorization': `Bearer ${user.accessToken}` };
    const baseUrl = 'https://api.prod.whoop.com/developer';

    // 1. Fetch Recoveries
    const recoveryRes = await fetch(`${baseUrl}/v2/recovery`, { headers });
    if (recoveryRes.ok) {
      const data = await recoveryRes.json();
      const records = data.records || [];
      for (const record of records) {
        await prisma.recovery.upsert({
          where: { whoopId: record.cycle_id.toString() },
          update: {
            score: record.score?.recovery_score,
            restingHr: record.score?.resting_heart_rate,
            hrv: record.score?.hrv_rmssd_milli,
            updatedAt: new Date(record.updated_at),
          },
          create: {
            whoopId: record.cycle_id.toString(),
            userId: user.id,
            createdAt: new Date(record.created_at),
            updatedAt: new Date(record.updated_at),
            score: record.score?.recovery_score,
            restingHr: record.score?.resting_heart_rate,
            hrv: record.score?.hrv_rmssd_milli,
          }
        });
      }
    }

    // 2. Fetch Sleep
    const sleepRes = await fetch(`${baseUrl}/v2/activity/sleep`, { headers });
    if (sleepRes.ok) {
      const data = await sleepRes.json();
      const records = data.records || [];
      for (const record of records) {
        // WHOOP reports sleep need as four components; the total need is their
        // sum, and need_from_recent_nap_milli arrives negative.
        const need = record.score?.sleep_needed;
        const sleepNeededMilli = need
          ? (need.baseline_milli ?? 0) +
            (need.need_from_sleep_debt_milli ?? 0) +
            (need.need_from_recent_strain_milli ?? 0) +
            (need.need_from_recent_nap_milli ?? 0)
          : null;

        const stages = record.score?.stage_summary;
        const sleepScore = {
          scoreState: record.score_state,
          sleepPerformancePercentage: record.score?.sleep_performance_percentage,
          sleepEfficiencyPercentage: record.score?.sleep_efficiency_percentage,
          sleepConsistencyPercentage: record.score?.sleep_consistency_percentage,
          respiratoryRate: record.score?.respiratory_rate,
          sleepNeededMilli,
          totalInBedTimeMilli: stages?.total_in_bed_time_milli,
          totalAwakeTimeMilli: stages?.total_awake_time_milli,
          totalNoDataTimeMilli: stages?.total_no_data_time_milli,
          totalLightSleepTimeMilli: stages?.total_light_sleep_time_milli,
          totalSlowWaveSleepTimeMilli: stages?.total_slow_wave_sleep_time_milli,
          totalRemSleepTimeMilli: stages?.total_rem_sleep_time_milli,
        };

        await prisma.sleep.upsert({
          where: { whoopId: record.id.toString() },
          update: {
            ...sleepScore,
            updatedAt: new Date(record.updated_at),
          },
          create: {
            ...sleepScore,
            whoopId: record.id.toString(),
            userId: user.id,
            createdAt: new Date(record.created_at),
            updatedAt: new Date(record.updated_at),
            start: new Date(record.start),
            end: new Date(record.end),
            timezoneOffset: record.timezone_offset,
          }
        });
      }
    }

    // 3. Fetch Workouts
    const workoutRes = await fetch(`${baseUrl}/v2/activity/workout`, { headers });
    if (workoutRes.ok) {
      const data = await workoutRes.json();
      const records = data.records || [];
      for (const record of records) {
        await prisma.workout.upsert({
          where: { whoopId: record.id.toString() },
          update: {
            scoreState: record.score_state,
            strain: record.score?.strain,
            averageHeartRate: record.score?.average_heart_rate,
            maxHeartRate: record.score?.max_heart_rate,
            kilojoules: record.score?.kilojoules ?? record.score?.kilojoule,
            zoneDurationMilli: (() => { const z = record.score?.zone_durations ?? record.score?.zone_duration; return z ? JSON.stringify(z) : null; })(),
          },
          create: {
            whoopId: record.id.toString(),
            userId: user.id,
            createdAt: new Date(record.created_at),
            updatedAt: new Date(record.updated_at),
            start: new Date(record.start),
            end: new Date(record.end),
            timezoneOffset: record.timezone_offset,
            sportId: record.sport_id,
            scoreState: record.score_state,
            strain: record.score?.strain,
            averageHeartRate: record.score?.average_heart_rate,
            maxHeartRate: record.score?.max_heart_rate,
            kilojoules: record.score?.kilojoules ?? record.score?.kilojoule,
            zoneDurationMilli: (() => { const z = record.score?.zone_durations ?? record.score?.zone_duration; return z ? JSON.stringify(z) : null; })(),
          }
        });
      }
    }

    // 4. Fetch Cycles
    const cycleRes = await fetch(`${baseUrl}/v2/cycle`, { headers });
    if (cycleRes.ok) {
      const data = await cycleRes.json();
      const records = data.records || [];
      for (const record of records) {
        // Day strain lives here and nowhere else. Workout.strain is per-activity
        // and does not sum into a day figure.
        const cycleScore = {
          scoreState: record.score_state,
          strain: record.score?.strain,
          kilojoule: record.score?.kilojoule ?? record.score?.kilojoules,
          averageHeartRate: record.score?.average_heart_rate,
          maxHeartRate: record.score?.max_heart_rate,
        };

        await prisma.cycle.upsert({
          where: { whoopId: record.id.toString() },
          update: {
            ...cycleScore,
            updatedAt: new Date(record.updated_at),
            end: record.end ? new Date(record.end) : null,
          },
          create: {
            ...cycleScore,
            whoopId: record.id.toString(),
            userId: user.id,
            createdAt: new Date(record.created_at),
            updatedAt: new Date(record.updated_at),
            start: new Date(record.start),
            end: record.end ? new Date(record.end) : null,
            timezoneOffset: record.timezone_offset,
          }
        });
      }
    }

    return NextResponse.json({ success: true, message: "Sync complete" });
  } catch (error: any) {
    console.error("Sync error:", error);
    return NextResponse.json({ error: "Failed to sync data", details: error.message }, { status: 500 });
  }
}
