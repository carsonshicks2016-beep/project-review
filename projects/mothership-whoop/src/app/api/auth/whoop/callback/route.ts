import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/lib/prisma';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const code = searchParams.get('code');

  if (!code) {
    return NextResponse.json({ error: "No code provided" }, { status: 400 });
  }

  const clientId = process.env.WHOOP_CLIENT_ID;
  const clientSecret = process.env.WHOOP_CLIENT_SECRET;
  
  if (!clientId || !clientSecret || clientId === 'your_client_id_here') {
    return new NextResponse("Whoop client credentials not configured.", { status: 500 });
  }

  const redirectUri = `${process.env.NEXT_PUBLIC_APP_URL}/api/auth/whoop/callback`;

  // Exchange code for token
  const tokenResponse = await fetch('https://api.prod.whoop.com/oauth/oauth2/token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code,
      client_id: clientId,
      client_secret: clientSecret,
      redirect_uri: redirectUri,
    }).toString(),
  });

  if (!tokenResponse.ok) {
    const errorText = await tokenResponse.text();
    console.error("Token exchange failed:", errorText);
    return NextResponse.redirect(`${process.env.NEXT_PUBLIC_APP_URL}?error=token_exchange_failed`);
  }

  const tokenData = await tokenResponse.json();
  const { access_token, refresh_token, expires_in } = tokenData;

  // Fetch basic profile to get user ID
  const profileResponse = await fetch('https://api.prod.whoop.com/developer/v2/user/profile/basic', {
    headers: {
      'Authorization': `Bearer ${access_token}`
    }
  });

  if (!profileResponse.ok) {
    return NextResponse.redirect(`${process.env.NEXT_PUBLIC_APP_URL}?error=profile_fetch_failed`);
  }

  const profileData = await profileResponse.json();
  const whoopUserId = profileData.user_id.toString();

  // Upsert user and profile in the database
  const tokenExpiresAt = new Date(Date.now() + expires_in * 1000);

  const user = await prisma.user.upsert({
    where: { whoopUserId },
    update: {
      accessToken: access_token,
      refreshToken: refresh_token,
      tokenExpiresAt,
      profile: {
        update: {
          firstName: profileData.first_name,
          lastName: profileData.last_name,
          email: profileData.email,
        }
      }
    },
    create: {
      whoopUserId,
      accessToken: access_token,
      refreshToken: refresh_token,
      tokenExpiresAt,
      profile: {
        create: {
          firstName: profileData.first_name,
          lastName: profileData.last_name,
          email: profileData.email,
        }
      }
    }
  });

  // Redirect to dashboard where we can trigger a sync
  // Pass the user ID as a simple query param for the local demo (in real app, use sessions)
  return NextResponse.redirect(`${process.env.NEXT_PUBLIC_APP_URL}?userId=${user.id}`);
}
