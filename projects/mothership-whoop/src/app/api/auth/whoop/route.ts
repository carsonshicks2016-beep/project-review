import { NextResponse } from 'next/server';

export async function GET() {
  const clientId = process.env.WHOOP_CLIENT_ID;
  
  if (!clientId || clientId === 'your_client_id_here') {
    return new NextResponse("WHOOP_CLIENT_ID is not configured in .env.local", { status: 500 });
  }

  const redirectUri = `${process.env.NEXT_PUBLIC_APP_URL}/api/auth/whoop/callback`;
  
  const scopes = [
    'offline',
    'read:recovery',
    'read:cycles',
    'read:workout',
    'read:sleep',
    'read:profile',
    'read:body_measurement'
  ].join(' ');

  const authUrl = new URL('https://api.prod.whoop.com/oauth/oauth2/auth');
  authUrl.searchParams.append('client_id', clientId);
  authUrl.searchParams.append('response_type', 'code');
  authUrl.searchParams.append('redirect_uri', redirectUri);
  authUrl.searchParams.append('scope', scopes);
  
  // In a real app, generate a secure random state and store it in a cookie/session
  authUrl.searchParams.append('state', 'whoop_sync_init'); 

  return NextResponse.redirect(authUrl.toString());
}
