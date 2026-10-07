# Privacy Policy

_Last updated: 28 July 2026_

This policy covers **Asclepius**, a personal, single-user tool that reads its
operator's own WHOOP data. It is not a commercial product, has no users other
than its operator, and is not offered to the public.

## What it accesses

With the operator's explicit authorization through WHOOP's OAuth flow, the
application reads the following from the WHOOP API:

- Physiological cycles (strain, energy expenditure, heart rate)
- Recovery (recovery score, resting heart rate, heart rate variability, blood
  oxygen, skin temperature)
- Sleep (duration, stages, respiratory rate, performance measures)
- Workouts (activity type, strain, heart rate zones, distance, elevation)
- Basic profile and body measurements (height, weight, maximum heart rate)

Only the operator's own account is ever accessed.

## Where the data goes

Nowhere. All retrieved data is written to a SQLite database on the operator's
own computer. There is no server, no hosted component, no analytics, no
telemetry, and no third-party processor. The data is never transmitted,
published, sold, or shared with anyone.

Network traffic is limited to requests to WHOOP's own API endpoints
(`api.prod.whoop.com`).

## Credentials

OAuth access and refresh tokens are stored in a file readable only by the
operator's user account. They are used solely to authenticate requests to the
WHOOP API and are never transmitted anywhere other than WHOOP.

## Retention and deletion

Data is retained locally until the operator deletes it. Deleting the local
database directory removes everything the application has stored. Authorization
can be revoked at any time from WHOOP account settings, which immediately ends
the application's access.

## Changes

Any change to this policy will be published at this URL.

## Contact

Questions about this policy: user@example.com
