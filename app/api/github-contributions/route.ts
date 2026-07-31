import { NextRequest, NextResponse } from "next/server"

const CONTRIBUTIONS_QUERY = `
  query ($username: String!) {
    user(login: $username) {
      contributionsCollection {
        contributionCalendar {
          totalContributions
          weeks {
            contributionDays {
              color
              contributionCount
              contributionLevel
              date
            }
          }
        }
      }
    }
  }
`

export async function GET(request: NextRequest) {
  const username = request.nextUrl.searchParams.get("username")
  if (!username) {
    return NextResponse.json({ error: "username is required" }, { status: 400 })
  }

  const token = process.env.GITHUB_TOKEN
  if (!token) {
    return NextResponse.json({ error: "GITHUB_TOKEN is not configured" }, { status: 500 })
  }

  const response = await fetch("https://api.github.com/graphql", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: CONTRIBUTIONS_QUERY,
      variables: { username },
    }),
  })

  if (!response.ok) {
    return NextResponse.json(
      { error: `GitHub API responded with ${response.status}` },
      { status: 502 }
    )
  }

  const json = await response.json()
  const calendar = json.data?.user?.contributionsCollection?.contributionCalendar
  if (!calendar) {
    return NextResponse.json(
      { error: json.errors?.[0]?.message ?? "User not found" },
      { status: 404 }
    )
  }

  return NextResponse.json(
    {
      totalContributions: calendar.totalContributions,
      contributions: calendar.weeks.map(
        (week: { contributionDays: unknown[] }) => week.contributionDays
      ),
    },
    {
      headers: {
        "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400",
      },
    }
  )
}
