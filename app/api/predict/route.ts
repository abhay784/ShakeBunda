import { NextResponse } from "next/server";
import { predictRI } from "@/lib/databricks/client";
import { PredictRequestSchema } from "@/lib/databricks/types";

export const runtime = "nodejs";

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }

  const parsed = PredictRequestSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "invalid request", issues: parsed.error.issues },
      { status: 400 },
    );
  }

  const result = await predictRI(parsed.data);
  return NextResponse.json(result);
}
