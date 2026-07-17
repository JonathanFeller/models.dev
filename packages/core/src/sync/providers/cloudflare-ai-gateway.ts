import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { z } from "zod";

import type { ExistingModel, SyncProvider, SyncedFullModel, SyncedModel } from "../index.js";
import { factorBaseModel, resolveCanonicalBaseModel } from "./openrouter.js";
import { resolveCloudflareBaseModel } from "./cloudflare-workers-ai.js";

const GATEWAY_BASE = "https://gateway.ai.cloudflare.com/v1";
const PROVIDERS_DIR = path.join(import.meta.dirname, "..", "..", "..", "..", "..", "providers");
const MODELS_DIR = path.join(import.meta.dirname, "..", "..", "..", "..", "..", "models");

export const CloudflareAiGatewayModel = z.object({
  id: z.string().min(1),
  object: z.string().optional(),
  created_at: z.number().optional(),
  cost_in: z.number().nullable().optional(),
  cost_out: z.number().nullable().optional(),
  owned_by: z.string().optional(),
}).passthrough();

const CloudflareAiGatewayResponse = z.object({
  object: z.string().optional(),
  data: z.array(CloudflareAiGatewayModel),
}).passthrough();

export type CloudflareAiGatewayModel = z.infer<typeof CloudflareAiGatewayModel>;

/** Workers AI models the bash pipeline historically skipped. */
const SKIP_SUBSTRINGS = ["whisper", "aura-1"];

/** OpenAI / Anthropic IDs we never auto-create (non-chat or junk aliases). */
const SKIP_NAME = /^(ft:|text-embedding|text-moderation|text-ada|text-curie|text-davinci|omni-moderation|chatgpt-image|gpt-image|gpt-audio|gpt-realtime|gpt-oss|ada$|babbage|curie$|davinci)/i;

const DATED_NAME = /\d{4}-\d{2}-\d{2}|\d{8}/;

type SourceToml = {
  base_model?: string;
  reasoning_options?: SyncedFullModel["reasoning_options"];
  cost?: SyncedFullModel["cost"];
  interleaved?: SyncedFullModel["interleaved"];
  temperature?: boolean;
  status?: SyncedFullModel["status"];
  name?: string;
  release_date?: string;
  last_updated?: string;
  attachment?: boolean;
  reasoning?: boolean;
  tool_call?: boolean;
  structured_output?: boolean;
  open_weights?: boolean;
  knowledge?: string;
  family?: string;
  description?: string;
  limit?: SyncedFullModel["limit"];
  modalities?: SyncedFullModel["modalities"];
};

export const cloudflareAiGateway = {
  id: "cloudflare-ai-gateway",
  name: "Cloudflare AI Gateway",
  modelsDir: "providers/cloudflare-ai-gateway/models",
  // Catalog membership is authoritative for the curated namespaces we manage.
  deleteMissing: true,
  sourceID(model) {
    return model.id;
  },
  skippedNotice(ids) {
    if (ids.length === 0) return [];
    return [
      `${ids.length} Cloudflare AI Gateway models were skipped (filtered aliases, non-chat modalities, or missing metadata for auto-create).`,
      `Skipped remote IDs: ${ids.map((id) => `\`${id}\``).join(", ")}`,
    ];
  },
  async fetchModels() {
    return fetchCloudflareAiGatewayModels();
  },
  parseModels(raw) {
    return CloudflareAiGatewayResponse.parse(raw).data.filter(isManagedModel);
  },
  translateModel(model, context) {
    const id = model.id;
    // Prefer authored (file contents) so base_model entries stay factored.
    const authored = context.authored(id);

    if (shouldSkipModel(id)) return undefined;

    if (authored !== undefined) {
      return { id, model: preserveWithApiCost(authored, model) };
    }

    if (!shouldAutoCreate(id)) return undefined;

    const built = buildNewModel(model);
    if (built === undefined) return undefined;
    return { id, model: built };
  },
} satisfies SyncProvider<CloudflareAiGatewayModel>;

export async function fetchCloudflareAiGatewayModels(fetcher: typeof fetch = fetch) {
  const accountID =
    process.env.CLOUDFLARE_WORKERS_AI_SYNC_ACCOUNT_ID
    ?? process.env.CLOUDFLARE_ACCOUNT_ID;
  const token =
    process.env.CLOUDFLARE_WORKERS_AI_SYNC_API_TOKEN
    ?? process.env.CLOUDFLARE_API_TOKEN;
  const gatewayID =
    process.env.CLOUDFLARE_AI_GATEWAY_ID
    ?? process.env.CLOUDFLARE_GATEWAY_ID
    ?? "default";

  if (accountID === undefined || token === undefined) {
    throw new Error(
      "Cloudflare AI Gateway sync requires CLOUDFLARE_WORKERS_AI_SYNC_ACCOUNT_ID and CLOUDFLARE_WORKERS_AI_SYNC_API_TOKEN (or CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN)",
    );
  }

  const url = `${GATEWAY_BASE}/${accountID}/${gatewayID}/compat/models`;
  const response = await fetcher(url, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    throw new Error(
      `Cloudflare AI Gateway models request failed: ${response.status} ${response.statusText}`,
    );
  }
  return response.json();
}

export function isManagedModel(model: CloudflareAiGatewayModel) {
  const id = model.id;
  if (id.startsWith("workers-ai/@cf/")) return true;
  if (id.startsWith("openai/") || id.startsWith("anthropic/")) {
    // Only provider/model — drop openai/openai/* alias nests.
    return id.split("/").length === 2;
  }
  return false;
}

export function shouldSkipModel(id: string) {
  return SKIP_SUBSTRINGS.some((part) => id.includes(part));
}

export function shouldAutoCreate(id: string) {
  if (id.startsWith("workers-ai/@cf/")) return !shouldSkipModel(id);

  const [provider, name] = id.split("/");
  if ((provider !== "openai" && provider !== "anthropic") || name === undefined) return false;
  if (DATED_NAME.test(name)) return false;
  if (name.includes(":")) return false;
  if (SKIP_NAME.test(name)) return false;
  if (/(audio|realtime|transcribe|tts|whisper|search-preview|image)/i.test(name)) return false;
  // Anthropic API aliases like claude-opus-4-0 / claude-sonnet-4-0 — keep the bare form only.
  if (provider === "anthropic" && /-\d-0$/.test(name)) return false;

  return sourceProviderToml(provider, name) !== undefined
    || metadataExists(provider, name);
}

export function apiCost(model: CloudflareAiGatewayModel): SyncedFullModel["cost"] | undefined {
  const input = perMillion(model.cost_in);
  const output = perMillion(model.cost_out);
  if (input === undefined || output === undefined) return undefined;
  return { input, output };
}

function perMillion(value: number | null | undefined) {
  if (value === null || value === undefined) return undefined;
  if (!Number.isFinite(value) || value < 0) return undefined;
  return Math.round(value * 1_000_000_000_000) / 1_000_000;
}

function preserveWithApiCost(
  authored: ExistingModel,
  model: CloudflareAiGatewayModel,
): SyncedModel {
  const fromApi = apiCost(model);
  const cost = fromApi === undefined
    ? authored.cost
    : {
        input: fromApi.input,
        output: fromApi.output,
        cache_read: authored.cost?.cache_read,
        cache_write: authored.cost?.cache_write,
        reasoning: authored.cost?.reasoning,
        context_over_200k: authored.cost?.context_over_200k,
        tiers: authored.cost?.tiers,
      };

  if (authored.base_model !== undefined) {
    // Keep only fields that were authored as overrides; do not re-emit
    // resolved base_model metadata (name, limits, modalities, …).
    const values: Partial<SyncedFullModel> = {
      cost,
      reasoning_options: authored.reasoning_options,
      temperature: authored.temperature,
      interleaved: authored.interleaved,
      status: authored.status,
      experimental: authored.experimental,
      provider: authored.provider,
      limit: authored.limit,
      modalities: authored.modalities,
    };
    return factorBaseModel(
      authored.base_model,
      values,
      {
        context: authored.limit?.context ?? 0,
        output: authored.limit?.output ?? 0,
        input: authored.limit?.input,
      },
      authored.base_model_omit,
    );
  }

  return {
    name: authored.name!,
    description: authored.description,
    family: authored.family,
    release_date: authored.release_date!,
    last_updated: authored.last_updated!,
    attachment: authored.attachment!,
    reasoning: authored.reasoning!,
    reasoning_options: authored.reasoning_options,
    temperature: authored.temperature,
    tool_call: authored.tool_call!,
    structured_output: authored.structured_output,
    knowledge: authored.knowledge,
    open_weights: authored.open_weights!,
    status: authored.status,
    interleaved: authored.interleaved,
    experimental: authored.experimental,
    provider: authored.provider,
    cost: cost!,
    limit: authored.limit!,
    modalities: authored.modalities!,
  } as SyncedFullModel;
}

function buildNewModel(model: CloudflareAiGatewayModel): SyncedModel | undefined {
  const cost = apiCost(model);
  if (cost === undefined) return undefined;

  if (model.id.startsWith("workers-ai/")) {
    return buildNewWorkersAi(model, cost);
  }

  const [provider, name] = model.id.split("/");
  if (provider === undefined || name === undefined) return undefined;

  const source = sourceProviderToml(provider, name);
  const baseModel = source?.base_model
    ?? (metadataExists(provider, name) ? `${provider}/${name}` : undefined)
    ?? resolveCanonicalBaseModel(`${provider}/${name}`);

  if (baseModel === undefined) return undefined;

  const mergedCost = {
    input: cost.input,
    output: cost.output,
    cache_read: source?.cost?.cache_read,
    cache_write: source?.cost?.cache_write,
    reasoning: source?.cost?.reasoning,
    tiers: source?.cost?.tiers,
  };

  const values: Partial<SyncedFullModel> = {
    cost: mergedCost,
    reasoning_options: source?.reasoning_options,
    interleaved: source?.interleaved,
    temperature: source?.temperature,
    status: source?.status,
  };

  const limit = source?.limit ?? { context: 0, output: 0 };
  return factorBaseModel(baseModel, values, {
    context: limit.context ?? 0,
    output: limit.output ?? 0,
    input: limit.input,
  });
}

function buildNewWorkersAi(
  model: CloudflareAiGatewayModel,
  cost: NonNullable<SyncedFullModel["cost"]>,
): SyncedModel | undefined {
  const workersId = model.id.replace(/^workers-ai\//, "");
  const source = readToml(path.join(PROVIDERS_DIR, "cloudflare-workers-ai", "models", `${workersId}.toml`));

  if (source?.base_model !== undefined) {
    return factorBaseModel(source.base_model, {
      cost: {
        input: cost.input,
        output: cost.output,
        cache_read: source.cost?.cache_read,
        cache_write: source.cost?.cache_write,
        reasoning: source.cost?.reasoning,
        tiers: source.cost?.tiers,
      },
      reasoning_options: source.reasoning_options,
      interleaved: source.interleaved,
      temperature: source.temperature,
      modalities: source.modalities,
      limit: source.limit,
      status: source.status,
    }, {
      context: source.limit?.context ?? 0,
      output: source.limit?.output ?? 0,
      input: source.limit?.input,
    }, source.base_model_omit as string[] | undefined);
  }

  const openRouterShape = {
    id: workersId.startsWith("@cf/") ? workersId : `@cf/${workersId}`,
    name: source?.name ?? workersId,
    created: model.created_at ?? 0,
    hugging_face_id: null,
    knowledge_cutoff: null,
    context_length: source?.limit?.context ?? 128_000,
    architecture: {
      input_modalities: source?.modalities?.input ?? ["text"],
      output_modalities: source?.modalities?.output ?? ["text"],
    },
    pricing: {
      prompt: String(cost.input / 1_000_000),
      completion: String(cost.output / 1_000_000),
    },
    top_provider: {
      context_length: source?.limit?.context ?? 128_000,
      max_completion_tokens: source?.limit?.output ?? null,
    },
    supported_parameters: [] as string[],
  };

  const baseModel = resolveCloudflareBaseModel(openRouterShape as never)
    ?? (source?.base_model);

  if (baseModel !== undefined) {
    return factorBaseModel(baseModel, {
      cost,
      reasoning_options: source?.reasoning_options,
      interleaved: source?.interleaved,
      temperature: source?.temperature,
      modalities: source?.modalities,
    }, {
      context: source?.limit?.context ?? 128_000,
      output: source?.limit?.output ?? 16_384,
    });
  }

  // Full inline model only when the Workers AI provider already authored one.
  if (source?.name && source.release_date && source.last_updated && source.limit && source.modalities) {
    return {
      name: source.name,
      description: source.description,
      family: source.family as SyncedFullModel["family"],
      release_date: source.release_date,
      last_updated: source.last_updated,
      attachment: source.attachment ?? false,
      reasoning: source.reasoning ?? false,
      reasoning_options: source.reasoning_options,
      temperature: source.temperature ?? true,
      tool_call: source.tool_call ?? false,
      structured_output: source.structured_output,
      knowledge: source.knowledge,
      open_weights: source.open_weights ?? false,
      status: source.status,
      interleaved: source.interleaved,
      cost,
      limit: source.limit,
      modalities: source.modalities,
    } as SyncedFullModel;
  }

  return undefined;
}

function sourceProviderToml(provider: string, name: string): SourceToml | undefined {
  return readToml(path.join(PROVIDERS_DIR, provider, "models", `${name}.toml`));
}

function metadataExists(provider: string, name: string) {
  return existsSync(path.join(MODELS_DIR, provider, `${name}.toml`));
}

function readToml(filePath: string): (SourceToml & { base_model_omit?: string[] }) | undefined {
  if (!existsSync(filePath)) return undefined;
  try {
    return Bun.TOML.parse(readFileSync(filePath, "utf8")) as SourceToml & { base_model_omit?: string[] };
  } catch {
    return undefined;
  }
}
