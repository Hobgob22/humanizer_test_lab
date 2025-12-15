import { useEffect, useMemo, useState, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChevronDown, ChevronUp, Gauge, Loader2, ShieldCheck, Sparkles, XCircle } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Input } from "../components/ui/Input";
import { Alert, AlertDescription, AlertIcons, AlertTitle } from "../components/ui/Alert";
import { VANILLA_MODELS } from "../data/models";
import { api } from "../lib/api";

// Filter out Groq models - they don't support complex structured outputs reliably
const SUPPORTED_MODELS = VANILLA_MODELS.filter(m => m.provider !== "Groq");

const FILE_LIMIT = 5;
const MAX_SINGLE_MB = 2;
const REASONING_EFFORTS = ["none", "minimal", "low", "medium", "high"];
const THINKING_MODES = ["light", "standard", "extended", "heavy"];

const HUMANIZER_OPTIONS = [
  { id: "gpt-4o-old-model", label: "gpt-4o-old-model" },
  { id: "dpo-h40-e5-b8-m10-b20-v20", label: "dpo-h40-e5-b8-m10-b20-v20" },
  { id: "dpo-h40-e3-b8-m15-b20-v10", label: "dpo-h40-e3-b8-m15-b20-v10" },
  { id: "dpo-h40-e5-b8-m10-b20-v20:ckpt2", label: "dpo-h40-e5-b8-m10-b20-v20:ckpt2" },
  { id: "gpt-4.1-mini-hum40naive", label: "gpt-4.1-mini-hum40naive" },
  { id: "raw-e3-b8-m15-v10", label: "raw-e3-b8-m15-v10" },
  { id: "rubx-e8-b32-m03-v15", label: "rubx-e8-b32-m03-v15" },
];

const REASONING_CAPABILITIES = {
  "gpt-5": { reasoningEffort: true, thinkingMode: true, thinkingBudget: true },
  "gpt-5-mini": { reasoningEffort: true, thinkingMode: true, thinkingBudget: true },
  "gpt-5-nano": { reasoningEffort: true, thinkingMode: true, thinkingBudget: true },
  "gpt-5.1": { reasoningEffort: true, thinkingMode: true, thinkingBudget: true },
  "claude-sonnet-4": { thinkingMode: true, thinkingBudget: true },
  "claude-sonnet-4.5": { thinkingMode: true, thinkingBudget: true },
  "claude-haiku-4.5": { thinkingBudget: true },
  "gemini-2.5-flash": { deepThink: true, thinkingBudget: true },
  "gemini-2.5-pro": { deepThink: true, thinkingBudget: true },
  "gemini-3-pro": { deepThink: true, thinkingBudget: true },
};

const STORAGE_KEY = "writing_profile_lab_state_v1";

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

const countWords = (value = "") => {
  const trimmed = value.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
};

const countCharacters = (value = "") => {
  return value.trim().length;
};

export function WritingProfile() {
  const [selectedModels, setSelectedModels] = useState([SUPPORTED_MODELS[0]?.id ?? ""]);
  const [sampleText, setSampleText] = useState("");
  const [files, setFiles] = useState([]);
  const [reasoningEffort, setReasoningEffort] = useState("medium");
  const [thinkingMode, setThinkingMode] = useState("standard");
  const [deepThink, setDeepThink] = useState(false);
  const [thinkingBudget, setThinkingBudget] = useState("5000");
  const [results, setResults] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [pricingTable, setPricingTable] = useState({});
  const [pricingLoadError, setPricingLoadError] = useState("");
  const [isHydrated, setIsHydrated] = useState(false);
  const [expandedProfiles, setExpandedProfiles] = useState({});
  const [humanizeInput, setHumanizeInput] = useState("");
  const [humanizeSourceText, setHumanizeSourceText] = useState("");
  const [selectedProfileModel, setSelectedProfileModel] = useState("");
  const [selectedHumanizer, setSelectedHumanizer] = useState(HUMANIZER_OPTIONS[0]?.id ?? "");
  const [profileMode, setProfileMode] = useState("user");
  const [humanizeResult, setHumanizeResult] = useState(null);
  const [aiScoreResult, setAiScoreResult] = useState(null);
  const [qualityResult, setQualityResult] = useState(null);
  const [styleAdherenceResult, setStyleAdherenceResult] = useState(null);
  const [humanizeLoading, setHumanizeLoading] = useState(false);
  const [aiScoreLoading, setAiScoreLoading] = useState(false);
  const [qualityLoading, setQualityLoading] = useState(false);
  const [styleAdherenceLoading, setStyleAdherenceLoading] = useState(false);
  const [playgroundError, setPlaygroundError] = useState("");
  const [playgroundSuccess, setPlaygroundSuccess] = useState("");
  const lastOriginalWordCount = humanizeResult ? countWords(humanizeSourceText || humanizeInput) : 0;
  const lastHumanizedWordCount = humanizeResult ? countWords(humanizeResult?.humanized_text || "") : 0;
  const lastWordDiff = humanizeResult ? lastHumanizedWordCount - lastOriginalWordCount : 0;
  const lastOriginalCharCount = humanizeResult ? countCharacters(humanizeSourceText || humanizeInput) : 0;
  const lastHumanizedCharCount = humanizeResult ? countCharacters(humanizeResult?.humanized_text || "") : 0;
  const lastCharDiff = humanizeResult ? lastHumanizedCharCount - lastOriginalCharCount : 0;
  
  // Current input text stats (for display below textarea)
  const currentInputWordCount = countWords(humanizeInput);
  const currentInputCharCount = countCharacters(humanizeInput);
  
  // Abort controllers for cancellation support
  const profileAbortControllerRef = useRef(null);
  const humanizeAbortControllerRef = useRef(null);
  const aiScoreAbortControllerRef = useRef(null);
  const qualityAbortControllerRef = useRef(null);
  const styleAdherenceAbortControllerRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    const fetchPricing = async () => {
      try {
        const table = await api.getPricing();
        if (!cancelled) {
          setPricingTable(table);
        }
      } catch (err) {
        if (!cancelled) {
          setPricingLoadError(err?.response?.data?.detail || err.message || "Failed to load pricing data.");
        }
      }
    };
    fetchPricing();
    return () => {
      cancelled = true;
    };
  }, []);
  
  // Cleanup: abort all pending requests on unmount
  useEffect(() => {
    return () => {
      if (profileAbortControllerRef.current) {
        profileAbortControllerRef.current.abort();
      }
      if (humanizeAbortControllerRef.current) {
        humanizeAbortControllerRef.current.abort();
      }
      if (aiScoreAbortControllerRef.current) {
        aiScoreAbortControllerRef.current.abort();
      }
      if (qualityAbortControllerRef.current) {
        qualityAbortControllerRef.current.abort();
      }
      if (styleAdherenceAbortControllerRef.current) {
        styleAdherenceAbortControllerRef.current.abort();
      }
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      setIsHydrated(true);
      return;
    }
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        const validIds = new Set(SUPPORTED_MODELS.map((m) => m.id));
        const restoredModels = Array.isArray(parsed.selectedModels)
          ? parsed.selectedModels.filter((id) => validIds.has(id))
          : [];
        if (restoredModels.length) {
          setSelectedModels(restoredModels);
        }
        if (typeof parsed.sampleText === "string") {
          setSampleText(parsed.sampleText);
        }
        if (parsed.reasoningEffort) {
          setReasoningEffort(parsed.reasoningEffort);
        }
        if (parsed.thinkingMode) {
          setThinkingMode(parsed.thinkingMode);
        }
        if (typeof parsed.deepThink === "boolean") {
          setDeepThink(parsed.deepThink);
        }
        if (parsed.thinkingBudget) {
          setThinkingBudget(parsed.thinkingBudget);
        }
        if (typeof parsed.profileMode === "string") {
          setProfileMode(parsed.profileMode);
        }
        if (parsed.results && typeof parsed.results === "object") {
          setResults(parsed.results);
        }
      }
    } catch (err) {
      console.warn("[WritingProfile] Failed to hydrate saved state:", err);
    } finally {
      setIsHydrated(true);
    }
  }, []);

  useEffect(() => {
    if (!isHydrated || typeof window === "undefined") {
      return;
    }
    try {
      const payload = {
        selectedModels,
        sampleText,
        reasoningEffort,
        thinkingMode,
        deepThink,
        thinkingBudget,
        profileMode,
        results,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch (err) {
      console.warn("[WritingProfile] Failed to persist state:", err);
    }
  }, [
    selectedModels,
    sampleText,
    reasoningEffort,
    thinkingMode,
    deepThink,
    thinkingBudget,
    profileMode,
    results,
    isHydrated,
  ]);

  const supportsReasoningEffort = useMemo(
    () => selectedModels.some((id) => REASONING_CAPABILITIES[id]?.reasoningEffort),
    [selectedModels]
  );
  const supportsThinkingMode = useMemo(
    () => selectedModels.some((id) => REASONING_CAPABILITIES[id]?.thinkingMode),
    [selectedModels]
  );
  const supportsDeepThink = useMemo(
    () => selectedModels.some((id) => REASONING_CAPABILITIES[id]?.deepThink),
    [selectedModels]
  );
  const supportsThinkingBudget = useMemo(
    () => selectedModels.some((id) => REASONING_CAPABILITIES[id]?.thinkingBudget),
    [selectedModels]
  );

  const handleModelToggle = (modelId) => {
    setResults({});
    setSelectedModels((prev) =>
      prev.includes(modelId) ? prev.filter((id) => id !== modelId) : [...prev, modelId]
    );
  };

  const handleSelectAll = () => {
    setResults({});
    setSelectedModels(SUPPORTED_MODELS.map((m) => m.id));
  };

  const handleClearSelection = () => {
    setResults({});
    setSelectedModels([]);
  };

  const handleFileChange = (event) => {
    const chosen = Array.from(event.target.files || []);
    if (chosen.length > FILE_LIMIT) {
      setError(`You can attach up to ${FILE_LIMIT} files.`);
      return;
    }
    setFiles(chosen);
  };

  const resetFeedback = () => {
    setError("");
    setSuccess("");
  };

  const runAnalysis = async () => {
    resetFeedback();
    if (!selectedModels.length) {
      setError("Select at least one vanilla model to run.");
      return;
    }
    if (!sampleText.trim() && files.length === 0) {
      setError("Provide sample text or upload at least one file.");
      return;
    }
    
    // Create new abort controller
    profileAbortControllerRef.current = new AbortController();
    setIsLoading(true);
    
    try {
      const nextResults = { ...results };
      const tasks = selectedModels.map((modelId) =>
        api
          .generateWritingProfile({
            modelId,
            sampleText,
            reasoningEffort: REASONING_CAPABILITIES[modelId]?.reasoningEffort ? reasoningEffort : "",
            thinkingMode: REASONING_CAPABILITIES[modelId]?.thinkingMode ? thinkingMode : "",
            deepThink: Boolean(REASONING_CAPABILITIES[modelId]?.deepThink && deepThink),
            thinkingBudget: REASONING_CAPABILITIES[modelId]?.thinkingBudget ? thinkingBudget : "",
            files,
            signal: profileAbortControllerRef.current.signal,
          })
          .then((response) => ({ modelId, response }))
          .catch((error) => {
            // Handle cancellation gracefully
            if (error.name === 'CanceledError' || error.code === 'ERR_CANCELED') {
              return { modelId, cancelled: true };
            }
            throw error;
          })
      );

      const settled = await Promise.allSettled(tasks);
      const failures = [];
      const cancelled = [];

      settled.forEach((result, index) => {
        const modelId = selectedModels[index];
        if (result.status === "fulfilled") {
          if (result.value.cancelled) {
            cancelled.push(getModelLabel(modelId));
          } else {
            nextResults[modelId] = result.value.response;
          }
        } else {
          const message = result.reason?.response?.data?.detail || result.reason?.message || "Unknown error";
          failures.push(`${getModelLabel(modelId)}: ${message}`);
        }
      });

      setResults(nextResults);
      
      if (cancelled.length > 0) {
        setError(`Operation cancelled for: ${cancelled.join(", ")}`);
      } else if (failures.length > 0) {
        setError(`Some models failed: ${failures.join(" | ")}`);
      } else {
        setSuccess("Writing profile generated successfully.");
      }
    } catch (error) {
      // Handle unexpected errors gracefully
      console.error("Unexpected error in runAnalysis:", error);
      setError(`Unexpected error: ${error.message || "Please try again"}`);
    } finally {
      setIsLoading(false);
      profileAbortControllerRef.current = null;
    }
  };
  
  const stopProfileGeneration = () => {
    if (profileAbortControllerRef.current) {
      profileAbortControllerRef.current.abort();
      setIsLoading(false);
      setError("Profile generation stopped by user.");
    }
  };

  const getModelLabel = (modelId) =>
    SUPPORTED_MODELS.find((m) => m.id === modelId)?.name || modelId;
  const getModelPricing = (modelId) => pricingTable?.[modelId];

  const availableProfileOptions = useMemo(() => {
    return Object.entries(results)
      .filter(([, data]) => data?.profile)
      .map(([modelId]) => ({
        id: modelId,
        label: getModelLabel(modelId),
      }));
  }, [results]);

  const hasProfiles = availableProfileOptions.length > 0;

  useEffect(() => {
    if (!selectedProfileModel && availableProfileOptions.length === 1) {
      setSelectedProfileModel(availableProfileOptions[0].id);
    }
  }, [availableProfileOptions, selectedProfileModel]);

  const toggleResultCard = (modelId) => {
    setExpandedProfiles((prev) => ({
      ...prev,
      [modelId]: !prev[modelId],
    }));
  };

  const resetPlaygroundStatus = () => {
    setPlaygroundError("");
    setPlaygroundSuccess("");
  };

  const handleHumanizePreview = async () => {
    resetPlaygroundStatus();
    setAiScoreResult(null);
    setQualityResult(null);
    setStyleAdherenceResult(null);

    const normalizedText = humanizeInput.trim();
    if (!normalizedText) {
      setPlaygroundError("Enter some text to humanize.");
      return;
    }
    if (!selectedProfileModel) {
      setPlaygroundError("Select a writing profile to condition on.");
      return;
    }
    if (!selectedHumanizer) {
      setPlaygroundError("Select a humanizer model.");
      return;
    }
    const profilePayload = results[selectedProfileModel]?.profile;
    if (!profilePayload) {
      setPlaygroundError("Selected profile data is unavailable.");
      return;
    }

    // Create new abort controller
    humanizeAbortControllerRef.current = new AbortController();
    setHumanizeLoading(true);
    
    try {
      setHumanizeSourceText(normalizedText);
      const response = await api.humanizeWithProfile({
        text: normalizedText,
        modelId: selectedHumanizer,
        writingProfile: profilePayload,
        profileMode,
        signal: humanizeAbortControllerRef.current.signal,
      });
      setHumanizeResult(response);
      setPlaygroundSuccess(`Draft humanized with ${response.model}.`);
    } catch (err) {
      // Handle cancellation gracefully
      if (err.name === 'CanceledError' || err.code === 'ERR_CANCELED') {
        setPlaygroundError("Humanization cancelled by user.");
      } else {
        setPlaygroundError(err?.response?.data?.detail || err.message || "Failed to humanize text.");
      }
    } finally {
      setHumanizeLoading(false);
      humanizeAbortControllerRef.current = null;
    }
  };
  
  const stopHumanization = () => {
    if (humanizeAbortControllerRef.current) {
      humanizeAbortControllerRef.current.abort();
      setHumanizeLoading(false);
      setPlaygroundError("Humanization stopped by user.");
    }
  };

  const handleAiScoreCheck = async () => {
    resetPlaygroundStatus();
    if (!humanizeResult?.humanized_text) {
      setPlaygroundError("Generate a humanized draft first.");
      return;
    }

    // Create new abort controller
    aiScoreAbortControllerRef.current = new AbortController();
    setAiScoreLoading(true);
    
    try {
      const response = await api.checkAiScore({ 
        text: humanizeResult.humanized_text,
        signal: aiScoreAbortControllerRef.current.signal,
      });
      setAiScoreResult(response);
      setPlaygroundSuccess("AI score retrieved.");
    } catch (err) {
      // Handle cancellation gracefully
      if (err.name === 'CanceledError' || err.code === 'ERR_CANCELED') {
        setPlaygroundError("AI score check cancelled by user.");
      } else {
        setPlaygroundError(err?.response?.data?.detail || err.message || "Failed to compute AI score.");
      }
    } finally {
      setAiScoreLoading(false);
      aiScoreAbortControllerRef.current = null;
    }
  };
  
  const stopAiScoreCheck = () => {
    if (aiScoreAbortControllerRef.current) {
      aiScoreAbortControllerRef.current.abort();
      setAiScoreLoading(false);
      setPlaygroundError("AI score check stopped by user.");
    }
  };

  const handleQualityCheck = async () => {
    resetPlaygroundStatus();
    const sourceText = (humanizeSourceText || humanizeInput).trim();
    if (!sourceText || !humanizeResult?.humanized_text) {
      setPlaygroundError("Provide both original text and a humanized draft before running quality checks.");
      return;
    }

    // Create new abort controller
    qualityAbortControllerRef.current = new AbortController();
    setQualityLoading(true);
    
    try {
      const response = await api.runQualityCheck({
        originalText: sourceText,
        humanizedText: humanizeResult.humanized_text,
        signal: qualityAbortControllerRef.current.signal,
      });
      setQualityResult(response.result);
      setPlaygroundSuccess("Quality check complete.");
    } catch (err) {
      // Handle cancellation gracefully
      if (err.name === 'CanceledError' || err.code === 'ERR_CANCELED') {
        setPlaygroundError("Quality check cancelled by user.");
      } else {
        setPlaygroundError(err?.response?.data?.detail || err.message || "Failed to run quality check.");
      }
    } finally {
      setQualityLoading(false);
      qualityAbortControllerRef.current = null;
    }
  };
  
  const stopQualityCheck = () => {
    if (qualityAbortControllerRef.current) {
      qualityAbortControllerRef.current.abort();
      setQualityLoading(false);
      setPlaygroundError("Quality check stopped by user.");
    }
  };

  const handleStyleAdherenceCheck = async () => {
    resetPlaygroundStatus();
    if (!humanizeResult?.humanized_text) {
      setPlaygroundError("Generate a humanized draft first.");
      return;
    }
    if (!selectedProfileModel) {
      setPlaygroundError("No writing profile selected.");
      return;
    }
    const profilePayload = results[selectedProfileModel]?.profile;
    if (!profilePayload) {
      setPlaygroundError("Selected profile data is unavailable.");
      return;
    }
    if (!humanizeInput?.trim()) {
      setPlaygroundError("Original text is required for style adherence evaluation.");
      return;
    }

    // Create new abort controller
    styleAdherenceAbortControllerRef.current = new AbortController();
    setStyleAdherenceLoading(true);
    
    try {
      const response = await api.checkStyleAdherence({
        writingProfile: profilePayload,
        originalText: humanizeInput.trim(),
        humanizedText: humanizeResult.humanized_text,
        signal: styleAdherenceAbortControllerRef.current.signal,
      });
      setStyleAdherenceResult(response.result);
      setPlaygroundSuccess("Style adherence evaluation complete.");
    } catch (err) {
      // Handle cancellation gracefully
      if (err.name === 'CanceledError' || err.code === 'ERR_CANCELED') {
        setPlaygroundError("Style adherence check cancelled by user.");
      } else {
        setPlaygroundError(err?.response?.data?.detail || err.message || "Failed to evaluate style adherence.");
      }
    } finally {
      setStyleAdherenceLoading(false);
      styleAdherenceAbortControllerRef.current = null;
    }
  };
  
  const stopStyleAdherenceCheck = () => {
    if (styleAdherenceAbortControllerRef.current) {
      styleAdherenceAbortControllerRef.current.abort();
      setStyleAdherenceLoading(false);
      setPlaygroundError("Style adherence check stopped by user.");
    }
  };

  const renderResultCard = (modelId) => {
    const data = results[modelId];
    if (!data) return null;
    
    // Debug logging
    console.log(`[WritingProfile] Rendering card for ${modelId}:`, {
      hasMarkdown: !!data.markdown_preview,
      markdownLength: data.markdown_preview?.length,
      hasProfile: !!data.profile,
      hasPricing: !!data.pricing,
      keys: Object.keys(data)
    });
    
    const isExpanded = expandedProfiles[modelId] ?? false;

    return (
      <Card key={modelId} className="mt-6">
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <CardTitle>{getModelLabel(modelId)}</CardTitle>
            <CardDescription>
              Reasoning options applied:&nbsp;
              {Object.keys(data.reasoning || {}).length
                ? Object.entries(data.reasoning)
                    .map(([key, value]) => `${key}: ${value}`)
                    .join(", ")
                : "default"}
            </CardDescription>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="self-start sm:self-auto"
            onClick={() => toggleResultCard(modelId)}
          >
            {isExpanded ? (
              <>
                <ChevronUp className="h-4 w-4 mr-1" />
                Hide details
              </>
            ) : (
              <>
                <ChevronDown className="h-4 w-4 mr-1" />
                Show details
              </>
            )}
          </Button>
        </CardHeader>
        {isExpanded && (
          <CardContent className="space-y-4">
          <div>
            <p className="text-sm text-muted-foreground mb-2">Sample preview</p>
            <pre className="bg-muted rounded p-3 text-xs max-h-48 overflow-auto whitespace-pre-wrap">
              {data.sample_preview}
            </pre>
          </div>
          {data.sources?.length ? (
            <div>
              <p className="text-sm text-muted-foreground mb-1">Uploaded files</p>
              <ul className="text-sm list-disc list-inside text-foreground">
                {data.sources.map((src) => (
                  <li key={`${modelId}-${src.name}`}>
                    {src.name} ({formatBytes(src.size_bytes)})
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {/* Markdown Preview - Main Display */}
          <div className="border rounded-lg p-6 bg-background">
            <h3 className="text-lg font-semibold mb-4">📝 Writing Style Profile</h3>
            {data.markdown_preview ? (
              <div className="prose prose-sm max-w-none dark:prose-invert
                prose-headings:font-semibold prose-h1:text-2xl prose-h2:text-xl prose-h2:mt-6 prose-h2:mb-3
                prose-h3:text-lg prose-h3:mt-4 prose-h3:mb-2
                prose-p:my-2 prose-li:my-1 prose-table:w-full prose-table:text-xs
                prose-th:text-left prose-td:align-top prose-strong:text-foreground prose-code:text-sm prose-code:bg-muted prose-code:px-1 prose-code:rounded">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    // Prevent errors from breaking the entire render
                    p: ({node, ...props}) => <p {...props} />,
                    h1: ({node, ...props}) => <h1 {...props} />,
                    h2: ({node, ...props}) => <h2 {...props} />,
                    h3: ({node, ...props}) => <h3 {...props} />,
                    ul: ({node, ...props}) => <ul {...props} />,
                    li: ({node, ...props}) => <li {...props} />,
                    code: ({node, ...props}) => <code {...props} />,
                    em: ({node, ...props}) => <em {...props} />,
                    strong: ({node, ...props}) => <strong {...props} />,
                  }}
                >
                  {data.markdown_preview}
                </ReactMarkdown>
              </div>
            ) : data.profile ? (
              <div>
                <p className="text-sm text-muted-foreground mb-3">
                  Markdown preview not available. Showing structured data:
                </p>
                <pre className="text-xs overflow-auto max-h-96 bg-muted rounded p-3">
                  {JSON.stringify(data.profile, null, 2)}
                </pre>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                No data available.
              </p>
            )}
          </div>

          {/* Copyable profile payload */}
          {data.profile && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium">Profile payload</p>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => navigator.clipboard.writeText(JSON.stringify(data.profile, null, 2))}
                >
                  Copy
                </Button>
              </div>
              <textarea
                readOnly
                className="w-full min-h-[120px] p-2 text-sm font-mono border rounded resize-y bg-muted/30"
                value={JSON.stringify(data.profile, null, 2)}
              />
            </div>
          )}

          {/* Collapsible Sections */}
          <details className="bg-muted rounded p-3 text-sm">
            <summary className="cursor-pointer font-medium">📋 View Raw JSON Output</summary>
            <pre className="mt-3 text-xs overflow-auto max-h-96 bg-background border rounded p-3">
              {JSON.stringify(data.profile, null, 2)}
            </pre>
          </details>

          <details className="bg-muted rounded p-3 text-sm">
            <summary className="cursor-pointer font-medium">🔧 View Prompts Used</summary>
            <p className="mt-3 font-semibold">System prompt</p>
            <pre className="text-xs whitespace-pre-wrap">{data.system_prompt}</pre>
            <p className="mt-3 font-semibold">User prompt</p>
            <pre className="text-xs whitespace-pre-wrap">{data.user_prompt}</pre>
          </details>
          {data.pricing && (
            <div className="rounded border p-3 bg-muted/40 space-y-1">
              <p className="text-sm font-semibold">
                Estimated cost per call: ${data.pricing.estimated_cost.toFixed(4)}
              </p>
              <p className="text-xs text-muted-foreground">
                Input {data.pricing.input_tokens.toLocaleString()} tok · Output{" "}
                {data.pricing.output_tokens.toLocaleString()} tok · Thinking{" "}
                {data.pricing.thinking_tokens.toLocaleString()} tok
              </p>
              <p className="text-xs text-muted-foreground">
                Unit rates: ${data.pricing.unit_rates.input_per_mtok}/MTok in · $
                {data.pricing.unit_rates.output_per_mtok}/MTok out
              </p>
            </div>
          )}
          </CardContent>
        )}
      </Card>
    );
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Writing Profile Lab</h1>
        <p className="text-muted-foreground mt-1">
          Upload writing samples, fine-tune reasoning settings, and capture a JSON style profile.
        </p>
      </div>

      {(error || success) && (
        <Alert variant={error ? "destructive" : "success"}>
          {error ? <AlertIcons.error className="h-4 w-4" /> : <AlertIcons.success className="h-4 w-4" />}
          <AlertTitle>{error ? "Error" : "Success"}</AlertTitle>
          <AlertDescription>{error || success}</AlertDescription>
        </Alert>
      )}
      {pricingLoadError && (
        <Alert variant="warning">
          <AlertIcons.warning className="h-4 w-4" />
          <AlertTitle>Pricing unavailable</AlertTitle>
          <AlertDescription>{pricingLoadError}</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Inputs</CardTitle>
          <CardDescription>Provide at least one sample as text or file uploads.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium">Sample text</label>
            <textarea
              className="w-full min-h-[180px] rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              placeholder="Paste up to ~16k characters of representative writing..."
              value={sampleText}
              onChange={(e) => setSampleText(e.target.value)}
              disabled={isLoading}
            />
            <p className="text-xs text-muted-foreground">
              Plain text is truncated to ~16k characters before sending to the model.
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Upload files (optional)</label>
            <Input
              type="file"
              multiple
              accept=".txt,.docx,.pdf"
              onChange={handleFileChange}
              disabled={isLoading}
            />
            <p className="text-xs text-muted-foreground">
              Up to {FILE_LIMIT} files. Supported: .txt, .docx, .pdf (max {MAX_SINGLE_MB} MB each).
            </p>
            {files.length > 0 && (
              <ul className="text-sm text-foreground">
                {files.map((file) => (
                  <li key={file.name}>
                    {file.name} ({formatBytes(file.size)})
                  </li>
                ))}
              </ul>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Model Selection</CardTitle>
          <CardDescription>Choose one or more vanilla models to benchmark.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2 flex-wrap">
            <Button variant="ghost" size="sm" onClick={handleSelectAll} disabled={isLoading}>
              Select All
            </Button>
            <Button variant="ghost" size="sm" onClick={handleClearSelection} disabled={isLoading}>
              Clear
            </Button>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
            {SUPPORTED_MODELS.map((model) => (
              <label
                key={model.id}
                className="flex items-start gap-2 rounded border p-3 cursor-pointer hover:border-primary"
              >
                <input
                  type="checkbox"
                  className="mt-1"
                  checked={selectedModels.includes(model.id)}
                  onChange={() => handleModelToggle(model.id)}
                  disabled={isLoading}
                />
                <div className="space-y-1">
                  <p className="font-semibold">{model.name}</p>
                  <p className="text-xs text-muted-foreground">{model.provider}</p>
                  {getModelPricing(model.id) && (
                    <p className="text-xs text-muted-foreground">
                      ${getModelPricing(model.id).input_per_mtok}/MTok in · $
                      {getModelPricing(model.id).output_per_mtok}/MTok out
                    </p>
                  )}
                </div>
              </label>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Reasoning Controls</CardTitle>
          <CardDescription>
            Toggle advanced reasoning features discovered in latest research for Sonnet, Gemini 2.5+, and GPT-5/5.1.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Reasoning effort (GPT-5 family){!supportsReasoningEffort && " – not applicable to current selection"}
              </label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={reasoningEffort}
                onChange={(e) => setReasoningEffort(e.target.value)}
                disabled={!supportsReasoningEffort || isLoading}
              >
                {REASONING_EFFORTS.map((level) => (
                  <option key={level} value={level}>
                    {level.charAt(0).toUpperCase() + level.slice(1)}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Thinking time (GPT-5 & Claude thinking variants)
                {!supportsThinkingMode && " – not applicable"}
              </label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={thinkingMode}
                onChange={(e) => setThinkingMode(e.target.value)}
                disabled={!supportsThinkingMode || isLoading}
              >
                {THINKING_MODES.map((mode) => (
                  <option key={mode} value={mode}>
                    {mode.charAt(0).toUpperCase() + mode.slice(1)}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="deepThink"
              checked={deepThink}
              onChange={(e) => setDeepThink(e.target.checked)}
              disabled={!supportsDeepThink || isLoading}
            />
            <label htmlFor="deepThink" className="text-sm">
              Enable Gemini Deep Think (parallel reasoning)
              {!supportsDeepThink && " – select Gemini 2.5/3 Pro to enable"}
            </label>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium flex items-center gap-2">
              Thinking budget tokens (default 5000)
              {!supportsThinkingBudget && <span className="text-xs text-muted-foreground">(not applicable)</span>}
            </label>
            <Input
              type="number"
              min="0"
              value={thinkingBudget}
              onChange={(e) => setThinkingBudget(e.target.value)}
              disabled={!supportsThinkingBudget || isLoading}
            />
            <p className="text-xs text-muted-foreground">
              Applies to Claude thinking mode and Gemini Deep Think variants where available.
            </p>
          </div>
        </CardContent>
      </Card>

      <div className="flex gap-3 flex-wrap">
        <Button onClick={runAnalysis} disabled={isLoading}>
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Running {selectedModels.length} model(s)...
            </>
          ) : (
            "Generate Writing Profile"
          )}
        </Button>
        {isLoading && (
          <Button
            variant="destructive"
            onClick={stopProfileGeneration}
          >
            <XCircle className="h-4 w-4 mr-2" />
            Stop Generation
          </Button>
        )}
        <Button
          variant="outline"
          onClick={() => {
            setSampleText("");
            setFiles([]);
            setThinkingBudget("5000");
            setReasoningEffort("medium");
            setThinkingMode("standard");
            setDeepThink(false);
            resetFeedback();
          }}
          disabled={isLoading}
        >
          Reset Inputs
        </Button>
      </div>

      {Object.keys(results).length === 0 && (
        <p className="text-sm text-muted-foreground">Run the lab to see structured outputs per model.</p>
      )}

      {Object.keys(results).map((modelId) => {
        try {
          return renderResultCard(modelId);
        } catch (err) {
          console.error(`[WritingProfile] Error rendering ${modelId}:`, err);
          return (
            <Card key={modelId} className="mt-6">
              <CardHeader>
                <CardTitle>{getModelLabel(modelId)}</CardTitle>
              </CardHeader>
              <CardContent>
                <Alert variant="destructive">
                  <AlertIcons.error className="h-4 w-4" />
                  <AlertTitle>Rendering Error</AlertTitle>
                  <AlertDescription>
                    Failed to render results: {err.message}
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          );
        }
      })}

      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Humanize Playground</CardTitle>
          <CardDescription>
            Paste any text, apply one of the extracted writing profiles, and instantly humanize it with your
            preferred model. Then run quick AI-score and quality checks.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {playgroundError && (
            <Alert variant="destructive">
              <AlertIcons.error className="h-4 w-4" />
              <AlertTitle>Playground error</AlertTitle>
              <AlertDescription>{playgroundError}</AlertDescription>
            </Alert>
          )}
          {playgroundSuccess && (
            <Alert variant="success">
              <AlertIcons.success className="h-4 w-4" />
              <AlertTitle>Success</AlertTitle>
              <AlertDescription>{playgroundSuccess}</AlertDescription>
            </Alert>
          )}
          {!hasProfiles && (
            <Alert variant="warning">
              <AlertIcons.warning className="h-4 w-4" />
              <AlertTitle>Profiles needed</AlertTitle>
              <AlertDescription>
                Generate at least one writing profile above to unlock the playground controls.
              </AlertDescription>
            </Alert>
          )}

          <div className="space-y-2">
            <label className="text-sm font-medium">Text to humanize</label>
            <textarea
              className="w-full min-h-[160px] rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              placeholder="Paste a paragraph or short passage..."
              value={humanizeInput}
              onChange={(e) => setHumanizeInput(e.target.value)}
              disabled={!hasProfiles || humanizeLoading}
            />
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>
                {currentInputWordCount} {currentInputWordCount === 1 ? "word" : "words"} · {currentInputCharCount.toLocaleString()} {currentInputCharCount === 1 ? "character" : "characters"}
              </span>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">Writing profile</label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={selectedProfileModel}
                onChange={(e) => setSelectedProfileModel(e.target.value)}
                disabled={!hasProfiles || humanizeLoading}
              >
                <option value="">Select a profile</option>
                {availableProfileOptions.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
              <p className="text-xs text-muted-foreground">
                Options include every model that successfully generated a profile above.
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Humanizer model</label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={selectedHumanizer}
                onChange={(e) => setSelectedHumanizer(e.target.value)}
                disabled={!hasProfiles || humanizeLoading}
              >
                {HUMANIZER_OPTIONS.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
              <p className="text-xs text-muted-foreground">Only the vetted lab models are available here.</p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Writing profile injection</label>
              <select
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={profileMode}
                onChange={(e) => setProfileMode(e.target.value)}
                disabled={!hasProfiles || humanizeLoading}
              >
                <option value="user">Add profile to user prompt</option>
                <option value="system">Add profile to system prompt</option>
              </select>
              <p className="text-xs text-muted-foreground">
                User prompt mode matches the original behavior. System prompt mode prepends the profile to the model's system
                instructions instead.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <Button
              onClick={handleHumanizePreview}
              disabled={!hasProfiles || humanizeLoading}
            >
              {humanizeLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Humanizing...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Generate Humanized Draft
                </>
              )}
            </Button>
            {humanizeLoading && (
              <Button
                variant="destructive"
                size="sm"
                onClick={stopHumanization}
              >
                <XCircle className="h-4 w-4 mr-2" />
                Stop
              </Button>
            )}
            <Button
              variant="outline"
              onClick={handleAiScoreCheck}
              disabled={!humanizeResult?.humanized_text || aiScoreLoading}
            >
              {aiScoreLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Checking AI score...
                </>
              ) : (
                <>
                  <Gauge className="h-4 w-4 mr-2" />
                  Check AI Score
                </>
              )}
            </Button>
            {aiScoreLoading && (
              <Button
                variant="destructive"
                size="sm"
                onClick={stopAiScoreCheck}
              >
                <XCircle className="h-4 w-4 mr-2" />
                Stop
              </Button>
            )}
            <Button
              variant="outline"
              onClick={handleQualityCheck}
              disabled={!humanizeResult?.humanized_text || qualityLoading}
            >
              {qualityLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Running quality check...
                </>
              ) : (
                <>
                  <ShieldCheck className="h-4 w-4 mr-2" />
                  Run Quality Check
                </>
              )}
            </Button>
            {qualityLoading && (
              <Button
                variant="destructive"
                size="sm"
                onClick={stopQualityCheck}
              >
                <XCircle className="h-4 w-4 mr-2" />
                Stop
              </Button>
            )}
            <Button
              variant="outline"
              onClick={handleStyleAdherenceCheck}
              disabled={!humanizeResult?.humanized_text || styleAdherenceLoading}
            >
              {styleAdherenceLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Evaluating style...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Check Style Adherence
                </>
              )}
            </Button>
            {styleAdherenceLoading && (
              <Button
                variant="destructive"
                size="sm"
                onClick={stopStyleAdherenceCheck}
              >
                <XCircle className="h-4 w-4 mr-2" />
                Stop
              </Button>
            )}
          </div>

          {humanizeResult && (
            <div className="space-y-2">
              <div className="flex flex-col gap-2">
                <p className="text-sm font-medium">
                  Output from {humanizeResult.model}
                </p>
                <div className="flex flex-col gap-1 text-xs">
                  <div className="flex flex-wrap gap-3 text-muted-foreground">
                    <span>
                      <strong className="text-foreground">Initial:</strong> {lastOriginalWordCount} {lastOriginalWordCount === 1 ? "word" : "words"} · {lastOriginalCharCount.toLocaleString()} {lastOriginalCharCount === 1 ? "character" : "characters"}
                    </span>
                    <span>
                      <strong className="text-foreground">Humanized:</strong> {lastHumanizedWordCount} {lastHumanizedWordCount === 1 ? "word" : "words"} · {lastHumanizedCharCount.toLocaleString()} {lastHumanizedCharCount === 1 ? "character" : "characters"}
                    </span>
                    <span>
                      <strong className="text-foreground">Difference:</strong> {lastWordDiff >= 0 ? "+" : ""}{lastWordDiff} {lastWordDiff === 1 || lastWordDiff === -1 ? "word" : "words"} · {lastCharDiff >= 0 ? "+" : ""}{lastCharDiff.toLocaleString()} {lastCharDiff === 1 || lastCharDiff === -1 ? "character" : "characters"}
                    </span>
                  </div>
                  <div className="text-muted-foreground">
                    Profile: {humanizeResult.profile_summary} · Mode: {humanizeResult.profile_mode === "system" ? "system prompt" : "user prompt"}
                  </div>
                </div>
              </div>
              <pre className="bg-muted rounded p-3 text-sm whitespace-pre-wrap max-h-80 overflow-auto border">
                {humanizeResult.humanized_text}
              </pre>
              <details className="text-xs text-muted-foreground">
                <summary className="cursor-pointer text-sm font-medium">View applied style instructions</summary>
                <pre className="mt-2 bg-background border rounded p-3 whitespace-pre-wrap overflow-auto text-xs max-h-64">
                  {humanizeResult.instruction_preview}
                </pre>
              </details>
            </div>
          )}

          {aiScoreResult && (
            <div className="rounded border p-3 bg-muted/40 space-y-1">
              <p className="text-sm font-semibold">
                GPTZero (version {aiScoreResult.version}) ·{" "}
                {aiScoreResult.completely_generated_prob !== null && aiScoreResult.completely_generated_prob !== undefined
                  ? `Doc probability ${(aiScoreResult.completely_generated_prob * 100).toFixed(2)}%`
                  : "Score unavailable"}
              </p>
              <pre className="text-xs whitespace-pre-wrap overflow-auto max-h-64 bg-background rounded p-3 border">
                {JSON.stringify(aiScoreResult.raw_document, null, 2)}
              </pre>
            </div>
          )}

          {qualityResult && (
            <div className="rounded border p-3 bg-muted/40 space-y-2">
              <p className="text-sm font-semibold">Quality summary</p>
              <div className="grid gap-2 sm:grid-cols-2">
                <div className="text-sm">
                  <span className="font-medium">Same meaning:</span> {qualityResult.same_meaning_level}/10
                  {qualityResult.same_meaning ? " ✅" : " ⚠️"}
                </div>
                <div className="text-sm">
                  <span className="font-medium">Missing info:</span> {qualityResult.missing_info_level}/10
                  {qualityResult.no_missing_info ? " ✅" : " ⚠️"}
                </div>
                <div className="text-sm">
                  <span className="font-medium">Grammar:</span> {qualityResult.grammar_level}/10
                </div>
                <div className="text-sm">
                  <span className="font-medium">Length within ±15 words:</span>{" "}
                  {qualityResult.length_ok ? "Yes" : "No"}
                </div>
              </div>
              <details className="text-xs text-muted-foreground">
                <summary className="cursor-pointer text-sm font-medium">Detailed diagnostics</summary>
                <pre className="mt-2 bg-background border rounded p-3 whitespace-pre-wrap overflow-auto text-xs max-h-72">
                  {JSON.stringify(qualityResult, null, 2)}
                </pre>
              </details>
            </div>
          )}

          {styleAdherenceResult && (
            <div className="rounded border p-3 bg-muted/40 space-y-2">
              <p className="text-sm font-semibold">Style adherence evaluation</p>
              <div className="space-y-2">
                <div className="text-sm">
                  <span className="font-medium">Overall adherence:</span> {styleAdherenceResult.overall_adherence?.score}/10
                </div>
                <div className="text-xs text-muted-foreground">
                  {styleAdherenceResult.overall_adherence?.summary}
                </div>
                <div className="grid gap-2 sm:grid-cols-2 text-sm">
                  <div>
                    <span className="font-medium">Hedging:</span> {styleAdherenceResult.hedging?.score}/10
                  </div>
                  <div>
                    <span className="font-medium">Formality:</span> {styleAdherenceResult.formality?.score}/10
                  </div>
                  <div>
                    <span className="font-medium">Vocabulary:</span> {styleAdherenceResult.vocabulary?.score}/10
                  </div>
                  <div>
                    <span className="font-medium">Sentence structure:</span> {styleAdherenceResult.sentence_structure?.score}/10
                  </div>
                </div>
                {styleAdherenceResult.strengths?.length > 0 && (
                  <div className="text-xs">
                    <p className="font-medium mb-1">Strengths:</p>
                    <ul className="list-disc list-inside text-muted-foreground space-y-0.5">
                      {styleAdherenceResult.strengths.map((s, i) => (
                        <li key={i}>{s}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {styleAdherenceResult.weaknesses?.length > 0 && (
                  <div className="text-xs">
                    <p className="font-medium mb-1">Weaknesses:</p>
                    <ul className="list-disc list-inside text-muted-foreground space-y-0.5">
                      {styleAdherenceResult.weaknesses.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
              <details className="text-xs text-muted-foreground">
                <summary className="cursor-pointer text-sm font-medium">Detailed evaluation</summary>
                <pre className="mt-2 bg-background border rounded p-3 whitespace-pre-wrap overflow-auto text-xs max-h-72">
                  {JSON.stringify(styleAdherenceResult, null, 2)}
                </pre>
              </details>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

