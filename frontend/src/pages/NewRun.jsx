import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Input } from "../components/ui/Input";
import { Alert, AlertDescription, AlertTitle, AlertIcons } from "../components/ui/Alert";
import { PlayCircle, Loader2, ChevronDown, ChevronRight } from "lucide-react";
import { VANILLA_MODELS } from "../data/models";

// Old fine-tuned models
const OLD_FINETUNES = [
  { id: "gpt-4o-old-model", name: "GPT-4o Old Model", base: "gpt-4o-mini" },
  { id: "gpt-4o-hum30raw", name: "GPT-4o hum30raw", base: "gpt-4o" },
  { id: "gpt-4o-hum40naive", name: "GPT-4o hum40naive", base: "gpt-4o" },
  { id: "gpt-4.1-hum30start", name: "GPT-4.1 hum30start", base: "gpt-4.1" },
  { id: "gpt-4.1-hum40naive", name: "GPT-4.1 hum40naive", base: "gpt-4.1" },
  { id: "gpt-4.1-mini-hum40naive", name: "GPT-4.1 Mini hum40naive", base: "gpt-4.1-mini" },
  { id: "gpt-4.1-hum30raw-fix1", name: "GPT-4.1 hum30raw-fix1", base: "gpt-4.1" },
  { id: "gpt-4o-hum30raw-fix1", name: "GPT-4o hum30raw-fix1", base: "gpt-4o" },
  { id: "gpt-4o-hum30raw-fix2", name: "GPT-4o hum30raw-fix2", base: "gpt-4o" },
  { id: "gpt-4.1-hum30raw-fix2", name: "GPT-4.1 hum30raw-fix2", base: "gpt-4.1" },
  { id: "gpt-4o-hum30raw-retrain1", name: "GPT-4o hum30raw-retrain1", base: "gpt-4o" },
  { id: "gpt-4o-mini-hum30raw-fix1", name: "GPT-4o Mini hum30raw-fix1", base: "gpt-4o-mini" },
];

// New fine-tuned models from CSV with checkpoint support
const NEW_FINETUNES = {
  "gpt-4o-mini": [
    { prefix: "min-e3-b8-m10-v10", desc: "Minimal prompt, both_raw", hasCheckpoints: true },
    { prefix: "raw-e5-b16-m08-v10", desc: "No prompt, no scores", hasCheckpoints: true },
    { prefix: "cmp-e4-b24-m12-v15", desc: "Compact prompt, both_binned", hasCheckpoints: true },
    { prefix: "min-e6-b16-m05-v20", desc: "Minimal prompt, ai_score_raw", hasCheckpoints: true },
    { prefix: "rubx-e8-b32-m03-v15", desc: "Rich prompt w/ counter examples", hasCheckpoints: true },
    { prefix: "raw-e3-b8-m15-v10", desc: "No prompt, ai_score_binned", hasCheckpoints: true },
  ],
  "gpt-4.1-mini": [
    { prefix: "min-e3-b8-m10-v10-d2", desc: "Minimal prompt, both_raw", hasCheckpoints: true },
    { prefix: "cmp-e5-b16-m08-v15", desc: "Compact prompt, both_binned", hasCheckpoints: true },
    { prefix: "raw-e4-b24-m06-v20", desc: "No prompt, no scores", hasCheckpoints: true },
    { prefix: "rdesc-e6-b20-m10-v10", desc: "Rich prompt w/ negative examples", hasCheckpoints: true },
    { prefix: "min-e8-b32-m04-v20", desc: "Minimal prompt, both_raw", hasCheckpoints: true },
    { prefix: "cmp-e3-b12-m12-v12", desc: "Compact prompt, ai_score_binned", hasCheckpoints: true },
  ],
  "gpt-4.1": [
    { prefix: "min-e3-b6-m08-v15", desc: "Minimal prompt, both_raw", hasCheckpoints: true },
    { prefix: "raw-e4-b8-m06-v10", desc: "No prompt, no scores", hasCheckpoints: true },
    { prefix: "cmp-e5-b8-m05-v20", desc: "Compact prompt, both_binned", hasCheckpoints: true },
    { prefix: "rchi-e6-b12-m04-v15", desc: "Rich prompt w/ focus areas", hasCheckpoints: true },
    { prefix: "min-e8-b16-m03-v20", desc: "Minimal prompt, both_raw", hasCheckpoints: true },
    { prefix: "raw-e3-b10-m10-v10", desc: "No prompt, ai_score_raw", hasCheckpoints: false },
  ],
  "gpt-4.1-nano": [
    { prefix: "min-e3-b32-m10-v10", desc: "Minimal prompt, both_binned", hasCheckpoints: true },
    { prefix: "raw-e5-b48-m08-v15", desc: "No prompt, no scores", hasCheckpoints: true },
    { prefix: "min-e4-b40-m12-v12", desc: "Minimal prompt, ai_score_binned", hasCheckpoints: true },
    { prefix: "raw-e3-b24-m15-v10", desc: "No prompt, no scores", hasCheckpoints: true },
  ],
  "gpt-4o": [
    { prefix: "min-e3-b4-m06-v15", desc: "Minimal prompt, both_raw", hasCheckpoints: true },
    { prefix: "raw-e4-b6-m05-v10", desc: "No prompt, no scores", hasCheckpoints: true },
    { prefix: "cmp-e5-b6-m04-v20", desc: "Compact prompt, both_binned", hasCheckpoints: true },
    { prefix: "rich-e6-b8-m03-v20", desc: "Rich prompt standard", hasCheckpoints: true },
    { prefix: "min-e3-b5-m10-v10", desc: "Minimal prompt, ai_score_raw", hasCheckpoints: true },
    { prefix: "raw-e8-b8-m025-v20", desc: "No prompt, no scores", hasCheckpoints: true },
  ],
};

// New DPO models from Batch 2
const DPO_MODELS = {
  "gpt-4.1-mini": [
    { prefix: "dpo-41m-min-e3-b32-m10-k05-v10", desc: "Minimal style guardrails, both_raw", hasCheckpoints: true },
    { prefix: "dpo-41m-raw-e4-b32-m08-k08-v10", desc: "No prompt, no scores", hasCheckpoints: true },
    { prefix: "dpo-41m-cmp-e3-b32-m10-k08-v15", desc: "Compact guidelines rubric, ai_score_raw", hasCheckpoints: true },
    { prefix: "dpo-41m-rch-e4-b32-m06-k08-v15", desc: "Rich prompt standard, both_binned", hasCheckpoints: true },
    { prefix: "dpo-41m-min-e3-b8-m06-k08-v15", desc: "Minimal style guardrails, both_raw", hasCheckpoints: true },
    { prefix: "dpo-41m-raw-e3-b10-m04-k10-v20", desc: "No prompt, no scores", hasCheckpoints: true },
    { prefix: "dpo-41m-min-e3-b32-m08-k03-v10", desc: "Minimal style guardrails, both_raw", hasCheckpoints: true },
  ],
  "gpt-4.1": [
    { prefix: "dpo-41-min-e3-b16-m08-k05-v10", desc: "Minimal style guardrails, both_raw", hasCheckpoints: true },
    { prefix: "dpo-41-cmp-e4-b16-m06-k08-v15", desc: "Compact guidelines rubric, ai_score_binned", hasCheckpoints: true },
    { prefix: "dpo-41-rch-e3-b12-m10-k10-v10", desc: "Rich prompt standard, ai_score_raw", hasCheckpoints: true },
    { prefix: "dpo-41-raw-e5-b16-m05-k08-v20", desc: "No prompt, no scores", hasCheckpoints: true },
    { prefix: "dpo-41-min-e3-b8-m05-k08-v15", desc: "Minimal style guardrails, both_raw", hasCheckpoints: true },
    { prefix: "dpo-41-raw-e4-b10-m04-k10-v20", desc: "No prompt, no scores", hasCheckpoints: true },
    { prefix: "dpo-41-min-e3-b16-m08-k03-v10", desc: "Minimal style guardrails, both_raw", hasCheckpoints: true },
  ],
  "gpt-4o": [
    { prefix: "dpo-4o-min-e3-b16-m08-k05-v10", desc: "Minimal style guardrails, both_raw", hasCheckpoints: true },
    { prefix: "dpo-4o-min-e3-b8-m05-k08-v15", desc: "Minimal style guardrails, both_raw", hasCheckpoints: true },
    { prefix: "dpo-4o-raw-e4-b6-m04-k10-v20", desc: "No prompt, no scores", hasCheckpoints: true },
    { prefix: "dpo-4o-min-e3-b16-m06-k03-v10", desc: "Minimal style guardrails, ai_score_raw", hasCheckpoints: true },
  ],
  "gpt-4.1-nano": [
    { prefix: "dpo-nano-min-e3-b32-m10-k05-v10", desc: "Minimal style guardrails, both_raw", hasCheckpoints: true },
    { prefix: "dpo-nano-cmp-e4-b32-m08-k08-v15", desc: "Compact guidelines rubric, no scores", hasCheckpoints: true },
    { prefix: "dpo-nano-min-e3-b8-m07-k05-v10", desc: "Minimal style guardrails, both_binned", hasCheckpoints: true },
    { prefix: "dpo-nano-raw-e4-b10-m05-k08-v15", desc: "No prompt, no scores", hasCheckpoints: true },
    { prefix: "dpo-nano-cmp-e4-b32-m08-k03-v10", desc: "Compact guidelines rubric, both_raw", hasCheckpoints: true },
  ],
};

// DPO models based on hum40-naive-auto
const DPO_H40_MODELS = [
  { prefix: "dpo-h40-e2-b8-m08-b30-v10", desc: "DPO on hum40-naive-auto", hasCheckpoints: true, hasCheckpoint2: false },
  { prefix: "dpo-h40-e3-b8-m10-b25-v10", desc: "DPO on hum40-naive-auto", hasCheckpoints: true, hasCheckpoint2: true },
  { prefix: "dpo-h40-e3-b8-m10-b15-v10", desc: "DPO on hum40-naive-auto", hasCheckpoints: true, hasCheckpoint2: true },
  { prefix: "dpo-h40-e3-b12-m10-b25-v10", desc: "DPO on hum40-naive-auto", hasCheckpoints: true, hasCheckpoint2: true },
  { prefix: "dpo-h40-e4-b12-m07-b35-v15", desc: "DPO on hum40-naive-auto", hasCheckpoints: true, hasCheckpoint2: true },
  { prefix: "dpo-h40-e4-b16-m05-b40-v15", desc: "DPO on hum40-naive-auto", hasCheckpoints: true, hasCheckpoint2: true },
  { prefix: "dpo-h40-e3-b8-m15-b20-v10", desc: "DPO on hum40-naive-auto", hasCheckpoints: true, hasCheckpoint2: true },
  { prefix: "dpo-h40-e5-b8-m10-b20-v20", desc: "DPO on hum40-naive-auto", hasCheckpoints: true, hasCheckpoint2: true },
  { prefix: "dpo-h40-e3-b8-m10-bauto-v10", desc: "DPO on hum40-naive-auto", hasCheckpoints: true, hasCheckpoint2: true },
  { prefix: "dpo-h40-e4-b12-m08-bauto-v15", desc: "DPO on hum40-naive-auto", hasCheckpoints: true, hasCheckpoint2: true },
];

// User-style humanization test models
const USER_STYLE_MODELS = [
  { id: "gpt-4o-old-model", name: "gpt-4o-old-model" },
  { id: "dpo-h40-e5-b8-m10-b20-v20", name: "dpo-h40-e5-b8-m10-b20-v20" },
  { id: "dpo-h40-e3-b8-m15-b20-v10", name: "dpo-h40-e3-b8-m15-b20-v10" },
  { id: "dpo-h40-e5-b8-m10-b20-v20:ckpt2", name: "dpo-h40-e5-b8-m10-b20-v20:ckpt2" },
  { id: "gpt-4.1-mini-hum40naive", name: "gpt-4.1-mini-hum40naive" },
  { id: "raw-e3-b8-m15-v10", name: "raw-e3-b8-m15-v10" },
  { id: "rubx-e8-b32-m03-v15", name: "rubx-e8-b32-m03-v15" },
];

const FOLDERS = {
  "AI texts": "data/ai_texts",
  "Human texts": "data/human_texts",
  "AI paragraphs": "data/ai_paras",
  "Human paragraphs": "data/human_paras",
};

export function NewRun() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // Expandable sections state
  const [expandedSections, setExpandedSections] = useState({
    vanilla: false,
    oldFinetunes: false,
    newFinetunes: false,
    dpoModels: false,
    dpoH40Models: false,
    userStyle: false,
  });

  const [expandedBaseModels, setExpandedBaseModels] = useState({
    "gpt-4o-mini": false,
    "gpt-4.1-mini": false,
    "gpt-4.1": false,
    "gpt-4.1-nano": false,
    "gpt-4o": false,
  });

  const [expandedDpoBaseModels, setExpandedDpoBaseModels] = useState({
    "gpt-4.1-mini": false,
    "gpt-4.1": false,
    "gpt-4o": false,
    "gpt-4.1-nano": false,
  });

  const [formData, setFormData] = useState({
    runName: `run_${Date.now()}`,
    folders: ["data/ai_texts"],
    models: [],
    iterations: 1,
    docCounts: {},
    includeDocMode: false,
    useGptzero: true,
    useSapling: false,
    userStyleProfile: "",
    userStyleProfileMode: "system",
    useStyleAdherence: true,
  });

  const [bulkModelInput, setBulkModelInput] = useState("");

  const toggleSection = (section) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const toggleBaseModel = (baseModel) => {
    setExpandedBaseModels((prev) => ({ ...prev, [baseModel]: !prev[baseModel] }));
  };

  const toggleDpoBaseModel = (baseModel) => {
    setExpandedDpoBaseModels((prev) => ({ ...prev, [baseModel]: !prev[baseModel] }));
  };

  const selectAllRuns = () => {
    setSelectedRuns(runs.map((r) => r.name));
  };

  const deselectAllRuns = () => {
    setSelectedRuns([]);
  };

  const selectAllFolders = () => {
    setFormData((prev) => ({ ...prev, folders: Object.values(FOLDERS) }));
  };

  const deselectAllFolders = () => {
    setFormData((prev) => ({ ...prev, folders: [] }));
  };

  const selectAllDpoBase = (baseModel) => {
    const models = DPO_MODELS[baseModel] || [];
    selectAllInGroup(models);
  };

  const deselectAllDpoBase = (baseModel) => {
    const models = DPO_MODELS[baseModel] || [];
    deselectAllInGroup(models);
  };

  const selectAllDpoH40 = () => selectAllInGroup(DPO_H40_MODELS);
  const deselectAllDpoH40 = () => deselectAllInGroup(DPO_H40_MODELS);

  const handleModelToggle = (modelId) => {
    setFormData((prev) => {
      const newModels = prev.models.includes(modelId)
        ? prev.models.filter((m) => m !== modelId)
        : [...prev.models, modelId];
      return { ...prev, models: newModels };
    });
  };

  const handleCheckpointToggle = (prefix, checkpoint) => {
    const modelId = checkpoint ? `${prefix}:${checkpoint}` : prefix;
    handleModelToggle(modelId);
  };

  const isModelSelected = (modelId) => formData.models.includes(modelId);

  const countSelectedInGroup = (models) => {
    return models.filter((m) => isModelSelected(m.id || m.prefix)).length;
  };

  const countSelectedCheckpoints = (prefix, hasCheckpoints) => {
    let count = 0;
    if (isModelSelected(prefix)) count++;
    if (hasCheckpoints) {
      if (isModelSelected(`${prefix}:ckpt1`)) count++;
      if (isModelSelected(`${prefix}:ckpt2`)) count++;
    }
    return count;
  };

  const handleFolderToggle = (folder) => {
    setFormData((prev) => {
      const newFolders = prev.folders.includes(folder)
        ? prev.folders.filter((f) => f !== folder)
        : [...prev.folders, folder];
      return { ...prev, folders: newFolders };
    });
  };

  const handleDocCountChange = (folder, count) => {
    setFormData((prev) => ({
      ...prev,
      docCounts: {
        ...prev.docCounts,
        [folder]: count ? parseInt(count) : undefined,
      },
    }));
  };

  const handleBulkModelInput = async () => {
    // Parse line-separated model IDs and add them to selection
    const modelIds = bulkModelInput
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    console.log("[Bulk Input] Raw input lines:", modelIds);

    if (modelIds.length === 0) return;

    try {
      // Fetch the model registry from the API to map full OpenAI IDs to short keys
      const response = await fetch("http://localhost:8000/api/models");
      const models = await response.json();

      console.log("[Bulk Input] Available models from API:", models.length);

      // Create reverse mapping: full OpenAI model ID -> short key
      const reverseMap = {};
      Object.entries(models).forEach(([shortKey, metadata]) => {
        if (metadata.model) {
          reverseMap[metadata.model] = shortKey;
        }
      });

      console.log("[Bulk Input] Created reverse mapping with", Object.keys(reverseMap).length, "entries");

      setFormData((prev) => {
        const newModels = [...prev.models];
        let addedCount = 0;
        let notFoundCount = 0;

        modelIds.forEach((id) => {
          // Try to find the short key for this ID
          let shortKey = reverseMap[id];

          // If not found in reverse map, maybe it's already a short key
          if (!shortKey && models[id]) {
            shortKey = id;
          }

          if (shortKey) {
            if (!newModels.includes(shortKey)) {
              newModels.push(shortKey);
              addedCount++;
              console.log("[Bulk Input] ✓ Added:", id, "→", shortKey);
            } else {
              console.log("[Bulk Input] - Already selected:", shortKey);
            }
          } else {
            // Treat as custom model
            if (!newModels.includes(id)) {
              newModels.push(id);
              addedCount++;
              console.log("[Bulk Input] ✓ Added custom model:", id);
            } else {
              console.log("[Bulk Input] - Already selected (custom):", id);
            }
          }
        });

        console.log(`[Bulk Input] Summary: ${addedCount} added, ${notFoundCount} not found`);
        return { ...prev, models: newModels };
      });

      setBulkModelInput("");
    } catch (err) {
      console.error("[Bulk Input] Error fetching models:", err);
      setError("Failed to fetch model registry. Please try again.");
    }
  };

  const selectAllInGroup = (models) => {
    setFormData((prev) => {
      const newModels = [...prev.models];
      models.forEach((model) => {
        const id = model.id || model.prefix;
        if (!newModels.includes(id)) {
          newModels.push(id);
        }
      });
      return { ...prev, models: newModels };
    });
  };

  const deselectAllInGroup = (models) => {
    setFormData((prev) => {
      const ids = models.map((m) => m.id || m.prefix);
      const newModels = prev.models.filter((m) => !ids.includes(m));
      return { ...prev, models: newModels };
    });
  };

  const selectAllCheckpoints = () => {
    setFormData((prev) => {
      const newModels = [...prev.models];
      Object.values(NEW_FINETUNES).flat().forEach((model) => {
        // Add final version
        if (!newModels.includes(model.prefix)) {
          newModels.push(model.prefix);
        }
        // Add checkpoints if available
        if (model.hasCheckpoints) {
          const ckpt1 = `${model.prefix}:ckpt1`;
          const ckpt2 = `${model.prefix}:ckpt2`;
          if (!newModels.includes(ckpt1)) newModels.push(ckpt1);
          if (!newModels.includes(ckpt2)) newModels.push(ckpt2);
        }
      });
      return { ...prev, models: newModels };
    });
  };

  const deselectAllCheckpoints = () => {
    setFormData((prev) => {
      const checkpointIds = new Set();
      Object.values(NEW_FINETUNES).flat().forEach((model) => {
        checkpointIds.add(model.prefix);
        if (model.hasCheckpoints) {
          checkpointIds.add(`${model.prefix}:ckpt1`);
          checkpointIds.add(`${model.prefix}:ckpt2`);
        }
      });
      const newModels = prev.models.filter((m) => !checkpointIds.has(m));
      return { ...prev, models: newModels };
    });
  };

  const selectAllDpoCheckpoints = () => {
    setFormData((prev) => {
      const newModels = [...prev.models];
      Object.values(DPO_MODELS).flat().forEach((model) => {
        // Add final version
        if (!newModels.includes(model.prefix)) {
          newModels.push(model.prefix);
        }
        // Add checkpoints if available
        if (model.hasCheckpoints) {
          const ckpt1 = `${model.prefix}:ckpt1`;
          const ckpt2 = `${model.prefix}:ckpt2`;
          if (!newModels.includes(ckpt1)) newModels.push(ckpt1);
          if (!newModels.includes(ckpt2)) newModels.push(ckpt2);
        }
      });
      return { ...prev, models: newModels };
    });
  };

  const deselectAllDpoCheckpoints = () => {
    setFormData((prev) => {
      const checkpointIds = new Set();
      Object.values(DPO_MODELS).flat().forEach((model) => {
        checkpointIds.add(model.prefix);
        if (model.hasCheckpoints) {
          checkpointIds.add(`${model.prefix}:ckpt1`);
          checkpointIds.add(`${model.prefix}:ckpt2`);
        }
      });
      const newModels = prev.models.filter((m) => !checkpointIds.has(m));
      return { ...prev, models: newModels };
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    setLoading(true);

    // Validation
    if (!formData.runName) {
      setError("Please provide a run name");
      setLoading(false);
      return;
    }

    if (formData.folders.length === 0) {
      setError("Please select at least one folder");
      setLoading(false);
      return;
    }

    if (formData.models.length === 0) {
      setError("Please select at least one model");
      setLoading(false);
      return;
    }

    try {
      // Determine which selected models are from the user-style section
      const userStyleModelIds = USER_STYLE_MODELS.map(m => m.id);
      const selectedUserStyleModels = formData.models.filter(m => userStyleModelIds.includes(m));
      
      const jobData = {
        run_name: formData.runName,
        folders: formData.folders,
        models: formData.models,
        iterations: formData.iterations,
        doc_counts: formData.docCounts,
        include_doc_mode: formData.includeDocMode,
        use_gptzero: formData.useGptzero,
        use_sapling: formData.useSapling,
        user_style_profile: formData.userStyleProfile || null,
        user_style_profile_mode: formData.userStyleProfileMode || null,
        use_style_adherence: formData.useStyleAdherence,
        user_style_models: selectedUserStyleModels,
      };

      const response = await api.createJob(jobData);
      setSuccess(`Job created successfully! Job ID: ${response.job_id}`);

      // Navigate to job status after a short delay
      setTimeout(() => {
        navigate("/");
      }, 2000);
    } catch (err) {
      // Handle FastAPI validation errors (422)
      if (err.response?.status === 422 && Array.isArray(err.response?.data?.detail)) {
        const validationErrors = err.response.data.detail
          .map((e) => `${e.loc?.join(" → ") || "Field"}: ${e.msg}`)
          .join("; ");
        setError(`Validation error: ${validationErrors}`);
      } else {
        setError(err.response?.data?.detail || err.message || "Failed to create job");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-3xl font-bold">New Benchmark Run</h1>
        <p className="text-muted-foreground mt-1">Create and configure a new humanization benchmark</p>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertIcons.error className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert variant="success">
          <AlertIcons.success className="h-4 w-4" />
          <AlertTitle>Success</AlertTitle>
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Run Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>Run Configuration</CardTitle>
            <CardDescription>Basic settings for the benchmark run</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="runName" className="text-sm font-medium">
                Run Name
              </label>
              <Input
                id="runName"
                value={formData.runName}
                onChange={(e) => setFormData({ ...formData, runName: e.target.value })}
                placeholder="Enter run name"
                disabled={loading}
              />
            </div>

            <div className="space-y-2">
              <label htmlFor="iterations" className="text-sm font-medium">
                Iterations per Document
              </label>
              <Input
                id="iterations"
                type="number"
                min="1"
                max="20"
                value={formData.iterations}
                onChange={(e) => setFormData({ ...formData, iterations: parseInt(e.target.value) })}
                disabled={loading}
              />
              <p className="text-xs text-muted-foreground">
                Number of humanization attempts per document
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Document Folders */}
        <Card>
          <CardHeader>
            <CardTitle>Document Folders</CardTitle>
            <CardDescription>Select which document folders to process</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm" onClick={selectAllFolders} disabled={loading}>
                Select All
              </Button>
              <Button variant="ghost" size="sm" onClick={deselectAllFolders} disabled={loading}>
                Deselect All
              </Button>
            </div>
            {Object.entries(FOLDERS).map(([name, path]) => (
              <div key={path} className="flex items-start gap-3 p-3 border rounded-md">
                <input
                  type="checkbox"
                  id={path}
                  checked={formData.folders.includes(path)}
                  onChange={() => handleFolderToggle(path)}
                  disabled={loading}
                  className="mt-1"
                />
                <div className="flex-1">
                  <label htmlFor={path} className="text-sm font-medium cursor-pointer">
                    {name}
                  </label>
                  <p className="text-xs text-muted-foreground">{path}</p>
                  {formData.folders.includes(path) && (
                    <div className="mt-2">
                      <Input
                        type="number"
                        min="1"
                        placeholder="All documents (leave empty)"
                        value={formData.docCounts[path] || ""}
                        onChange={(e) => handleDocCountChange(path, e.target.value)}
                        disabled={loading}
                        className="max-w-xs"
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        Limit number of documents (optional)
                      </p>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Model Selection */}
        <Card>
          <CardHeader>
            <CardTitle>Humanization Models</CardTitle>
            <CardDescription>
              Select which models to test ({formData.models.length} selected)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Bulk Model Input */}
            <div className="border rounded-md p-4 space-y-3 bg-muted/20">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Bulk Model Selection</label>
                <span className="text-xs text-muted-foreground">Paste OpenAI fine-tune IDs</span>
              </div>
              <textarea
                value={bulkModelInput}
                onChange={(e) => setBulkModelInput(e.target.value)}
                placeholder="ft:gpt-4o-mini-2024-07-18:litero-ai:min-e3-b8-m10-v10:CahO6KtY&#10;ft:gpt-4o-mini-2024-07-18:litero-ai:raw-e5-b16-m08-v10:CaRGQPzO&#10;ft:gpt-4o-mini-2024-07-18:litero-ai:cmp-e4-b24-m12-v15:CahcnBqU"
                className="w-full min-h-[100px] p-2 text-sm font-mono border rounded resize-y"
                disabled={loading}
              />
              <Button
                type="button"
                variant="secondary"
                size="sm"
                onClick={handleBulkModelInput}
                disabled={loading || !bulkModelInput.trim()}
              >
                Add Models from List
              </Button>
            </div>

            {/* Vanilla Models */}
            <div className="border rounded-md">
              <div className="flex items-center justify-between p-3">
                <button
                  type="button"
                  onClick={() => toggleSection("vanilla")}
                  className="flex items-center gap-2 hover:bg-muted/50 flex-1"
                >
                  {expandedSections.vanilla ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                  <span className="font-medium">Vanilla Models</span>
                  <span className="text-sm text-muted-foreground">
                    ({countSelectedInGroup(VANILLA_MODELS)}/{VANILLA_MODELS.length})
                  </span>
                </button>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => selectAllInGroup(VANILLA_MODELS)}
                    disabled={loading}
                    className="text-xs h-7"
                  >
                    Select All
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => deselectAllInGroup(VANILLA_MODELS)}
                    disabled={loading}
                    className="text-xs h-7"
                  >
                    Deselect All
                  </Button>
                </div>
              </div>
              {expandedSections.vanilla && (
                <div className="p-3 pt-0 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                  {VANILLA_MODELS.map((model) => (
                    <div key={model.id} className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id={model.id}
                        checked={isModelSelected(model.id)}
                        onChange={() => handleModelToggle(model.id)}
                        disabled={loading}
                      />
                      <label htmlFor={model.id} className="text-sm cursor-pointer flex-1">
                        {model.name}
                        <span className="text-xs text-muted-foreground ml-1">({model.provider})</span>
                      </label>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Old Finetunes */}
            <div className="border rounded-md">
              <div className="flex items-center justify-between p-3">
                <button
                  type="button"
                  onClick={() => toggleSection("oldFinetunes")}
                  className="flex items-center gap-2 hover:bg-muted/50 flex-1"
                >
                  {expandedSections.oldFinetunes ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                  <span className="font-medium">Old Fine-tunes</span>
                  <span className="text-sm text-muted-foreground">
                    ({countSelectedInGroup(OLD_FINETUNES)}/{OLD_FINETUNES.length})
                  </span>
                </button>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => selectAllInGroup(OLD_FINETUNES)}
                    disabled={loading}
                    className="text-xs h-7"
                  >
                    Select All
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => deselectAllInGroup(OLD_FINETUNES)}
                    disabled={loading}
                    className="text-xs h-7"
                  >
                    Deselect All
                  </Button>
                </div>
              </div>
              {expandedSections.oldFinetunes && (
                <div className="p-3 pt-0 grid grid-cols-1 md:grid-cols-2 gap-2">
                  {OLD_FINETUNES.map((model) => (
                    <div key={model.id} className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id={model.id}
                        checked={isModelSelected(model.id)}
                        onChange={() => handleModelToggle(model.id)}
                        disabled={loading}
                      />
                      <label htmlFor={model.id} className="text-sm cursor-pointer flex-1">
                        {model.name}
                        <span className="text-xs text-muted-foreground ml-1">({model.base})</span>
                      </label>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* New Finetunes */}
            <div className="border rounded-md">
              <div className="flex items-center justify-between p-3">
                <button
                  type="button"
                  onClick={() => toggleSection("newFinetunes")}
                  className="flex items-center gap-2 hover:bg-muted/50 flex-1"
                >
                  {expandedSections.newFinetunes ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                  <span className="font-medium">New Fine-tunes (with Checkpoints)</span>
                  <span className="text-sm text-muted-foreground">
                    (30 models across 5 base models)
                  </span>
                </button>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={selectAllCheckpoints}
                    disabled={loading}
                    className="text-xs h-7"
                  >
                    Select All
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={deselectAllCheckpoints}
                    disabled={loading}
                    className="text-xs h-7"
                  >
                    Deselect All
                  </Button>
                </div>
              </div>
              {expandedSections.newFinetunes && (
                <div className="p-3 pt-0 space-y-3">
                  {Object.entries(NEW_FINETUNES).map(([baseModel, models]) => (
                    <div key={baseModel} className="border rounded-md">
                      <button
                        type="button"
                        onClick={() => toggleBaseModel(baseModel)}
                        className="w-full flex items-center justify-between p-2 hover:bg-muted/50"
                      >
                        <div className="flex items-center gap-2">
                          {expandedBaseModels[baseModel] ? (
                            <ChevronDown className="h-3 w-3" />
                          ) : (
                            <ChevronRight className="h-3 w-3" />
                          )}
                          <span className="text-sm font-medium">{baseModel}</span>
                          <span className="text-xs text-muted-foreground">
                            ({models.length} models)
                          </span>
                        </div>
                      </button>
                      {expandedBaseModels[baseModel] && (
                        <div className="p-2 pt-0 space-y-2">
                          {models.map((model) => (
                            <div
                              key={model.prefix}
                              className="p-2 border rounded bg-muted/20 space-y-2"
                            >
                              <div className="font-mono text-sm font-medium">{model.prefix}</div>
                              <div className="text-xs text-muted-foreground">{model.desc}</div>
                              <div className="flex gap-3 flex-wrap">
                                <label className="flex items-center gap-1.5 cursor-pointer">
                                  <input
                                    type="checkbox"
                                    checked={isModelSelected(model.prefix)}
                                    onChange={() => handleCheckpointToggle(model.prefix, null)}
                                    disabled={loading}
                                  />
                                  <span className="text-sm">Final</span>
                                </label>
                                {model.hasCheckpoints && (
                                  <>
                                    <label className="flex items-center gap-1.5 cursor-pointer">
                                      <input
                                        type="checkbox"
                                        checked={isModelSelected(`${model.prefix}:ckpt1`)}
                                        onChange={() => handleCheckpointToggle(model.prefix, "ckpt1")}
                                        disabled={loading}
                                      />
                                      <span className="text-sm">Checkpoint 1</span>
                                    </label>
                                    <label className="flex items-center gap-1.5 cursor-pointer">
                                      <input
                                        type="checkbox"
                                        checked={isModelSelected(`${model.prefix}:ckpt2`)}
                                        onChange={() => handleCheckpointToggle(model.prefix, "ckpt2")}
                                        disabled={loading}
                                      />
                                      <span className="text-sm">Checkpoint 2</span>
                                    </label>
                                  </>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* DPO Models */}
            <div className="border rounded-md">
              <div className="flex items-center justify-between p-3">
                <button
                  type="button"
                  onClick={() => toggleSection("dpoModels")}
                  className="flex items-center gap-2 hover:bg-muted/50 flex-1"
                >
                  {expandedSections.dpoModels ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                  <span className="font-medium">DPO Models (Batch 2)</span>
                  <span className="text-sm text-muted-foreground">
                    (23 models across 4 base models)
                  </span>
                </button>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={selectAllDpoCheckpoints}
                    disabled={loading}
                    className="text-xs h-7"
                  >
                    Select All
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={deselectAllDpoCheckpoints}
                    disabled={loading}
                    className="text-xs h-7"
                  >
                    Deselect All
                  </Button>
                </div>
              </div>
              {expandedSections.dpoModels && (
                <div className="p-3 pt-0 space-y-3">
                  {Object.entries(DPO_MODELS).map(([baseModel, models]) => (
                    <div key={baseModel} className="border rounded-md">
                      <button
                        type="button"
                        onClick={() => toggleDpoBaseModel(baseModel)}
                        className="w-full flex items-center justify-between p-2 hover:bg-muted/50"
                      >
                        <div className="flex items-center gap-2">
                          {expandedDpoBaseModels[baseModel] ? (
                            <ChevronDown className="h-3 w-3" />
                          ) : (
                            <ChevronRight className="h-3 w-3" />
                          )}
                          <span className="text-sm font-medium">{baseModel}</span>
                          <span className="text-xs text-muted-foreground">
                            ({models.length} models)
                          </span>
                        </div>
                        <div className="flex gap-2">
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              selectAllDpoBase(baseModel);
                            }}
                            disabled={loading}
                            className="text-xs h-7"
                          >
                            Select All
                          </Button>
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              deselectAllDpoBase(baseModel);
                            }}
                            disabled={loading}
                            className="text-xs h-7"
                          >
                            Deselect All
                          </Button>
                        </div>
                      </button>
                      {expandedDpoBaseModels[baseModel] && (
                        <div className="p-2 pt-0 space-y-2">
                          {models.map((model) => (
                            <div
                              key={model.prefix}
                              className="p-2 border rounded bg-muted/20 space-y-2"
                            >
                              <div className="font-mono text-sm font-medium">{model.prefix}</div>
                              <div className="text-xs text-muted-foreground">{model.desc}</div>
                              <div className="flex gap-3 flex-wrap">
                                <label className="flex items-center gap-1.5 cursor-pointer">
                                  <input
                                    type="checkbox"
                                    checked={isModelSelected(model.prefix)}
                                    onChange={() => handleCheckpointToggle(model.prefix, null)}
                                    disabled={loading}
                                  />
                                  <span className="text-sm">Final</span>
                                </label>
                                {model.hasCheckpoints && (
                                  <>
                                    <label className="flex items-center gap-1.5 cursor-pointer">
                                      <input
                                        type="checkbox"
                                        checked={isModelSelected(`${model.prefix}:ckpt1`)}
                                        onChange={() => handleCheckpointToggle(model.prefix, "ckpt1")}
                                        disabled={loading}
                                      />
                                      <span className="text-sm">Checkpoint 1</span>
                                    </label>
                                    {(model.hasCheckpoint2 !== false) && (
                                      <label className="flex items-center gap-1.5 cursor-pointer">
                                        <input
                                          type="checkbox"
                                          checked={isModelSelected(`${model.prefix}:ckpt2`)}
                                          onChange={() => handleCheckpointToggle(model.prefix, "ckpt2")}
                                          disabled={loading}
                                        />
                                        <span className="text-sm">Checkpoint 2</span>
                                      </label>
                                    )}
                                  </>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* DPO H40 Models (based on hum40-naive-auto) */}
            <div className="border rounded-md">
              <div className="flex items-center justify-between p-3">
                <button
                  type="button"
                  onClick={() => toggleSection("dpoH40Models")}
                  className="flex items-center gap-2 hover:bg-muted/50 flex-1"
                >
                  {expandedSections.dpoH40Models ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                  <span className="font-medium">DPO H40 Models</span>
                  <span className="text-sm text-muted-foreground">
                    (Based on hum40-naive-auto, {DPO_H40_MODELS.length} models)
                  </span>
                </button>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={selectAllDpoH40}
                    disabled={loading}
                    className="text-xs h-7"
                  >
                    Select All
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={deselectAllDpoH40}
                    disabled={loading}
                    className="text-xs h-7"
                  >
                    Deselect All
                  </Button>
                </div>
              </div>
              {expandedSections.dpoH40Models && (
                <div className="p-3 pt-0 space-y-2">
                  {DPO_H40_MODELS.map((model) => (
                    <div
                      key={model.prefix}
                      className="p-2 border rounded bg-muted/20 space-y-2"
                    >
                      <div className="font-mono text-sm font-medium">{model.prefix}</div>
                      <div className="text-xs text-muted-foreground">{model.desc}</div>
                      <div className="flex gap-3 flex-wrap">
                        <label className="flex items-center gap-1.5 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={isModelSelected(model.prefix)}
                            onChange={() => handleCheckpointToggle(model.prefix, null)}
                            disabled={loading}
                          />
                          <span className="text-sm">Final</span>
                        </label>
                        {model.hasCheckpoints && (
                          <>
                            <label className="flex items-center gap-1.5 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={isModelSelected(`${model.prefix}:ckpt1`)}
                                onChange={() => handleCheckpointToggle(model.prefix, "ckpt1")}
                                disabled={loading}
                              />
                              <span className="text-sm">Checkpoint 1</span>
                            </label>
                            {model.hasCheckpoint2 && (
                              <label className="flex items-center gap-1.5 cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={isModelSelected(`${model.prefix}:ckpt2`)}
                                  onChange={() => handleCheckpointToggle(model.prefix, "ckpt2")}
                                  disabled={loading}
                                />
                                <span className="text-sm">Checkpoint 2</span>
                              </label>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* User-Style Humanization Test Models */}
            <div className="border rounded-md">
              <div className="flex items-center justify-between p-3">
                <button
                  type="button"
                  onClick={() => toggleSection("userStyle")}
                  className="flex items-center gap-2 hover:bg-muted/50 flex-1"
                >
                  {expandedSections.userStyle ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                  <span className="font-medium">User-Style Humanization Tests</span>
                  <span className="text-sm text-muted-foreground">
                    ({USER_STYLE_MODELS.filter((m) => formData.models.includes(m.id)).length}/{USER_STYLE_MODELS.length})
                  </span>
                </button>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => selectAllInGroup(USER_STYLE_MODELS)}
                    disabled={loading}
                    className="text-xs h-7"
                  >
                    Select All
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => deselectAllInGroup(USER_STYLE_MODELS)}
                    disabled={loading}
                    className="text-xs h-7"
                  >
                    Deselect All
                  </Button>
                </div>
              </div>
                <div className="p-3 pt-0 space-y-3">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {USER_STYLE_MODELS.map((model) => (
                      <div key={model.id} className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          id={model.id}
                          checked={isModelSelected(model.id)}
                          onChange={() => handleModelToggle(model.id)}
                          disabled={loading}
                        />
                        <label htmlFor={model.id} className="text-sm cursor-pointer flex-1">
                          {model.name}
                        </label>
                      </div>
                    ))}
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Writing profile (paste from Writing Profile Lab)</label>
                    <textarea
                      value={formData.userStyleProfile}
                      onChange={(e) => setFormData({ ...formData, userStyleProfile: e.target.value })}
                      placeholder="Paste the profile JSON or text you want to condition on."
                      className="w-full min-h-[140px] p-2 text-sm font-mono border rounded resize-y"
                      disabled={loading}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Profile injection mode</label>
                    <div className="flex gap-4 text-sm">
                      <label className="flex items-center gap-2">
                        <input
                          type="radio"
                          name="userStyleMode"
                          value="user"
                          checked={formData.userStyleProfileMode === "user"}
                          onChange={(e) => setFormData({ ...formData, userStyleProfileMode: e.target.value })}
                          disabled={loading}
                        />
                        User prompt
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="radio"
                          name="userStyleMode"
                          value="system"
                          checked={formData.userStyleProfileMode === "system"}
                          onChange={(e) => setFormData({ ...formData, userStyleProfileMode: e.target.value })}
                          disabled={loading}
                        />
                        System prompt
                      </label>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 pt-2">
                    <input
                      type="checkbox"
                      id="useStyleAdherence"
                      checked={formData.useStyleAdherence}
                      onChange={(e) => setFormData({ ...formData, useStyleAdherence: e.target.checked })}
                      disabled={loading}
                    />
                    <label htmlFor="useStyleAdherence" className="text-sm font-medium cursor-pointer">
                      Evaluate style adherence (using Gemini 2.5 Flash)
                    </label>
                  </div>
                </div>
            </div>
          </CardContent>
        </Card>

        {/* Detection & Analysis Options */}
        <Card>
          <CardHeader>
            <CardTitle>Detection & Analysis Options</CardTitle>
            <CardDescription>Configure detection and quality checking</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="includeDocMode"
                checked={formData.includeDocMode}
                onChange={(e) => setFormData({ ...formData, includeDocMode: e.target.checked })}
                disabled={loading}
              />
              <label htmlFor="includeDocMode" className="text-sm font-medium cursor-pointer">
                Include Document-Level Rewriting
              </label>
            </div>

            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="useGptzero"
                checked={formData.useGptzero}
                onChange={(e) => setFormData({ ...formData, useGptzero: e.target.checked })}
                disabled={loading}
              />
              <label htmlFor="useGptzero" className="text-sm font-medium cursor-pointer">
                Use GPTZero Detection
              </label>
            </div>

            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="useSapling"
                checked={formData.useSapling}
                onChange={(e) => setFormData({ ...formData, useSapling: e.target.checked })}
                disabled={loading}
              />
              <label htmlFor="useSapling" className="text-sm font-medium cursor-pointer">
                Use Sapling Detection
              </label>
            </div>
          </CardContent>
        </Card>

        {/* Submit */}
        <div className="flex gap-3">
          <Button type="submit" disabled={loading} className="min-w-32">
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <PlayCircle className="h-4 w-4 mr-2" />
                Start Run
              </>
            )}
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={() => navigate("/")}
            disabled={loading}
          >
            Cancel
          </Button>
        </div>
      </form>
    </div>
  );
}
