import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Input } from "../components/ui/Input";
import { Alert, AlertDescription, AlertTitle, AlertIcons } from "../components/ui/Alert";
import { PlayCircle, Loader2, ChevronDown, ChevronRight } from "lucide-react";

// Vanilla models (OpenAI, Claude, Gemini)
const VANILLA_MODELS = [
  { id: "gpt-4.1", name: "GPT-4.1", provider: "OpenAI" },
  { id: "gpt-4.1-mini", name: "GPT-4.1 Mini", provider: "OpenAI" },
  { id: "gpt-4o", name: "GPT-4o", provider: "OpenAI" },
  { id: "claude-sonnet-4", name: "Claude Sonnet 4", provider: "Anthropic" },
  { id: "claude-sonnet-3.7", name: "Claude Sonnet 3.7", provider: "Anthropic" },
  { id: "claude-haiku-3.5", name: "Claude Haiku 3.5", provider: "Anthropic" },
  { id: "gemini-2.0-flash", name: "Gemini 2.0 Flash", provider: "Google" },
  { id: "gemini-2.0-flash-lite", name: "Gemini 2.0 Flash Lite", provider: "Google" },
  { id: "gemini-2.5-flash", name: "Gemini 2.5 Flash", provider: "Google" },
  { id: "gemini-2.5-pro", name: "Gemini 2.5 Pro", provider: "Google" },
];

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
    newFinetunes: true,
  });

  const [expandedBaseModels, setExpandedBaseModels] = useState({
    "gpt-4o-mini": true,
    "gpt-4.1-mini": false,
    "gpt-4.1": false,
    "gpt-4.1-nano": false,
    "gpt-4o": false,
  });

  const [formData, setFormData] = useState({
    runName: `run_${Date.now()}`,
    folders: ["data/ai_texts"],
    models: ["gpt-4.1"],
    iterations: 5,
    docCounts: {},
    includeDocMode: true,
    useGptzero: true,
    useSapling: true,
  });

  const toggleSection = (section) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const toggleBaseModel = (baseModel) => {
    setExpandedBaseModels((prev) => ({ ...prev, [baseModel]: !prev[baseModel] }));
  };

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
      const jobData = {
        run_name: formData.runName,
        folders: formData.folders,
        models: formData.models,
        iterations: formData.iterations,
        doc_counts: formData.docCounts,
        include_doc_mode: formData.includeDocMode,
        use_gptzero: formData.useGptzero,
        use_sapling: formData.useSapling,
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
            {/* Vanilla Models */}
            <div className="border rounded-md">
              <button
                type="button"
                onClick={() => toggleSection("vanilla")}
                className="w-full flex items-center justify-between p-3 hover:bg-muted/50"
              >
                <div className="flex items-center gap-2">
                  {expandedSections.vanilla ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                  <span className="font-medium">Vanilla Models</span>
                  <span className="text-sm text-muted-foreground">
                    ({countSelectedInGroup(VANILLA_MODELS)}/{VANILLA_MODELS.length})
                  </span>
                </div>
              </button>
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
              <button
                type="button"
                onClick={() => toggleSection("oldFinetunes")}
                className="w-full flex items-center justify-between p-3 hover:bg-muted/50"
              >
                <div className="flex items-center gap-2">
                  {expandedSections.oldFinetunes ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                  <span className="font-medium">Old Fine-tunes</span>
                  <span className="text-sm text-muted-foreground">
                    ({countSelectedInGroup(OLD_FINETUNES)}/{OLD_FINETUNES.length})
                  </span>
                </div>
              </button>
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
              <button
                type="button"
                onClick={() => toggleSection("newFinetunes")}
                className="w-full flex items-center justify-between p-3 hover:bg-muted/50"
              >
                <div className="flex items-center gap-2">
                  {expandedSections.newFinetunes ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                  <span className="font-medium">New Fine-tunes (with Checkpoints)</span>
                  <span className="text-sm text-muted-foreground">
                    (30 models across 5 base models)
                  </span>
                </div>
              </button>
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
